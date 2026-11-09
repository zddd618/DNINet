"""Single-GPU DNINet evaluation, profiling and diagnostic export.

Uses the original model and dataset implementations; no DDP, no silent checkpoint
fallback. Metrics are per-image means to match the project's evaluation scripts.
"""
import argparse
import csv
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=ROOT / "data")
    p.add_argument("--source", choices=("kitti", "nyu"), required=True)
    p.add_argument("--target", choices=("kitti", "nyu"), required=True)
    p.add_argument("--kitti-split", choices=("val", "val_rel"), default="val_rel")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--limit", type=int, default=0, help="0 evaluates the whole set")
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeat", type=int, default=20)
    p.add_argument("--skip-profile", action="store_true", help="Skip warm-up and timed passes; use for parallel full validation")
    p.add_argument("--save-npz", type=int, default=0, help="Save this many examples")
    p.add_argument("--iterations", type=int, default=0, help="0 uses trained value")
    p.add_argument("--flip-tta", action="store_true", help="Average original and horizontally flipped inference")
    p.add_argument("--target-depth-range-m", type=float, default=0.0,
                   help="For NYUv2->KITTI only: feed target metres/R to the NYUv2 model and restore output by R/10; 0 keeps raw metric transfer")
    p.add_argument("--clip-sparse-to-range", action="store_true",
                   help="After target-range scaling, cap normalized sparse input at 1")
    return p.parse_args()


def load_model(source, path, device, iterations):
    source_dir = ROOT / ("kitti" if source == "kitti" else "nyu")
    sys.path.insert(0, str(source_dir))
    sys.path.insert(0, str(ROOT / "ops" / "BpOps"))
    sys.path.insert(0, str(ROOT / "ops" / "DCNv2"))
    import model
    import _ext
    net = model.Completion3D()
    state = torch.load(path, map_location="cpu", weights_only=False)
    state = state.get("model_state_dict", state)
    state = {k.removeprefix("module."): v for k, v in state.items()}
    net.load_state_dict(state, strict=True)
    if iterations:
        net.iters = iterations
    net.to(device).eval()
    return net, source_dir, str(Path(_ext.__file__).resolve())


def samples(target, source_dir, start_index, kitti_split, data_root):
    if target == "nyu":
        sys.path.insert(0, str(ROOT / "nyu"))
        from dataset_nyu import NYU
        ds = NYU("val", path=str(data_root / "NYUBIG"), num_sample=500,
                 num_mask=1, scale_kcam=True, rand_scale=False)
        for i in range(start_index, len(ds)):
            inputs, gt, kcam, *_ = ds[i]
            yield i, [x.unsqueeze(0) for x in inputs], gt.unsqueeze(0), torch.as_tensor(kcam).unsqueeze(0)
    else:
        kitti_dir = ROOT / "kitti"
        if source_dir != kitti_dir:
            for name in list(sys.modules):
                if name == "deeptool" or name.startswith("deeptool."):
                    del sys.modules[name]
        sys.path.insert(0, str(kitti_dir))
        import deeptool.lydata.utils as utils
        utils.base_path = str(data_root) + "/"
        spec = importlib.util.spec_from_file_location("kitti_response_dataset", ROOT / "kitti" / "dataset.py")
        dataset = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dataset)
        ds = dataset.G2_data(kitti_split, ["KITTIDC"])
        for i in range(start_index, len(ds)):
            inputs, gt, kcam, _ = ds[i]
            yield i, [x.unsqueeze(0) for x in inputs], gt.unsqueeze(0), torch.as_tensor(kcam).unsqueeze(0)


def inference(net, inputs, kcam, source, flip_tta=False):
    if source == "nyu":
        out = net(inputs[0], inputs[1], inputs[2], kcam)
        if flip_tta:
            kflip = kcam.clone(); kflip[:, 0, 2] = inputs[0].shape[-1] - 1 - kflip[:, 0, 2]
            flipped = net(*(torch.flip(x, [-1]) for x in inputs[:3]), kflip)
    else:
        prefill = inputs[3] if len(inputs) > 3 else inputs[2][:, :1]
        out = net(inputs[0], inputs[1], inputs[2], prefill, kcam)
        if flip_tta:
            kflip = kcam.clone(); kflip[:, 0, 2] = inputs[0].shape[-1] - 1 - kflip[:, 0, 2]
            flipped = net(*(torch.flip(x, [-1]) for x in (inputs[0], inputs[1], inputs[2], prefill)), kflip)
    if flip_tta:
        out["prediction"][-1] = (out["prediction"][-1] + torch.flip(flipped["prediction"][-1], [-1])) / 2
    return out


def adapt_inputs(inputs, source, target):
    """Map target dataset tensors into the source model's training normalization."""
    if source == target:
        return inputs
    imagenet_mean = inputs[0].new_tensor((.485, .456, .406))[None, :, None, None]
    imagenet_std = inputs[0].new_tensor((.229, .224, .225))[None, :, None, None]
    kitti_mean = inputs[0].new_tensor((90.995/255, 96.2278/255, 94.3213/255))[None, :, None, None]
    kitti_std = inputs[0].new_tensor((79.2382/255, 80.5267/255, 82.1483/255))[None, :, None, None]
    if source == "nyu":
        inputs[0] = ((inputs[0] * kitti_std + kitti_mean) - imagenet_mean) / imagenet_std
        factor = 12.8  # KITTI input depth in metres/128; NYU expects metres/10
    else:
        inputs[0] = ((inputs[0] * imagenet_std + imagenet_mean) - kitti_mean) / kitti_std
        factor = 1 / 12.8
    inputs[1][:, :1] *= factor
    inputs[2] *= factor
    if source == "nyu" and target == "kitti":
        # The NYUv2 model's third input is the same sparse depth as its
        # second input, as in dataset_nyu.py. KITTI's third input is instead
        # an outlier-filtered sparse map, so do not forward it here.
        inputs[2] = inputs[1][:, :1].clone()
    if len(inputs) > 3:
        inputs[3] *= factor
    return inputs


def to_metric(pred, gt, target, output_scale=1.0):
    if target == "nyu":
        return pred[:, :, 14:242, 8:312], gt[:, :, 14:242, 8:312] * 10.0
    return pred * ((65535.0 / 65536.0) * output_scale), gt * (65535.0 / 256.0)


def metrics(pred, gt):
    valid = torch.isfinite(gt) & (gt > 1e-6) & torch.isfinite(pred)
    if not valid.any():
        return None
    prediction = pred[valid]
    truth = gt[valid]
    delta = prediction - truth
    absolute = delta.abs()
    # A shared 1 mm positive floor makes inverse/log metrics finite when a
    # model emits zero or negative depth. RMSE, MAE, AbsRel, and SqRel use
    # the original prediction, preserving the earlier protocol.
    floor_m = 1e-3
    positive = prediction.clamp_min(floor_m)
    inverse_delta = positive.reciprocal() - truth.clamp_min(floor_m).reciprocal()
    ratio = torch.maximum(positive / truth, truth / positive)
    log_delta = torch.log(positive) - torch.log(truth)
    silog_variance = (log_delta.square().mean() - log_delta.mean().square()).clamp_min(0)
    return {"rmse_m": torch.sqrt(delta.square().mean()).item(),
            "mae_m": absolute.mean().item(),
            "irmse_1_km": (1000 * torch.sqrt(inverse_delta.square().mean())).item(),
            "imae_1_km": (1000 * inverse_delta.abs().mean()).item(),
            "absrel": (absolute / truth).mean().item(),
            "sqrel_m": (delta.square() / truth).mean().item(),
            "silog_pct": (100 * torch.sqrt(silog_variance)).item(),
            "delta1": (ratio < 1.25).float().mean().item(),
            "delta2": (ratio < 1.25 ** 2).float().mean().item(),
            "delta3": (ratio < 1.25 ** 3).float().mean().item(),
            "d102": (ratio < 1.02).float().mean().item(),
            "d105": (ratio < 1.05).float().mean().item(),
            "d110": (ratio < 1.10).float().mean().item(),
            "depth_floor_pixels": int((prediction < floor_m).sum()),
            "valid_pixels": int(valid.sum())}


def profile(net, inputs, kcam, source, device, warmup, repeat, flip_tta):
    if device.type != "cuda":
        return {}
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for _ in range(warmup):
            inference(net, inputs, kcam, source, flip_tta)
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(repeat):
            inference(net, inputs, kcam, source, flip_tta)
        torch.cuda.synchronize(device)
    sec = (time.perf_counter() - start) / repeat
    return {"latency_ms": 1000 * sec, "fps": 1 / sec,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30}


def main():
    args = parse_args()
    if args.target_depth_range_m and (args.target_depth_range_m <= 0 or (args.source, args.target) != ("nyu", "kitti")):
        raise ValueError("--target-depth-range-m requires a positive range and NYUv2->KITTI")
    if args.clip_sparse_to_range and not args.target_depth_range_m:
        raise ValueError("--clip-sparse-to-range requires --target-depth-range-m")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    os.environ["DNINET_DATA_ROOT"] = str(args.data_root.resolve())
    os.environ["DNINET_NYU_ROOT"] = str((args.data_root / "NYUBIG").resolve())
    device = torch.device(args.device)
    net, source_dir, dcn_backend = load_model(args.source, args.checkpoint, device, args.iterations)
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    timing = {}
    with torch.inference_mode():
        for i, inputs, gt, kcam in samples(args.target, source_dir, args.start_index, args.kitti_split, args.data_root):
            inputs = [x.to(device) for x in inputs]
            inputs = adapt_inputs(inputs, args.source, args.target)
            if args.target_depth_range_m:
                # The NYUv2 model was trained on metres/10 and returns metres
                # after multiplying its internal depth by 10. For a target
                # range R, feed metres/R and restore its metric output by R/10.
                ratio = 10.0 / args.target_depth_range_m
                inputs[1][:, :1].mul_(ratio)
                inputs[2].mul_(ratio)
                if args.clip_sparse_to_range:
                    inputs[1][:, :1].clamp_(max=1.0)
                    inputs[2].clamp_(max=1.0)
            gt, kcam = gt.to(device), kcam.to(device)
            if not timing and not args.skip_profile:
                timing = profile(net, inputs, kcam, args.source, device, args.warmup, args.repeat, args.flip_tta)
            out = inference(net, inputs, kcam, args.source, args.flip_tta)
            pred, gt_m = to_metric(out["prediction"][-1], gt, args.target,
                                   output_scale=args.target_depth_range_m / 10.0 if args.target_depth_range_m else 1.0)
            item = metrics(pred, gt_m)
            if item is not None:
                rows.append({"index": i, **item})
            if len(rows) <= args.save_npz:
                rgb = inputs[0][0].cpu().numpy()
                np.savez_compressed(args.output / f"sample_{i:04d}.npz",
                                    rgb=rgb, sparse=inputs[1][0, :1].cpu().numpy(),
                                    pred=pred[0, 0].cpu().numpy(), gt=gt_m[0, 0].cpu().numpy(),
                                    normal=out["prediction_normal"][0].cpu().numpy(),
                                    depth_steps=np.stack([x[0, 0].cpu().numpy() for x in out["predictions_list"]]),
                                    normal_steps=np.stack([x[0].cpu().numpy() for x in out["predictions_normal_list"]]))
            if args.limit and len(rows) >= args.limit:
                break
    if not rows:
        raise RuntimeError("No valid samples")
    with (args.output / "per_image.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    result = {"source": args.source, "target": args.target,
              "checkpoint": str(args.checkpoint.resolve()), "samples": len(rows),
              "limit": args.limit, "full_validation": len(rows) == (1000 if args.target == "kitti" else 654),
              "start_index": args.start_index,
              "kitti_split": args.kitti_split if args.target == "kitti" else None,
              "target_depth_range_m": args.target_depth_range_m or None,
              "clip_sparse_to_range": args.clip_sparse_to_range,
              "protocol": ("horizontal flip TTA" if args.flip_tta else "single forward, no flip TTA") + "; per-image mean; GT valid pixels",
              "flip_tta": args.flip_tta,
              "dcn_backend": dcn_backend,
              "rmse_mm": 1000 * float(np.mean([r["rmse_m"] for r in rows])),
              "mae_mm": 1000 * float(np.mean([r["mae_m"] for r in rows])),
              "metric_definitions": "Per-image means at the same valid-GT/finite-prediction pixels; inverse/log/threshold metrics use a 0.001 m positive prediction floor; iRMSE and iMAE in 1/km; SILog in percent",
              **{name: float(np.mean([r[name] for r in rows])) for name in ("irmse_1_km", "imae_1_km", "absrel", "sqrel_m", "silog_pct", "delta1", "delta2", "delta3", "d102", "d105", "d110")},
              "depth_floor_pixels": int(sum(r["depth_floor_pixels"] for r in rows)),
              "parameters_m": sum(p.numel() for p in net.parameters()) / 1e6,
              "iterations": net.iters, **timing}
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
