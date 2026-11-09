# DNINet

Code and trained model weights for **DNINet: Depth Completion with Multi-Scale Depth-Normal Interaction**.

This repository contains the DNINet implementations for KITTI and NYUv2, the CUDA operators they require, a single-GPU evaluation script, and two model-only checkpoints. It does not contain baseline implementations or datasets.

## Contents

| Path | Purpose |
| --- | --- |
| kitti/ | KITTI DNINet model, data loader, losses, training and evaluation code |
| nyu/ | NYUv2 DNINet model, data loader, losses, training and evaluation code |
| ops/BpOps/, ops/DCNv2/ | CUDA operator source used by DNINet |
| scripts/evaluate.py | Single-GPU evaluation and profiling |
| weights/kitti_ema11.pt | Model state used by the original KITTI submission script |
| weights/nyu_ema.pt | NYUv2-trained model state used in the response experiments |

The weights are stored with Git LFS. Run **git lfs pull** after cloning if the files are only small pointer files. See weights/README.md for checkpoint provenance and metric limits.

## Environment

The code was exercised with Python 3.11, PyTorch 2.5.1+cu121, torchvision 0.20.1+cu121, and CUDA 12.1. A CUDA-capable GPU is needed for the custom operators.

~~~bash
conda env create -f environment.yml
conda activate dninet
python -m pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
export CUDA_HOME="$CONDA_PREFIX"
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
(cd ops/BpOps && MAX_JOBS=2 python setup.py build_ext --inplace)
(cd ops/DCNv2 && MAX_JOBS=2 python setup.py build_ext --inplace)
~~~

## Data

Datasets and pseudo-normal supervision are not redistributed. Put the prepared datasets under a common root, for example data/KITTIDC and data/NYUBIG, or pass another root with --data-root. The KITTI loader expects the original split directories and index files (img.txt, raw.txt, gt.txt, and related files); the NYUv2 loader expects HDF5 samples in NYUBIG/{train,val}/h5_set. Training also expects precomputed pseudo-normal maps in each dataset's normal directory. Field names and transformations are defined in kitti/dataset.py and nyu/dataset_nyu.py.

## Evaluation

From the repository root, after preparing the data:

~~~bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py \
  --source kitti --target kitti \
  --checkpoint weights/kitti_ema11.pt \
  --data-root /path/to/data --kitti-split val_rel \
  --output results/kitti

CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py \
  --source nyu --target nyu \
  --checkpoint weights/nyu_ema.pt \
  --data-root /path/to/data \
  --output results/nyu
~~~

For a one-image smoke test, add --limit 1 --skip-profile. The script writes summary.json and per_image.csv to the output directory. It uses one forward pass by default and computes per-image means at valid ground-truth pixels. The KITTI online test result in the manuscript uses the official hidden test split; it cannot be reproduced from the local validation split.

## Training code

The original training entry points are kitti/train.py and nyu/train.py. They use PyTorch distributed initialization and can be launched from their respective directories with torchrun --standalone --nproc_per_node=1 train.py. Set DNINET_DATA_ROOT to the parent of KITTIDC and NYUBIG, and DNINET_NYU_ROOT to the NYUv2 directory. These scripts preserve the experimental training logic and checkpoint continuation behavior; the released model-only weights are intended for inference, not optimizer-state resumption.

## External code

The dataset loader adapted from CompletionFormer and the DCNv2/CUDA operator implementations are included only as dependencies required to run DNINet. No CompletionFormer, NLSPN, BP-Net, or other baseline model is included. See THIRD_PARTY.md.
