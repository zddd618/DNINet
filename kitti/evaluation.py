import numpy as np
import dataset as data
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import random
import model
import os
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
from collections import OrderedDict
import importlib
import build_essential
import imageio
import time

from deeptool import lytest, lydebug, lyvision

alpha = 0.000001

def val_it(local_rank, proc_it="test", name="val", dataset_name="DIODE"):

    G2_model = model.Completion3D()
    # G2_model.eval()
    G2_model.to(device)
    # lydebug.check_model(G2_model)
    G2_model = torch.nn.parallel.DistributedDataParallel(G2_model, 
                device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)  # 这句加载到多GPU上

    dataset_val = data.G2_data(name, [dataset_name])
    # lydebug.check_dataset(dataset_val)
    datasampler_val = DistributedSampler(dataset_val)
    dataloader_val = DataLoader(dataset_val, batch_size=1, sampler=datasampler_val,
                                num_workers=5, pin_memory=True, drop_last=False)

    if os.path.exists(os.environ.get("DNINET_KITTI_WEIGHTS", "../weights/kitti_ema11.pt")):
        print("Load checkpoint ...")
        checkpoint = torch.load(os.environ.get("DNINET_KITTI_WEIGHTS", "../weights/kitti_ema11.pt"), map_location="cpu")
        G2_model.load_state_dict(checkpoint["model_state_dict"])

    # model_paths = ['/root/lanyun-tmp/checkpoint/encode-cat/checkpoint_ema8.pt', '/root/lanyun-tmp/checkpoint/encode-cat/checkpoint_ema12.pt', '/root/lanyun-tmp/checkpoint/encode-cat/checkpoint_ema14.pt']

    # model_paths = ['/root/lanyun-tmp/checkpoint/encode-cat/checkpoint8.pt','/root/lanyun-tmp/checkpoint/encode-cat/checkpoint9.pt','/root/lanyun-tmp/checkpoint/encode-cat/checkpoint10.pt','/root/lanyun-tmp/checkpoint/encode-cat/checkpoint11.pt','/root/lanyun-tmp/checkpoint/encode-cat/checkpoint12.pt']

    # models = [torch.load(path, map_location="cpu")["model_state_dict"] for path in model_paths]
    
    # average_model = models[0].copy()

    # for key in average_model.keys():
    #     average_model[key] = torch.mean(torch.stack([model[key] for model in models]), dim=0)

    RMSE_loss = torch.tensor(0.).to(device)
    ARE_loss = torch.tensor(0.).to(device)
    MAE_loss = torch.tensor(0.).to(device)
    num_all = torch.tensor(0).to(device)

    record = {}

    # G2_model.load_state_dict(average_model)

    G2_model.eval()

    if dist.get_rank() == 0:
        print("Start Val : val dataset: " + dataset_name + " val name: " + name + " ...")

    with torch.no_grad():
        loop = tqdm(dataloader_val, desc="Val")
        for i, (input, gt, Kcam, Kcam_full_path) in enumerate(loop):
            # lydebug.check_tensor(img)
            # lydebug.check_tensor(gt)
            # lydebug.check_tensor(img_ori)
            # lydebug.check_tensor(gt_ori)

            input = [input[0].to(device), input[1].to(device), input[2].to(device), input[3].to(device)]
            gt = gt.to(device)
            Kcam = Kcam.to(device)

            # pred = G2_model(input[0], input[1], input[2], input[3], Kcam)
            # pred = pred["prediction"][-1]

            record_time = time.time()
            pred = G2_model(input[0], input[1], input[2], input[3], Kcam)
            print("Inference time: ", time.time() - record_time)
            pred_2 = pred["prediction"][-1]

            Kcam[:, 0, 2]=1215.0-Kcam[:, 0, 2]
            pred = G2_model(torch.flip(input[0], [3]), torch.flip(input[1], [3]), torch.flip(input[2], [3]), torch.flip(input[3], [3]), Kcam)
            pred_1 = torch.flip(pred["prediction"][-1], [3])

            pred = (pred_1 + pred_2) / 2.0

            # print(torch.max(input[2]*65535.0/512.0))
            # print(torch.max(input[1][:,:1]*65535.0/512.0))
            # print(torch.max(gt*65535.0/256.0))
            # pred = torch.clip(pred, 0., 80.0)

            # pred = pred * build_essential.coefficients.to(device)

            RMSE_loss += lytest.RMSE_tensor(pred*65535.0/65536.0, gt*65535.0/256.0)
            ARE_loss += lytest.ARE_tensor(pred*65535.0/65536.0, gt*65535.0/256.0)
            MAE_loss += lytest.MAE_tensor(pred*65535.0/65536.0, gt*65535.0/256.0)

            # pred = pred.cpu().clamp(0., 1.)

            # lyvision.vision_color(pred/256.0, name="pred.png", path="./saved/test/")
            # lyvision.vision_color(gt, name="gt.png", path="./saved/test/")
            # lyvision.vision_rgb(input[:, :3, :, :], name="img.png", path="./saved/test/")

            num_all += 1
            
            loop.set_description('Val')
            loop.set_postfix(RMSE_loss=RMSE_loss.item() / num_all.item(), ARE_loss=ARE_loss.item() / num_all.item(),
                            MAE_loss=MAE_loss.item() / num_all.item())
                
    dist.all_reduce(RMSE_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(ARE_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(MAE_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(num_all, op=dist.ReduceOp.SUM)

    record[dataset_name + "_" + name] = {'RMSE_loss':RMSE_loss.item() / num_all.item(), 'ARE_loss':ARE_loss.item() / num_all.item(),
                                         'MAE_loss':MAE_loss.item() / num_all.item()}
    
    if dist.get_rank() == 0:
        lytest.save_result(record, name=dataset_name+"_"+name, path="./saved/"+proc_it+"/")

        print("End Val , result saved in " + proc_it + " ...")


def val_submit(local_rank, proc_it="test", name="val", dataset_name="DIODE"):

    G2_model = model.Completion3D()
    G2_model.eval()
    G2_model.to(device)
    # lydebug.check_model(G2_model)
    G2_model = torch.nn.parallel.DistributedDataParallel(G2_model, 
                device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)  # 这句加载到多GPU上

    dataset_val = data.G2_data(name, [dataset_name])
    # lydebug.check_dataset(dataset_val)
    datasampler_val = DistributedSampler(dataset_val)
    dataloader_val = DataLoader(dataset_val, batch_size=1, sampler=datasampler_val,
                                num_workers=5, pin_memory=True, drop_last=False)

    if os.path.exists(os.environ.get("DNINET_KITTI_WEIGHTS", "../weights/kitti_ema11.pt")):
        print("Load checkpoint ...")
        checkpoint = torch.load(os.environ.get("DNINET_KITTI_WEIGHTS", "../weights/kitti_ema11.pt"), map_location="cpu")
        G2_model.load_state_dict(checkpoint["model_state_dict"])

    # RMSE_loss = torch.tensor(0.).to(device)
    # ARE_loss = torch.tensor(0.).to(device)
    # MAE_loss = torch.tensor(0.).to(device)
    # num_all = torch.tensor(0).to(device)

    # record = {}

    os.makedirs("./saved/submit", exist_ok=True)
    print("Submit dir created ...")

    if dist.get_rank() == 0:
        print("Start Val : val dataset: " + dataset_name + " val name: " + name + " ...")

    with torch.no_grad():
        loop = tqdm(dataloader_val, desc="Val")
        for i, (input, jpg_name, Kcam) in enumerate(loop):
            # lydebug.check_tensor(img)
            # lydebug.check_tensor(gt)
            # lydebug.check_tensor(img_ori)
            # lydebug.check_tensor(gt_ori)

            input = [input[0].to(device), input[1].to(device), input[2].to(device), input[3].to(device)]
            Kcam = Kcam.to(device)

            pred = G2_model(input[0], input[1], input[2], input[3], Kcam)
            pred_2 = pred["prediction"][-1]

            # Kcam[0, 2]=1215.0-Kcam[0, 2]
            pred = G2_model(torch.flip(input[0], [3]), torch.flip(input[1], [3]), torch.flip(input[2], [3]), torch.flip(input[3], [3]), Kcam)
            pred_1 = torch.flip(pred["prediction"][-1], [3])

            pred = (pred_1 + pred_2) / 2.0

            save_path = "./saved/submit/" + jpg_name[0]
            pred_np = pred[0][0].cpu().numpy()
            pred_np = pred_np * 65535. / 65536.
            pred_np = np.clip(pred_np, 0, 80.0)
            pred_np = pred_np * 256.0
            pred_np = pred_np.astype(np.uint16)
            pred_np = np.pad(pred_np, ((352 - 256, 0), (0, 0)), 'constant', constant_values=0)
            for i in range(96):
                pred_np[i] = pred_np[96]
            imageio.imwrite(save_path, pred_np)

            # RMSE_loss += lytest.RMSE_tensor(pred * 65535. / 65536., gt * 65535. / 256.)
            # ARE_loss += lytest.ARE_tensor(pred * 65535. / 65536., gt * 65535. / 256.)
            # MAE_loss += lytest.MAE_tensor(pred * 65535. / 65536., gt * 65535. / 256.)

            # pred = pred.cpu().clamp(0., 1.)

            # lyvision.vision_color(pred, name="pred.png", path="./saved/test/")
            # lyvision.vision_color(gt, name="gt.png", path="./saved/test/")
            # lyvision.vision_rgb(input[:, :3, :, :], name="img.png", path="./saved/test/")

            # num_all += 1
            
            # loop.set_description('Val')
            # loop.set_postfix(RMSE_loss=RMSE_loss.item() / num_all.item(), ARE_loss=ARE_loss.item() / num_all.item(),
            #                 MAE_loss=MAE_loss.item() / num_all.item())
                
    # dist.all_reduce(RMSE_loss, op=dist.ReduceOp.SUM)
    # dist.all_reduce(ARE_loss, op=dist.ReduceOp.SUM)
    # dist.all_reduce(MAE_loss, op=dist.ReduceOp.SUM)
    # dist.all_reduce(num_all, op=dist.ReduceOp.SUM)

    # record[dataset_name + "_" + name] = {'RMSE_loss':RMSE_loss.item() / num_all.item(), 'ARE_loss':ARE_loss.item() / num_all.item(),
    #                                      'MAE_loss':MAE_loss.item() / num_all.item()}

    # if dist.get_rank() == 0:
    #     lytest.save_result(record, name=dataset_name+"_"+name, path="./saved/" + proc_it +"/")

    #     print("End Val , result saved in " + proc_it + " ...")


def val_intime(device, G2_model, epoch, proc_it="train", name="val", dataset_name="DIODE"):

    dataset_val = data.G2_data(name, [dataset_name])
    datasampler_val = DistributedSampler(dataset_val)
    dataloader_val = DataLoader(dataset_val, batch_size=1, sampler=datasampler_val,
                                num_workers=5, pin_memory=True, drop_last=False)

    RMSE_loss = torch.tensor(0.).to(device)
    ARE_loss = torch.tensor(0.).to(device)
    MAE_loss = torch.tensor(0.).to(device)
    num_all = torch.tensor(0).to(device)

    record = {}

    if dist.get_rank() == 0:
        print("Start Val in epoch " + str(epoch) + ": val dataset: " + dataset_name + " val name: " + name + " ...")

    with torch.no_grad():
        loop = tqdm(dataloader_val, desc="Val")
        for i, (input, gt, Kcam, full_path) in enumerate(loop):
            # lydebug.check_tensor(img)
            # lydebug.check_tensor(gt)
            # lydebug.check_tensor(img_ori)
            # lydebug.check_tensor(gt_ori)

            input = [input[0].to(device), input[1].to(device), input[2].to(device), input[3].to(device)]
            gt = gt.to(device)
            Kcam = Kcam.to(device)

            pred = G2_model(input[0], input[1], input[2], input[3], Kcam)
            pred = pred["prediction"][-1]

            # print(torch.max(input[2]*65535.0/512.0))
            # print(torch.max(input[1][:,:1]*65535.0/512.0))
            # print(torch.max(gt*65535.0/256.0))
            # pred = torch.clip(pred, 0., 80.0)

            # pred = pred * build_essential.coefficients.to(device)

            RMSE_loss += lytest.RMSE_tensor(pred*65535.0/65536.0, gt*65535.0/256.0)
            ARE_loss += lytest.ARE_tensor(pred*65535.0/65536.0, gt*65535.0/256.0)
            MAE_loss += lytest.MAE_tensor(pred*65535.0/65536.0, gt*65535.0/256.0)

            # pred = pred.cpu().clamp(0., 1.)

            # lyvision.vision_color(pred, name="pred.png", path="./saved/test/")
            # lyvision.vision_color(gt, name="gt.png", path="./saved/test/")
            # lyvision.vision_rgb(input[:, :3, :, :], name="img.png", path="./saved/test/")

            num_all += 1
            
            loop.set_description('Val')
            loop.set_postfix(RMSE_loss=RMSE_loss.item() / num_all.item(), ARE_loss=ARE_loss.item() / num_all.item(),
                            MAE_loss=MAE_loss.item() / num_all.item())
                
    dist.all_reduce(RMSE_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(ARE_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(MAE_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(num_all, op=dist.ReduceOp.SUM)

    record[dataset_name + "_" + name] = {'RMSE_loss':RMSE_loss.item() / num_all.item(), 'ARE_loss':ARE_loss.item() / num_all.item(),
                                         'MAE_loss':MAE_loss.item() / num_all.item()}
    
    if dist.get_rank() == 0:
        lytest.save_result(record, name=dataset_name+"_"+name, path="./saved/"+proc_it+"/"+str(epoch).zfill(3)+"/")

        print("End Val , result saved in " + proc_it + " ...")


if __name__ == "__main__":
    import os
    
    os.environ["OMP_NUM_THREADS"] = "5"
    
    # build_essential.build_it()

    dist.init_process_group('nccl', init_method='env://')

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    global device
    device = torch.device("cuda", local_rank)

    for i in ["KITTIDC"]:
        for j in ["val"]:
            val_it(device, name=j, dataset_name=i)

    # for i in ["KITTIDC"]:
    #     for j in ["test"]:
    #         val_submit(device, name=j, dataset_name=i)
