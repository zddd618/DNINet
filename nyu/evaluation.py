import numpy as np
import dataset as data
import dataset_nyu as data_nyu
import torch
from torch import nn
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
from depth_to_normal import Depth2Normal
from scipy.optimize import linear_sum_assignment

from deeptool import lytest, lydebug, lyvision

alpha = 0.000001
class MetricALL(nn.Module):
    def __init__(self, mul_factor):
        super().__init__()
        self.t_valid = 0.000001
        self.mul_factor = mul_factor
        self.metric_name = [
            'RMSE', 'MAE', 'REL', 'D^1', 'D^2', 'D^3', 'D102', 'D105', 'D110'
        ]

    def forward(self, pred, gt):
        with torch.no_grad():
            pred = pred.detach() / self.mul_factor
            gt = gt.detach() / self.mul_factor
            pred_inv = 1.0 / (pred + 1e-8)
            gt_inv = 1.0 / (gt + 1e-8)

            # For numerical stability
            mask = gt > self.t_valid
            # num_valid = mask.sum()
            B = mask.size(0)
            num_valid = torch.sum(mask.view(B, -1), -1, keepdim=True)

            # pred = pred[mask]
            # gt = gt[mask]
            pred = pred * mask
            gt = gt * mask

            # pred_inv = pred_inv[mask]
            # gt_inv = gt_inv[mask]
            pred_inv = pred_inv * mask
            gt_inv = gt_inv * mask

            # pred_inv[pred <= self.t_valid] = 0.0
            # gt_inv[gt <= self.t_valid] = 0.0

            # RMSE / MAE
            diff = pred - gt
            diff_abs = torch.abs(diff)
            diff_sqr = torch.pow(diff, 2)

            rmse = torch.sum(diff_sqr.view(B, -1), -1, keepdim=True) / (num_valid + 1e-8)
            rmse = torch.sqrt(rmse)

            mae = torch.sum(diff_abs.view(B, -1), -1, keepdim=True) / (num_valid + 1e-8)

            # iRMSE / iMAE
            diff_inv = pred_inv - gt_inv
            diff_inv_abs = torch.abs(diff_inv)
            diff_inv_sqr = torch.pow(diff_inv, 2)

            irmse = torch.sum(diff_inv_sqr.view(B, -1), -1, keepdim=True) / (num_valid + 1e-8)
            irmse = torch.sqrt(irmse)

            imae = torch.sum(diff_inv_abs.view(B, -1), -1, keepdim=True) / (num_valid + 1e-8)

            # Rel
            rel = diff_abs / (gt + 1e-8)
            rel = torch.sum(rel.view(B, -1), -1, keepdim=True) / (num_valid + 1e-8)

            # delta
            r1 = gt / (pred + 1e-8)
            r2 = pred / (gt + 1e-8)
            ratio = torch.max(r1, r2)

            ratio = torch.max(ratio, 10000 * (1 - mask.float()))

            del_1 = (ratio < 1.25).type_as(ratio)
            del_2 = (ratio < 1.25 ** 2).type_as(ratio)
            del_3 = (ratio < 1.25 ** 3).type_as(ratio)
            del_102 = (ratio < 1.02).type_as(ratio)
            del_105 = (ratio < 1.05).type_as(ratio)
            del_110 = (ratio < 1.10).type_as(ratio)

            del_1 = torch.sum(del_1.view(B, -1), -1, keepdim=True) / (num_valid + 1e-8)
            del_2 = torch.sum(del_2.view(B, -1), -1, keepdim=True) / (num_valid + 1e-8)
            del_3 = torch.sum(del_3.view(B, -1), -1, keepdim=True) / (num_valid + 1e-8)
            del_102 = torch.sum(del_102.view(B, -1), -1, keepdim=True) / (num_valid + 1e-8)
            del_105 = torch.sum(del_105.view(B, -1), -1, keepdim=True) / (num_valid + 1e-8)
            del_110 = torch.sum(del_110.view(B, -1), -1, keepdim=True) / (num_valid + 1e-8)

            result = [rmse, mae, irmse, imae, rel, del_1, del_2, del_3, del_102, del_105, del_110]

        return result
    
def find_min_rmse_mapping(tensor1, tensor2):
    """
    计算两个形状为[1,3,256,256]的张量之间的通道映射关系
    
    参数:
        tensor1: torch.Tensor, 形状[1, 3, 256, 256]
        tensor2: torch.Tensor, 形状[1, 3, 256, 256]
    
    返回:
        dict: 包含通道映射关系的字典，格式为 {源通道: 目标通道}
    """
    # 移除batch维度并确保形状正确
    tensor1 = tensor1.squeeze(0)  # [3, 256, 256]
    tensor2 = tensor2.squeeze(0)  # [3, 256, 256]
    
    # 验证通道数
    assert tensor1.shape[0] == 3 and tensor2.shape[0] == 3, "输入张量必须有3个通道"
    
    # 初始化RMSE矩阵
    rmse_matrix = torch.zeros((3, 3))
    
    # 计算所有通道组合的RMSE
    for i in range(3):
        for j in range(3):
            # 计算两个通道之间的MSE，然后取平方根得到RMSE
            mse = torch.mean((tensor1[i] - tensor2[j]) ** 2)
            rmse_matrix[i, j] = torch.sqrt(mse)
    
    # 使用匈牙利算法找到最小权重匹配
    cost_matrix = rmse_matrix.detach().numpy()
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # 创建映射字典
    mapping = {int(row): int(col) for row, col in zip(row_ind, col_ind)}
    print("Mapping from tensor1 channels to tensor2 channels:", mapping)
    return mapping
    
def val_it(local_rank, proc_it="test", name="val", dataset_name="DIODE"):

    G2_model = model.Completion3D()
    # G2_model.eval()
    G2_model.to(device)
    # lydebug.check_model(G2_model)
    G2_model = torch.nn.parallel.DistributedDataParallel(G2_model, 
                device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)  # 这句加载到多GPU上
    
    dataset_val = data_nyu.NYU('val', device=device)
    datasampler_val = DistributedSampler(dataset_val)
    dataloader_val = DataLoader(dataset_val, batch_size=1, sampler=datasampler_val,
                                num_workers=5, pin_memory=True, drop_last=False)

    if os.path.exists(os.environ.get("DNINET_NYU_WEIGHTS", "../weights/nyu_ema.pt")):
        print("Load checkpoint ...")
        checkpoint = torch.load(os.environ.get("DNINET_NYU_WEIGHTS", "../weights/nyu_ema.pt"), map_location="cpu")
        G2_model.load_state_dict(checkpoint["model_state_dict"])

    G2_model.eval()

    d2n = Depth2Normal()

    RMSE_loss = torch.tensor(0.).to(device)
    REL_loss = torch.tensor(0.).to(device)
    MAE_loss = torch.tensor(0.).to(device)
    DEL1_loss = torch.tensor(0.).to(device)
    DEL2_loss = torch.tensor(0.).to(device)
    DEL3_loss = torch.tensor(0.).to(device)
    num_all = torch.tensor(0).to(device)

    record = {}

    metric = MetricALL(1)

    if dist.get_rank() == 0:
        print("Start Val : val dataset: " + dataset_name + " val name: " + name + " ...")

    all_normal_0 = 0
    all_normal_1 = 0
    all_normal_2 = 0
    count = 0
    mean_angle_error = 0.0

    with torch.no_grad():
        loop = tqdm(dataloader_val, desc="Val")
        for i, (input, gt, Kcam, depth_t_crop, mask_normal, intrin, path_file) in enumerate(loop):
            # lydebug.check_tensor(img)
            # lydebug.check_tensor(gt)
            # lydebug.check_tensor(img_ori)
            # lydebug.check_tensor(gt_ori)
            path_file_num = path_file[0].split('/')[-1].split('.')[0]

            input = [input[0].to(device), input[1].to(device), input[2].to(device)]
            gt = gt.to(device)
            Kcam = Kcam.to(device)

            record_time = time.time()
            pred = G2_model(input[0], input[1], input[2], Kcam)
            print("Inference time: ", time.time() - record_time)
            pred_normal = pred["prediction_normal"]
            
            pred_mid = pred["predictions_list"][-1]
            pred = pred["prediction"][-1]

            pred = pred[:,:,14:242,8:312]
            pred_mid = pred_mid[:,:,14:242,8:312]
            gt = gt[:,:,14:242,8:312]
            print(pred_normal.shape)
            pred_normal = pred_normal[:,:,14:242,8:312]

            depth_t_crop = depth_t_crop.to(device)
            mask_normal = mask_normal.to(device)
            intrin = intrin.to(device)

            normal_gt, normal_gt_mask = d2n(depth_t_crop, intrin, mask_normal, 1.0)
            
            valid_pics = torch.sum(normal_gt_mask, dtype=torch.float32) + 1e-6

            normal_gt_tmp = normal_gt[:,2:, :, :].clone()
            normal_gt[:,2:, :, :] = normal_gt[:,:1, :, :]
            normal_gt[:,:1, :, :] = normal_gt_tmp

            if valid_pics < 10:
                return
            pred_normal = torch.nn.functional.interpolate(pred_normal, [normal_gt.shape[2], normal_gt.shape[3]], mode='bilinear')
            find_min_rmse_mapping(pred_normal, normal_gt)
            normal_error = torch.cosine_similarity(pred_normal, normal_gt, dim=1)
            normal_error = torch.clamp(normal_error, min=-1.0, max=1.0)
            angle_error = torch.acos(normal_error) * 180.0 / torch.pi
            angle_error = angle_error[:, None, :, :]
            angle_error = angle_error[normal_gt_mask]
            # Calculation error
            mean_error = angle_error.sum() / valid_pics
            rmse_error = torch.sqrt( torch.sum(torch.square(angle_error)) / valid_pics )
            median_error = angle_error.median()
            a1_error = 100.0 * (torch.sum(angle_error < 5) / valid_pics)
            a2_error = 100.0 * (torch.sum(angle_error < 7.5) / valid_pics)
            
            a3_error = 100.0 * (torch.sum(angle_error < 11.25) / valid_pics)
            a4_error = 100.0 * (torch.sum(angle_error < 22.5) / valid_pics)
            a5_error = 100.0 * (torch.sum(angle_error < 30) / valid_pics)

            vec1 = pred_normal.squeeze(0).view(3, -1)  # [3, 65536]
            vec2 = normal_gt.squeeze(0).view(3, -1)  # [3, 65536]
            
            # 计算点积 [65536]
            dot_product = (vec1 * vec2).sum(dim=0)
            
            # 计算模长
            norm1 = torch.norm(vec1, dim=0)
            norm2 = torch.norm(vec2, dim=0)
            
            # 计算余弦相似度 (带数值稳定性处理)
            cos_sim = dot_product / (norm1 * norm2 + 1e-8)
            
            # 防止浮点误差导致的值超出[-1,1]范围
            cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
            
            # 计算角度(弧度)
            angles_rad = torch.acos(cos_sim)
            
            # 转换为角度
            angles_deg = torch.rad2deg(angles_rad)
            
            # 计算平均角度误差
            mae = torch.mean(angles_deg).item()

            all_normal_0 += a3_error
            all_normal_1 += a4_error
            all_normal_2 += a5_error
            count += 1
            mean_angle_error += mae

            print(a3_error, a4_error, a5_error)

            # if valid_pics > 1e-5:
            # If the current node gets data with valid normal
            dist_node_cnt = (valid_pics - 1e-6) / valid_pics

            # pred = pred * build_essential.coefficients.to(device)

            result = metric(pred, gt*10.0)

            RMSE_loss += result[0].squeeze()
            MAE_loss += result[1].squeeze()
            REL_loss += result[4].squeeze()
            DEL1_loss += result[5].squeeze()
            DEL2_loss += result[6].squeeze()
            DEL3_loss += result[7].squeeze()

            # pred = pred.cpu().clamp(0., 1.)

            pred_normal = (pred_normal + 1.0) / 2.0
            normal_gt = (normal_gt + 1.0) / 2.0

            lyvision.vision_rgb(pred_normal, name="pred.png", path="./saved/test/")
            lyvision.vision_rgb(normal_gt, name="gt.png", path="./saved/test/")
            # lyvision.vision_rgb(input[:, :3, :, :], name="img.png", path="./saved/test/")
            # lyvision.vision_color(pred/10.0, name="pred.png", path="./saved/test/vision_final/"+str(i)+"/")
            # lyvision.vision_color(gt, name="gt.png", path="./saved/test/vision_final/"+str(i)+"/")
            # lyvision.vision_rgb(pred_normal, name="normal.png", path="./saved/test/vision_tmp/"+str(i)+"/")
            # lyvision.vision_rgb(input[:, :3, :, :], name="img.png", path="./saved/test/vision_final/"+str(i)+"/")
            # lyvision.vision_color(pred_mid/10.0, name="pred.png", path="./saved/test/vision_tmp/"+str(i)+"/")
            # lyvision.vision_color(gt, name="gt.png", path="./saved/test/vision_tmp/"+str(i)+"/")
            # lyvision.vision_rgb(input[:, :3, :, :], name="img.png", path="./saved/test/vision_tmp/"+str(i)+"/")
            mean_img = torch.tensor([90.995/255.0, 96.2278/255.0, 94.3213/255.0], device=input[0].device).view(1, 3, 1, 1)
            std_img  = torch.tensor([79.2382/255.0, 80.5267/255.0, 82.1483/255.0], device=input[0].device).view(1, 3, 1, 1)

            # img_recovered = img * std + mean
            # img_recovered = torch.clamp(img_recovered, 0.0, 1.0)

            lyvision.vision_color((pred - (pred).amin()) / ((pred).amax() - (pred).amin() + 1e-8), name="pred.png", path="./saved/test/vision_final/"+path_file_num+"/")
            lyvision.vision_color((gt - (gt).amin()) / ((gt).amax() - (gt).amin() + 1e-8), name="gt.png", path="./saved/test/vision_final/"+path_file_num+"/")
            lyvision.vision_rgb(pred_normal, name="normal.png", path="./saved/test/vision_tmp/"+path_file_num+"/")
            lyvision.vision_rgb(normal_gt, name="normal_gt.png", path="./saved/test/vision_tmp/"+path_file_num+"/")
            lyvision.vision_rgb(torch.clamp(input[0][:, :3, :, :]*std_img+mean_img, 0.0, 1.0), name="img.png", path="./saved/test/vision_final/"+path_file_num+"/")
            lyvision.vision_color((pred_mid - (pred_mid).amin()) / ((pred_mid).amax() - (pred_mid).amin() + 1e-8), name="pred.png", path="./saved/test/vision_tmp/"+path_file_num+"/")
            lyvision.vision_color((gt - (gt).amin()) / ((gt).amax() - (gt).amin() + 1e-8), name="gt.png", path="./saved/test/vision_tmp/"+path_file_num+"/")
            lyvision.vision_rgb(torch.clamp(input[0][:, :3, :, :]*std_img+mean_img, 0.0, 1.0), name="img.png", path="./saved/test/vision_tmp/"+path_file_num+"/")

            num_all += 1
            
            loop.set_description('Val')
            loop.set_postfix(RMSE_loss=RMSE_loss.item() / num_all.item(), REL_loss=REL_loss.item() / num_all.item(),
                            MAE_loss=MAE_loss.item() / num_all.item(), DEL1_loss=DEL1_loss.item() / num_all.item(),
                            DEL2_loss=DEL2_loss.item() / num_all.item(), DEL3_loss=DEL3_loss.item() / num_all.item())
                
    dist.all_reduce(RMSE_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(REL_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(MAE_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(DEL1_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(DEL2_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(DEL3_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(num_all, op=dist.ReduceOp.SUM)

    print("Normal angle error: ", all_normal_0 / count, all_normal_1 / count, all_normal_2 / count, 
          "mean angle error: ", mean_angle_error / count)

    record[dataset_name + "_" + name] = {'RMSE_loss':RMSE_loss.item() / num_all.item(), 'REL_loss':REL_loss.item() / num_all.item(),
                                         'MAE_loss':MAE_loss.item() / num_all.item(), 'DEL1_loss':DEL1_loss.item() / num_all.item(),
                                            'DEL2_loss':DEL2_loss.item() / num_all.item(), 'DEL3_loss':DEL3_loss.item() / num_all.item()}
    
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

    if os.path.exists("checkpoint/checkpoint_ema.pt"):
        print("Load checkpoint ...")
        checkpoint = torch.load("checkpoint/checkpoint_ema.pt", map_location="cpu")
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
        for i, (input, jpg_name) in enumerate(loop):
            # lydebug.check_tensor(img)
            # lydebug.check_tensor(gt)
            # lydebug.check_tensor(img_ori)
            # lydebug.check_tensor(gt_ori)

            input = [input[0].to(device), input[1].to(device), input[2].to(device)]

            pred = G2_model(input[0], input[1], input[2])
            pred_2 = pred["prediction"]

            pred = G2_model(torch.flip(input[0], [3]), torch.flip(input[1], [3]), torch.flip(input[2], [3]))
            pred_1 = torch.flip(pred["prediction"], [3])

            pred = (pred_1 + pred_2) / 2.0

            save_path = "./saved/submit/" + jpg_name[0]
            pred_np = pred[0][0].cpu().numpy()
            pred_np = pred_np * 65535. / 512.
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

    dataset_val = data_nyu.NYU('val')
    datasampler_val = DistributedSampler(dataset_val)
    dataloader_val = DataLoader(dataset_val, batch_size=1, sampler=datasampler_val,
                                num_workers=5, pin_memory=True, drop_last=False)

    RMSE_loss = torch.tensor(0.).to(device)
    REL_loss = torch.tensor(0.).to(device)
    MAE_loss = torch.tensor(0.).to(device)
    DEL1_loss = torch.tensor(0.).to(device)
    DEL2_loss = torch.tensor(0.).to(device)
    DEL3_loss = torch.tensor(0.).to(device)
    num_all = torch.tensor(0).to(device)

    record = {}

    metric = MetricALL(1)

    if dist.get_rank() == 0:
        print("Start Val in epoch " + str(epoch) + ": val dataset: " + dataset_name + " val name: " + name + " ...")

    with torch.no_grad():
        loop = tqdm(dataloader_val, desc="Val")
        for i, (input, gt, Kcam) in enumerate(loop):
            # lydebug.check_tensor(img)
            # lydebug.check_tensor(gt)
            # lydebug.check_tensor(img_ori)
            # lydebug.check_tensor(gt_ori)

            input = [input[0].to(device), input[1].to(device), input[2].to(device)]
            gt = gt.to(device)
            Kcam = Kcam.to(device)

            pred = G2_model(input[0], input[1], input[2], Kcam)
            pred = pred["prediction"][-1]

            pred = pred[:,:,14:242,8:312]
            gt = gt[:,:,14:242,8:312]

            # pred = pred * build_essential.coefficients.to(device)

            result = metric(pred, gt*10.0)

            RMSE_loss += result[0].squeeze()
            MAE_loss += result[1].squeeze()
            REL_loss += result[4].squeeze()
            DEL1_loss += result[5].squeeze()
            DEL2_loss += result[6].squeeze()
            DEL3_loss += result[7].squeeze()

            # pred = pred.cpu().clamp(0., 1.)

            # lyvision.vision_color(pred, name="pred.png", path="./saved/test/")
            # lyvision.vision_color(gt, name="gt.png", path="./saved/test/")
            # lyvision.vision_rgb(input[:, :3, :, :], name="img.png", path="./saved/test/")

            num_all += 1
            
            loop.set_description('Val')
            loop.set_postfix(RMSE_loss=RMSE_loss.item() / num_all.item(), REL_loss=REL_loss.item() / num_all.item(),
                            MAE_loss=MAE_loss.item() / num_all.item(), DEL1_loss=DEL1_loss.item() / num_all.item(),
                            DEL2_loss=DEL2_loss.item() / num_all.item(), DEL3_loss=DEL3_loss.item() / num_all.item())
                
    dist.all_reduce(RMSE_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(REL_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(MAE_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(DEL1_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(DEL2_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(DEL3_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(num_all, op=dist.ReduceOp.SUM)

    record[dataset_name + "_" + name] = {'RMSE_loss':RMSE_loss.item() / num_all.item(), 'REL_loss':REL_loss.item() / num_all.item(),
                                         'MAE_loss':MAE_loss.item() / num_all.item(), 'DEL1_loss':DEL1_loss.item() / num_all.item(),
                                            'DEL2_loss':DEL2_loss.item() / num_all.item(), 'DEL3_loss':DEL3_loss.item() / num_all.item()}
    
    if dist.get_rank() == 0:
        lytest.save_result(record, name=dataset_name+"_"+name, path="./saved/"+proc_it+"/"+str(epoch).zfill(3)+"/")

        print("End Val , result saved in " + proc_it + " ...")


if __name__ == "__main__":
    import os
    
    os.environ["OMP_NUM_THREADS"] = "5"

    torch.manual_seed(1500)
    torch.cuda.manual_seed(1500)
    torch.cuda.manual_seed_all(1500)
    
    # build_essential.build_it()

    dist.init_process_group('nccl', init_method='env://')

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    global device
    device = torch.device("cuda", local_rank)

    for i in ["KITTIDC"]:
        for j in ["val_rel"]:
            val_it(device, name=j, dataset_name=i)

    # for i in ["KITTIDC"]:
    #     for j in ["test"]:
    #         val_submit(device, name=j, dataset_name=i)
