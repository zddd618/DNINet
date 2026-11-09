import os
os.environ["OMP_NUM_THREADS"] = "2"

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR, OneCycleLR, LambdaLR
from torch.cuda.amp import autocast, GradScaler
import dataset as data
import torch.nn as nn
from copy import deepcopy
import torch.distributed
from tqdm import tqdm
import loss
import model
import torch
import build_essential
import evaluation
import math
from model import GenKernel
import functools

# torch.backends.cudnn.benchmark = True

# from torch.utils.tensorboard import SummaryWriter
import numpy as np

from deeptool import lyconfig, lytrain, lydebug

class EMA(nn.Module):
    def __init__(self, model, decay=0.9999, ddp=False):
        super().__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        if ddp:
            self.broadcast()
        self.decay = decay

    def broadcast(self):
        for ema_v in self.module.state_dict().values():
            torch.distributed.broadcast(ema_v, src=0, async_op=False)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)

def weights_init(m, mode='trunc'):
    from torch.nn.init import _calculate_fan_in_and_fan_out
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        if hasattr(m, 'weight'):
            if mode == 'trunc':
                fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight.data)
                std = math.sqrt(2.0 / float(fan_in + fan_out))
                torch.nn.init.trunc_normal_(m.weight.data, mean=0, std=std)
            elif mode == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data)
            else:
                raise ValueError(f'unknown mode = {mode}')
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    if classname.find('Conv1d') != -1:
        if hasattr(m, 'weight'):
            if mode == 'trunc':
                fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight.data)
                std = math.sqrt(2.0 / float(fan_in + fan_out))
                torch.nn.init.trunc_normal_(m.weight.data, mean=0, std=std)
            elif mode == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data)
            else:
                raise ValueError(f'unknown mode = {mode}')
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        if mode == 'trunc':
            fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight.data)
            std = math.sqrt(2.0 / float(fan_in + fan_out))
            torch.nn.init.trunc_normal_(m.weight.data, mean=0, std=std)
        elif mode == 'xavier':
            torch.nn.init.xavier_normal_(m.weight.data)
        else:
            raise ValueError(f'unknown mode = {mode}')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)

# 初始化一些基本内容
# build_essential.build_it()
# writer = SummaryWriter(log_dir='./saved/logs')  # 创建一个记录器

# torch.autograd.set_detect_anomaly(True)

# 初始化训练方法
epoch_num = 20
learning_rate = 0.0002

torch.distributed.init_process_group('nccl', init_method='env://')

torch.manual_seed(2025)
torch.cuda.manual_seed(2025)
torch.cuda.manual_seed_all(2025)

local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
global device
device = torch.device("cuda", local_rank)

G2_model = model.Completion3D()
# G2_model.apply(functools.partial(weights_init, mode='trunc'))
# for m in G2_model.modules():
#     if isinstance(m, GenKernel) and m.conv[1].conv.bn.weight is not None:
#         nn.init.constant_(m.conv[1].conv.bn.weight, 0)
# G2_model.apply(model.weights_init)
G2_model.to(device)  # 这句不能少，最好不要用model.cuda()
G2_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(G2_model)
G2_model = torch.nn.parallel.DistributedDataParallel(G2_model, device_ids=[local_rank], output_device=local_rank,
                                                  find_unused_parameters=True)  # 这句加载到多GPU上

G2_model.eval()

ema = EMA(G2_model, decay=0.9999, ddp=True)

scaler = GradScaler()

loss_func = loss.G2_loss()
loss_func = loss_func.cuda()

dataset_train = data.G2_data("train-fullres", ["KITTIDC"])
# dataset_val = data.G2_data("val", ["DIODE"])
# lydebug.check_dataset(dataset_train)
datasampler_train = DistributedSampler(dataset_train)
dataloader_train = DataLoader(dataset_train, batch_size=2, sampler=datasampler_train,
                              num_workers=4, pin_memory=True, drop_last=True)

accumulation_steps = 1

warm_up_cnt = 0.0
warm_up_max = 1000.0 * accumulation_steps

for name, param in G2_model.named_parameters():
    if "refine_depth" not in name and "bridge" not in name and "global_feat" not in name:
        param.requires_grad = False
for name, module in G2_model.named_modules(): 
    if "refine_depth" in name or "bridge" in name and "global_feat" in name: 
        module.train()

optimizer = torch.optim.AdamW([{'params':filter(lambda p : p.requires_grad, G2_model.parameters()), 'initial_lr':learning_rate}],
                        lr=learning_rate, weight_decay=0.05)
optimizer.zero_grad()

if os.path.exists("checkpoint/checkpoint.pt"):
    #map_location = {'cuda:0': 'cuda:0', 'cuda:1': 'cuda:1', 'cuda:2': 'cuda:0', 'cuda:3': 'cuda:1'}
    checkpoint = torch.load("checkpoint/checkpoint.pt", map_location="cpu")
    G2_model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    epoch_begin = checkpoint["epoch"] + 1
    # epoch_begin = 0
    checkpoint.clear()
    del checkpoint
    ema_checkpoint = torch.load("checkpoint/checkpoint_ema.pt", map_location="cpu")
    ema.module.load_state_dict(ema_checkpoint["model_state_dict"])
    ema.broadcast()
    ema_checkpoint.clear()
    del ema_checkpoint
elif os.path.exists("checkpoint/checkpoint_ori.pt"):
    checkpoint = torch.load("checkpoint/checkpoint_ori.pt", map_location="cpu")
    G2_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    # scaler.load_state_dict(checkpoint['scaler_state_dict'])
    # epoch_begin = checkpoint["epoch"] + 1
    epoch_begin = 0
    checkpoint.clear()
    del checkpoint
else:
    epoch_begin = 0

start_step = epoch_begin * len(dataloader_train)

# class PolyLr():
#     def __init__(self):
#         self.power = 1.0
#         self.total_iters=len(dataloader_train)*epoch_num
#         self.lr_end = 0.15
#         self.base_lr = 1.0

#     def __call__(self, steps):
#         coeff = (1 - steps / self.total_iters)**self.power
#         return (self.base_lr - self.lr_end) * coeff + self.lr_end
    
class CustomLr():
    def __init__(self, base_lr=1.0, total_iters=1000):
        self.base_lr = base_lr       # 峰值学习率（对应coeff=1.0时的值）
        self.total_iters = total_iters  # 总迭代次数
        self.warmup_iters = int(0.1 * total_iters)  # 前10%步数用于warmup
        self.cool_iters = total_iters - self.warmup_iters  # 剩余90%步数

    def __call__(self, steps):
        # Phase 1：前10%步数线性上升
        if steps < self.warmup_iters:
            # 从0.025线性增长到1.0
            alpha = steps / self.warmup_iters
            coeff = 0.025 + (1.0 - 0.025) * alpha
        # Phase 2：后90%步数余弦下降
        else:
            # 计算在cooling阶段的进度（0.0~1.0）
            current_step = steps - self.warmup_iters
            progress = current_step / self.cool_iters
            progress = min(progress, 1.0)  # 限制不超过1.0
            
            # 余弦退火公式（从1.0到0.1）
            coeff = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
        
        # 最终学习率 = 基准学习率 × 系数
        return self.base_lr * coeff
    
polylr = CustomLr(base_lr=1.0, total_iters=len(dataloader_train)*epoch_num)
    
scheduler = LambdaLR(optimizer, lr_lambda=polylr, last_epoch=start_step-1)

# optimizer.zero_grad()
ema.set(G2_model)

for epoch in range(epoch_begin, epoch_num):
    train_loss = 0
    train_loss_5 = 0
    train_loss_4 = 0
    train_loss_3 = 0
    train_loss_2 = 0
    train_loss_1 = 0
    train_num = 0

    # G2_model.train()
    datasampler_train.set_epoch(epoch)

    loop = tqdm(dataloader_train, desc="Train")
    for i, (input, hole, gt, normal, Kcam) in enumerate(loop, start=1):
        # if (epoch == 0) and (warm_up_cnt < warm_up_max):
        #     warm_up_cnt += 1
        #     for param_groups in optimizer.param_groups:
        #         param_groups['lr'] = param_groups['initial_lr'] * warm_up_cnt / warm_up_max

        input = [input[0].to(device, non_blocking=True), input[1].to(device, non_blocking=True), input[2].to(device, non_blocking=True), input[3].to(device, non_blocking=True)]
        hole = [hole[0].to(device, non_blocking=True), hole[1].to(device, non_blocking=True)]
        gt = gt.to(device, non_blocking=True)
        normal = normal.to(device, non_blocking=True)
        Kcam = Kcam.to(device, non_blocking=True)

        pred = G2_model(input[0], input[1], input[2], input[3], Kcam)
        loss, loss_1, loss_2, loss_3, loss_4, loss_5 = loss_func(pred, gt, hole)
        (loss / accumulation_steps).backward()
        if i % accumulation_steps == 0:
            clip_grad_norm_(G2_model.parameters(), max_norm=0.1, norm_type=2)
            optimizer.step()
            ema.update(G2_model)
            optimizer.zero_grad()
            
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()
        # optimizer.zero_grad()
        # loss = loss / accumulation_steps
        # loss.backward()
        # if i % accumulation_steps == 0:
        #     optimizer.step()
        #     optimizer.zero_grad()

        train_loss += loss.item() * input[0].size(0)
        train_loss_5 += loss_5.item() * input[0].size(0)
        train_loss_4 += loss_4.item() * input[0].size(0)
        train_loss_3 += loss_3.item() * input[0].size(0)
        train_loss_2 += loss_2.item() * input[0].size(0)
        train_loss_1 += loss_1.item() * input[0].size(0)
        train_num += input[0].size(0)

        loop.set_description(f'Epoch [{epoch+1}/{epoch_num}]')
        loop.set_postfix(loss=train_loss / train_num, loss_5=train_loss_5 / train_num, loss_4=train_loss_4 / train_num, loss_3=train_loss_3 / train_num, loss_2=train_loss_2 / train_num, loss_1=train_loss_1 / train_num)

    # writer.add_scalar('Loss/all_loss', train_loss, global_step=epoch)
    # writer.add_scalar('Loss/relative_loss', train_loss_r, global_step=epoch)
    # writer.add_scalar('Loss/abs_loss', train_loss_abs, global_step=epoch)
    # writer.add_scalar('Loss/gradient_loss', train_loss_g, global_step=epoch)
    # writer.add_scalar('Loss/f_loss', train_loss_f, global_step=epoch)
    # writer.add_scalar('Loss/s_loss', train_loss_s, global_step=epoch)
    # writer.add_scalar('Loss/tmps_loss', train_loss_tmps, global_step=epoch)
    
    # writer.add_image('gt', img, global_step=epoch)
    # writer.add_image('completion', img, global_step=epoch)

        scheduler.step()

    lytrain.save_checkpoint(epoch, lyconfig.checkpoint_path + "checkpoint.pt", G2_model, optimizer, scaler)
    lytrain.save_checkpoint(epoch, lyconfig.checkpoint_path + "checkpoint_ema.pt", ema.module, optimizer, scaler)

    if (epoch + 1) % 1 == 0:
        # writer.add_scalar('Val/rmse', train_loss, global_step=epoch)
        # writer.add_scalar('Val/abs_rel', train_loss_r, global_step=epoch)
        # G2_model.eval()
        for i in ["KITTIDC"]:
            for j in ["val_rel"]:
                evaluation.val_intime(device, ema.module, epoch+1, name=j, dataset_name=i)
        lytrain.save_checkpoint(epoch, lyconfig.history_checkpoint_path + build_essential.exam_name + "/checkpoint" + str(epoch + 1) + ".pt", G2_model, optimizer, scaler)
        lytrain.save_checkpoint(epoch, lyconfig.history_checkpoint_path + build_essential.exam_name + "/checkpoint_ema" + str(epoch + 1) + ".pt", ema.module, optimizer, scaler)

# writer.close()
