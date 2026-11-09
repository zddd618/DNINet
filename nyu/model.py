import torch.nn as nn
import torch
from torchvision.transforms import Resize
import torch.nn.init as init
import math
import torch.nn.functional as F
import time
from timm.models.layers import DropPath
from collections import OrderedDict
from torch.autograd import Function
from torch.cuda.amp import custom_fwd, custom_bwd
import BpOps
import functools
import numpy as np
from einops.layers.torch import Rearrange

from deeptool import lydebug, lyvision

from model_dcnv2 import Encoder_LRRU

from point_utils import PointNetSetAbstraction, PointNetSetAbstractionMsg

from spn import BasicDepthEncoder, pa_spn


class GlobalPoint(nn.Module):

    def __init__(self):
        super(GlobalPoint, self).__init__()

        normal_channel = True

        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstraction(npoint=256, radius=0.2, nsample=16, in_channel=in_channel, mlp=[32, 32, 64], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=64, radius=0.4, nsample=32, in_channel=64 + 3, mlp=[64, 64, 128], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=128 + 3, mlp=[128, 256, 512], group_all=True)
        self.fc1 = nn.Linear(512, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(128, 64)

    def pinv(self, S, K, xx, yy):
        fx, fy, cx, cy = K[:, 0:1, 0:1], K[:, 1:2, 1:2], K[:, 0:1, 2:3], K[:, 1:2, 2:3]
        S = S.view(S.shape[0], 1, -1)
        xx = xx.reshape(1, 1, -1)
        yy = yy.reshape(1, 1, -1)
        Px = S * (xx - cx) / fx
        Py = S * (yy - cy) / fy
        Pz = S
        Pxyz = torch.cat([Px, Py, Pz], dim=1).contiguous()
        return Pxyz

    def random_sample(self, point_cloud, num_samples):
        idx = np.random.choice(range(point_cloud.shape[2]), num_samples)
        return point_cloud[:, :, idx]

    def forward(self, sparse, normal, Kcam):
        B, _, H, W = sparse.shape
        
        xx, yy = torch.meshgrid(torch.arange(W, device=sparse.device), torch.arange(H, device=sparse.device),
                                indexing='xy')
        Pxyz = self.pinv(sparse, Kcam, xx, yy).view(sparse.shape[0], 3, sparse.shape[2], sparse.shape[3])

        Pxyz_mask = sparse > 0

        if B > 1:
            collect_xyz = []
            for i in range(B):
                Px1 = Pxyz[i:i+1,:1][Pxyz_mask[i:i+1]].view(1, 1, -1).contiguous()
                Py1 = Pxyz[i:i+1,1:2][Pxyz_mask[i:i+1]].view(1, 1, -1).contiguous()
                Pz1 = Pxyz[i:i+1,2:][Pxyz_mask[i:i+1]].view(1, 1, -1).contiguous()
                Pxn1 = normal[i:i+1,:1][Pxyz_mask[i:i+1]].view(1, 1, -1).contiguous()
                Pyn1 = normal[i:i+1,1:2][Pxyz_mask[i:i+1]].view(1, 1, -1).contiguous()
                Pzn1 = normal[i:i+1,2:][Pxyz_mask[i:i+1]].view(1, 1, -1).contiguous()
                Pxyz1 = torch.cat([Px1, Py1, Pz1, Pxn1, Pyn1, Pzn1], dim=1).contiguous()
                collect_xyz.append(Pxyz1)
            max_len = max([x.shape[2] for x in collect_xyz])
            for i in range(B):
                if collect_xyz[i].shape[2] < max_len:
                    collect_xyz[i] = torch.cat([collect_xyz[i], self.random_sample(collect_xyz[i], max_len - collect_xyz[i].shape[2])], dim=2)
            xyz = torch.cat(collect_xyz, dim=0).contiguous()
        else:
            Px = Pxyz[:,:1][Pxyz_mask].view(1, 1, -1).contiguous()
            Py = Pxyz[:,1:2][Pxyz_mask].view(1, 1, -1).contiguous()
            Pz = Pxyz[:,2:][Pxyz_mask].view(1, 1, -1).contiguous()
            Pxn = normal[:,:1][Pxyz_mask].view(1, 1, -1).contiguous()
            Pyn = normal[:,1:2][Pxyz_mask].view(1, 1, -1).contiguous()
            Pzn = normal[:,2:][Pxyz_mask].view(1, 1, -1).contiguous()
            xyz = torch.cat([Px, Py, Pz, Pxn, Pyn, Pzn], dim=1).contiguous()

        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 512)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)

        return x
    

class SEBlockFusion(nn.Module):
    def __init__(self, feature_map_channels=64, global_feature_dim=64, reduction=16):
        super(SEBlockFusion, self).__init__()
        self.feature_map_channels = feature_map_channels
        self.global_feature_dim = global_feature_dim
        
        # 全局平均池化和最大池化
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 计算权重的全连接层
        self.fc1 = nn.Linear(feature_map_channels * 3, feature_map_channels // reduction, bias=False)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.fc2 = nn.Linear(feature_map_channels // reduction, feature_map_channels, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.global_fc1 = nn.Linear(feature_map_channels * 3, feature_map_channels, bias=False)
        self.global_fc2 = nn.Linear(feature_map_channels, feature_map_channels, bias=False)
        self.global_relu = nn.LeakyReLU(0.2, inplace=True)

        self.global_down1 = nn.Linear(global_feature_dim, feature_map_channels, bias=False)
        self.global_down2 = nn.Linear(feature_map_channels, feature_map_channels, bias=False)
        self.global_sigmoid = nn.Sigmoid()

    def forward(self, x, global_feature):
        batch_size, num_channels, _, _ = x.size()
        
        # 从特征图中进行平均池化和最大池化
        avg_pool = self.global_avg_pool(x).view(batch_size, num_channels)
        max_pool = self.global_max_pool(x).view(batch_size, num_channels)

        global_down = self.global_down1(global_feature)
        global_down = self.global_relu(global_down)
        global_down = self.global_down2(global_down)

        # 将池化后的特征和全局特征拼接在一起
        concatenated_features = torch.cat([avg_pool, max_pool, global_down], dim=1)
        
        # 计算权重
        y = self.fc1(concatenated_features)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(batch_size, num_channels, 1, 1)
        
        # 调整特征图
        adjusted_feature_map = x * y.expand_as(x)
        
        # 将全局特征映射到特征图的通道数并调整
        mid_pool = self.global_fc1(concatenated_features)
        mid_pool = self.global_relu(mid_pool)
        mid_pool = self.global_fc2(mid_pool)
        mid_pool = self.global_sigmoid(mid_pool).view(batch_size, num_channels, 1, 1)

        expanded_global_feature = global_down.unsqueeze(-1).unsqueeze(-1).expand_as(adjusted_feature_map)
        expanded_global_feature = expanded_global_feature * mid_pool
        
        # 特征图和全局特征相加
        fused_feature_map = adjusted_feature_map + expanded_global_feature
        
        return fused_feature_map


class BpDist(Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, xy, idx, Valid, num, H, W):
        """
        """
        assert xy.is_contiguous()
        assert Valid.is_contiguous()
        _, Cc, M = xy.shape
        B = Valid.shape[0]
        N = H * W
        args = torch.zeros((B, num, N), dtype=torch.long, device=xy.device)
        IPCnum = torch.zeros((B, Cc, num, N), dtype=xy.dtype, device=xy.device)
        for b in range(B):
            Pc = torch.masked_select(xy, Valid[b:b + 1].view(1, 1, N)).reshape(1, 2, -1)
            BpOps.Dist(Pc, IPCnum[b:b + 1], args[b:b + 1], H, W)
            idx_valid = torch.masked_select(idx, Valid[b:b + 1].view(1, 1, N))
            args[b:b + 1] = torch.index_select(idx_valid, 0, args[b:b + 1].reshape(-1)).reshape(1, num, N)
        return IPCnum, args

    @staticmethod
    @custom_bwd
    def backward(ctx, ga=None, gb=None):
        return None, None, None, None


class BpConvLocal(Function):
    @staticmethod
    def forward(ctx, input, weight):
        assert input.is_contiguous()
        assert weight.is_contiguous()
        ctx.save_for_backward(input, weight)
        output = BpOps.Conv2dLocal_F(input, weight)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_input, grad_weight = BpOps.Conv2dLocal_B(input, weight, grad_output)
        return grad_input, grad_weight


bpdist = BpDist.apply
bpconvlocal = BpConvLocal.apply


class Basic2d(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=None, kernel_size=3, padding=1, padding_mode='zeros',
                 act=nn.ReLU, stride=1):
        super().__init__()
        if norm_layer:
            conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                             stride=stride, padding=padding, bias=False, padding_mode=padding_mode)
        else:
            conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                             stride=stride, padding=padding, bias=True, padding_mode=padding_mode)
        self.conv = nn.Sequential(OrderedDict([('conv', conv)]))
        if norm_layer:
            self.conv.add_module('bn', norm_layer(out_channels))
        self.conv.add_module('relu', act())

    def forward(self, x):
        out = self.conv(x)
        return out


class GenKernel(nn.Module):
    def __init__(self, in_channels, pk, norm_layer=nn.BatchNorm2d, act=nn.ReLU, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.conv = nn.Sequential(
            Basic2d(in_channels, in_channels, norm_layer=norm_layer, act=act),
            Basic2d(in_channels, pk * pk - 1, norm_layer=norm_layer, act=nn.Identity),
        )

    def forward(self, fout):
        weight = self.conv(fout)
        weight_sum = torch.sum(weight.abs(), dim=1, keepdim=True)
        weight = torch.div(weight, weight_sum + self.eps)
        weight_mid = 1 - torch.sum(weight, dim=1, keepdim=True)
        weight_pre, weight_post = torch.split(weight, [weight.shape[1] // 2, weight.shape[1] // 2], dim=1)
        weight = torch.cat([weight_pre, weight_mid, weight_post], dim=1).contiguous()
        return weight


class CSPN(nn.Module):
    """
    implementation of CSPN++
    """

    def __init__(self, in_channels, pt, norm_layer=nn.BatchNorm2d, act=nn.ReLU, eps=1e-6):
        super().__init__()
        self.pt = pt
        self.weight3x3 = GenKernel(in_channels, 3, norm_layer=norm_layer, act=act, eps=eps)
        self.weight5x5 = GenKernel(in_channels, 5, norm_layer=norm_layer, act=act, eps=eps)
        self.weight7x7 = GenKernel(in_channels, 7, norm_layer=norm_layer, act=act, eps=eps)
        self.convmask = nn.Sequential(
            Basic2d(in_channels, in_channels, norm_layer=norm_layer, act=act),
            Basic2d(in_channels, 3, norm_layer=None, act=nn.Sigmoid),
        )
        self.convck = nn.Sequential(
            Basic2d(in_channels, in_channels, norm_layer=norm_layer, act=act),
            Basic2d(in_channels, 3, norm_layer=None, act=functools.partial(nn.Softmax, dim=1)),
        )
        self.convct = nn.Sequential(
            Basic2d(in_channels + 3, in_channels, norm_layer=norm_layer, act=act),
            Basic2d(in_channels, 3, norm_layer=None, act=functools.partial(nn.Softmax, dim=1)),
        )

    @custom_fwd(cast_inputs=torch.float32)
    def forward(self, fout, hn, h0):
        weight3x3 = self.weight3x3(fout)
        weight5x5 = self.weight5x5(fout)
        weight7x7 = self.weight7x7(fout)
        mask3x3, mask5x5, mask7x7 = torch.split(self.convmask(fout) * (h0 > 1e-3).float(), 1, dim=1)
        conf3x3, conf5x5, conf7x7 = torch.split(self.convck(fout), 1, dim=1)
        hn3x3 = hn5x5 = hn7x7 = hn
        hns = [hn, ]
        for i in range(self.pt):
            hn3x3 = (1. - mask3x3) * bpconvlocal(hn3x3, weight3x3) + mask3x3 * h0
            hn5x5 = (1. - mask5x5) * bpconvlocal(hn5x5, weight5x5) + mask5x5 * h0
            hn7x7 = (1. - mask7x7) * bpconvlocal(hn7x7, weight7x7) + mask7x7 * h0
            if i == self.pt // 2 - 1:
                hns.append(conf3x3 * hn3x3 + conf5x5 * hn5x5 + conf7x7 * hn7x7)
        hns.append(conf3x3 * hn3x3 + conf5x5 * hn5x5 + conf7x7 * hn7x7)
        hns = torch.cat(hns, dim=1)
        wt = self.convct(torch.cat([fout, hns], dim=1))
        hn = torch.sum(wt * hns, dim=1, keepdim=True)
        return hn


def Conv1x1(in_planes, out_planes, stride=1, bias=False, groups=1, dilation=1, padding_mode='zeros'):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)


def Conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, padding_mode='zeros', bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, padding_mode=padding_mode, groups=groups, bias=bias, dilation=dilation)
    

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None, padding_mode='zeros', act=nn.ReLU,
                 last=True, drop_path=0.0):
        super().__init__()
        bias = False
        if norm_layer is None:
            bias = True
            norm_layer = nn.Identity
        self.conv1 = Conv3x3(inplanes, planes, stride, padding_mode=padding_mode, bias=bias)
        self.bn1 = norm_layer(planes)
        self.relu1 = act()
        self.conv2 = Conv3x3(planes, planes, padding_mode=padding_mode, bias=bias)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.last = last
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if last:
            self.relu2 = act()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.drop_path(out) + identity
        if self.last:
            out = self.relu2(out)
        return out
    

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


class Basic2dTrans(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=None, act=nn.ReLU):
        super().__init__()
        if norm_layer is None:
            bias = True
            norm_layer = nn.Identity
        else:
            bias = False
        self.conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4,
                                       stride=2, padding=1, bias=bias)
        self.bn = norm_layer(out_channels)
        self.relu = act()

    def forward(self, x):
        out = self.conv(x.contiguous())
        out = self.bn(out)
        out = self.relu(out)
        return out


class UpCC(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, norm_layer=None, kernel_size=3, padding=1,
                 padding_mode='zeros', act=nn.ReLU):
        super().__init__()
        self.upf = Basic2dTrans(in_channels, out_channels, norm_layer=norm_layer, act=act)
        self.conv = Basic2d(mid_channels + out_channels, out_channels,
                            norm_layer=norm_layer, kernel_size=kernel_size,
                            padding=padding, padding_mode=padding_mode, act=act)

    def forward(self, x, y):
        """
        """
        out = self.upf(x)
        fout = torch.cat([out, y], dim=1)
        fout = self.conv(fout)
        return fout


class UBNet(nn.Module):
    def __init__(self, inplanes, dplanes=1, norm_layer=nn.BatchNorm2d, padding_mode='zeros', act=nn.ReLU,
                 blocknum=2, uplayer=UpCC, depth=1, block=BasicBlock, drop_path=0.):
        super().__init__()
        self._norm_layer = norm_layer
        self._act = act
        self._padding_mode = padding_mode
        bc = inplanes // 2
        self.inplanes = bc * 2
        self._img_dpc = 0
        self._img_dprs = np.linspace(0, drop_path, blocknum * (1 + depth))
        encoder_list = [nn.Sequential(
            Basic2d(inplanes + dplanes, bc * 2, norm_layer=norm_layer, kernel_size=3, padding=1,
                    padding_mode=padding_mode, act=act),
            self._make_layer(block, bc * 2, blocknum, stride=1)
        ), ]
        decoder_list = []
        in_channels = bc * 2
        for i in range(depth):
            self.inplanes = in_channels
            out_channels = min(in_channels * 2, 256)
            encoder_list.append(self._make_layer(block, out_channels, blocknum, stride=2))
            decoder_list.append(uplayer(out_channels, in_channels, in_channels, norm_layer=norm_layer))
            in_channels = min(in_channels * 2, 256)
        self.encoder = nn.ModuleList(encoder_list)
        self.decoder = nn.ModuleList(decoder_list)

    def forward(self, x, d):
        feat = []
        if d is not None:
            x = torch.cat([x, d], dim=1)
        for layer in self.encoder:
            x = layer(x)
            feat.append(x)
        out = feat[-1]
        for idx in range(len(feat) - 2, -1, -1):
            out = self.decoder[idx](out, feat[idx])
        return out

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        act = self._act
        padding_mode = self._padding_mode
        downsample = None
        if norm_layer is None:
            bias = True
            norm_layer = nn.Identity
        else:
            bias = False
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                Conv1x1(self.inplanes, planes * block.expansion, stride, bias=bias),
                norm_layer(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer, act=act, padding_mode=padding_mode,
                            drop_path=self._img_dprs[self._img_dpc]))
        self._img_dpc += 1
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer, act=act, padding_mode=padding_mode,
                                drop_path=self._img_dprs[self._img_dpc]))
            self._img_dpc += 1

        return nn.Sequential(*layers)
    

class UpCat(nn.Module):
    def __init__(self, in_channels, out_channels, level=None, norm_layer=nn.BatchNorm2d, kernel_size=3, padding=1,
                 padding_mode='zeros', act=nn.ReLU):
        super().__init__()
        self.dim_level = [32, 64, 128, 128, 256]
        self.upf = Basic2dTrans(in_channels + 1, out_channels, norm_layer=norm_layer, act=act)
        self.conv = Basic2d(out_channels + self.dim_level[level], out_channels,
                            norm_layer=norm_layer, kernel_size=kernel_size,
                            padding=padding, padding_mode=padding_mode, act=act)

    def forward(self, y, x, d):
        """
        x is
        """
        fout = self.upf(torch.cat([x, d], dim=1))
        fout = self.conv(torch.cat([fout, y], dim=1))
        return fout


class UpCatFull(nn.Module):
    def __init__(self, in_channels, out_channels, level=None, norm_layer=nn.BatchNorm2d, kernel_size=3, padding=1,
                 padding_mode='zeros', act=nn.ReLU):
        super().__init__()
        self.dim_level = [32, 64, 128, 128, 256]
        self.dim_full = [64, 32, 16, 8]
        self.dim_full_in = [96, 96, 128, 256]
        self.upf0 = Basic2dTrans(self.dim_full_in[0] + 1, self.dim_full[0], norm_layer=norm_layer, act=act)
        self.upf1 = nn.Sequential(
            Basic2dTrans(self.dim_full_in[1] + 1, self.dim_full[1], norm_layer=norm_layer, act=act),
            Basic2dTrans(self.dim_full[1], self.dim_full[1], norm_layer=norm_layer, act=act),
        )
        self.upf2 = nn.Sequential(
            Basic2dTrans(self.dim_full_in[2] + 1, self.dim_full[2], norm_layer=norm_layer, act=act),
            Basic2dTrans(self.dim_full[2], self.dim_full[2], norm_layer=norm_layer, act=act),
            Basic2dTrans(self.dim_full[2], self.dim_full[2], norm_layer=norm_layer, act=act),
        )
        self.upf3 = nn.Sequential(
            Basic2dTrans(self.dim_full_in[3] + 1, self.dim_full[3], norm_layer=norm_layer, act=act),
            Basic2dTrans(self.dim_full[3], self.dim_full[3], norm_layer=norm_layer, act=act),
            Basic2dTrans(self.dim_full[3], self.dim_full[3], norm_layer=norm_layer, act=act),
            Basic2dTrans(self.dim_full[3], self.dim_full[3], norm_layer=norm_layer, act=act),
        )
        self.conv1 = Basic2d(self.dim_full[0] + self.dim_full[1] + self.dim_full[2] + self.dim_full[3] + self.dim_level[level], int(out_channels*1.5),
                            norm_layer=norm_layer, kernel_size=kernel_size,
                            padding=padding, padding_mode=padding_mode, act=act)
        self.conv2 = Basic2d(int(out_channels*1.5), out_channels, norm_layer=norm_layer, kernel_size=kernel_size,
                            padding=padding, padding_mode=padding_mode, act=act)

    def forward(self, y, x, d, c1, c2, c3):
        """
        x is
        """
        fout0 = self.upf0(torch.cat([x, d], dim=1))
        fout1 = self.upf1(c1)
        fout2 = self.upf2(c2)
        fout3 = self.upf3(c3)
        fout = self.conv1(torch.cat([fout0, fout1, fout2, fout3, y], dim=1))
        fout = self.conv2(fout)
        return fout
    

class Coef(nn.Module):
    """
    """
    def __init__(self, in_channels, out_channels=3, kernel_size=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=1, padding=padding, bias=True)

    def forward(self, x):
        feat = self.conv(x)
        XF, XB, XW = torch.split(feat, [1, 1, 1], dim=1)
        return XF, XB, XW


class Dist(nn.Module):
    """
    """

    def __init__(self, num):
        super().__init__()
        """
        """
        self.num = num

    def forward(self, S, xx, yy):
        """
        """
        num = self.num
        B, _, height, width = S.shape
        N = height * width
        S = S.reshape(B, 1, N)
        Valid = (S > 1e-6)
        xy = torch.stack((xx, yy), axis=0).reshape(1, 2, -1).float()
        idx = torch.arange(N, device=S.device).reshape(1, 1, N)
        Ofnum, args = bpdist(xy, idx, Valid, num, height, width)
        return Ofnum, args
    

class Permute(nn.Module):
    def __init__(self, in_channels, out_channels=1, stride=2, norm_layer=nn.BatchNorm2d, act=nn.ReLU):
        super().__init__()
        self.stride = stride
        self.out_channels = out_channels
        self.conv = nn.Sequential(
            Basic2d(in_channels=in_channels, out_channels=in_channels, norm_layer=norm_layer, act=act, kernel_size=1,
                    padding=0),
            Basic2d(in_channels=in_channels, out_channels=in_channels, norm_layer=norm_layer, act=act, kernel_size=1,
                    padding=0),
            Conv1x1(in_channels, out_channels * stride ** 2, bias=True),
            Rearrange('b (c h2 w2) h w -> b c (h h2) (w w2)', c=out_channels, h2=stride, w2=stride),
        )

    def forward(self, x):
        """
        """
        fout = self.conv(x)
        return fout
    

class WPool(nn.Module):
    def __init__(self, in_ch, level, drift=1e6):
        super().__init__()
        self.level = level
        self.drift = drift
        self.permute = Permute(in_ch, stride=2 ** level)

    def forward(self, S, fout):
        W = self.permute(fout)
        size = int(2 ** self.level)
        M = (S > 1e-3).float()
        with torch.no_grad():
            maxW = F.max_pool2d((W + self.drift) * M, size, stride=[size, size]) - self.drift
            maxW = F.upsample_nearest(maxW, scale_factor=size) * M
        expW = torch.exp(W * M - maxW) * M
        avgS = F.avg_pool2d(S * expW, kernel_size=size, stride=size)
        avgexpW = F.avg_pool2d(expW, kernel_size=size, stride=size)
        Sp = avgS / (avgexpW + 1e-6)
        return Sp
    

class WPool_F(nn.Module):
    def __init__(self, in_ch, level, drift=1e6):
        super().__init__()
        self.level = level
        self.drift = drift
        self.permute = Permute(in_ch, stride=2 ** level)

    def forward(self, S, fout):
        W = self.permute(fout)
        size = int(2 ** self.level)
        with torch.no_grad():
            maxW = F.max_pool2d((W + self.drift), size, stride=[size, size]) - self.drift
            maxW = F.upsample_nearest(maxW, scale_factor=size)
        expW = torch.exp(W - maxW)
        avgS = F.avg_pool2d(S * expW, kernel_size=size, stride=size)
        avgexpW = F.avg_pool2d(expW, kernel_size=size, stride=size)
        Sp = avgS / (avgexpW + 1e-6)
        return Sp


class Ident(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, *args):
        return args[0]
    

class Prop(nn.Module):
    """
    """

    def __init__(self, Cfi, Cfp=3, Cfo=2, act=nn.GELU, norm_layer=nn.BatchNorm2d):
        super().__init__()
        """
        """
        self.dist = lambda x: (x * x).sum(1)
        Ct = Cfo + Cfi + Cfi + Cfp
        self.convXF = nn.Sequential(
            Basic2d(in_channels=Ct, out_channels=Cfi, norm_layer=norm_layer, act=act, kernel_size=1,
                    padding=0),
            Basic2d(in_channels=Cfi, out_channels=Cfi, norm_layer=norm_layer, act=act, kernel_size=1,
                    padding=0),
        )
        self.convXL = nn.Sequential(
            Basic2d(in_channels=Cfi, out_channels=Cfi, norm_layer=norm_layer, act=act, kernel_size=1,
                    padding=0),
            Basic2d(in_channels=Cfi, out_channels=Cfi, norm_layer=norm_layer, act=nn.Identity, kernel_size=1,
                    padding=0),
        )
        self.act = act()
        self.coef = Coef(Cfi, 3)

    def forward(self, If, Pf, Ofnum, args):
        """
        """
        num = args.shape[-2]
        B, Cfi, H, W = If.shape
        N = H * W
        B, Cfp, M = Pf.shape
        If = If.view(B, Cfi, 1, N)
        Pf = Pf.view(B, Cfp, 1, M)
        Ifnum = If.expand(B, Cfi, num, N)  ## Ifnum is BxCfixnumxN
        IPfnum = torch.gather(
            input=If.expand(B, Cfi, num, N),
            dim=-1,
            index=args.view(B, 1, num, N).expand(B, Cfi, num, N))  ## IPfnum is BxCfixnumxN
        Pfnum = torch.gather(
            input=Pf.expand(B, Cfp, num, M),
            dim=-1,
            index=args.view(B, 1, num, N).expand(B, Cfp, num, N))  ## Pfnum is BxCfpxnumxN
        X = torch.cat([Ifnum, IPfnum, Pfnum, Ofnum], dim=1)
        XF = self.convXF(X)
        XF = self.act(XF + self.convXL(XF))
        Alpha, Beta, Omega = self.coef(XF)
        Omega = torch.softmax(Omega, dim=2)
        dout = torch.sum(((Alpha + 1) * Pfnum[:, -1:] + Beta) * Omega, dim=2, keepdim=True)
        return dout.view(B, 1, H, W)


class PMP(nn.Module):
    """
    Pre+MF+Post
    """

    def __init__(self, level, in_ch, out_ch, drop_path, fuse_prob, up=True, pool=True, block=BasicBlock):
        super().__init__()
        self.drift = 1e6
        self._norm_layer = nn.BatchNorm2d
        self._padding_mode = 'zeros'
        self._img_dpc = 0
        self._img_dprs = np.linspace(0, drop_path, 12)

        self.level = level
        if up:
            self.upcat = UpCat(in_ch, out_ch, level=level)
        else:
            self.upcat = Ident()
        if pool:
            self.wpool_s = WPool(out_ch, level=level)
            self.wpool_f = WPool_F(out_ch, level=level)
        else:
            self.wpool_s = Ident()
            self.wpool_f = Ident()
        if level == 0:
            self.weight_offset = BasicDepthEncoder(kernel_size=3, norm_layer=nn.BatchNorm2d)
            self.Post_process = pa_spn()
        # self.dist = Dist(num=4)
        # self.prop = Prop(out_ch)
        self.fuse = UBNet(out_ch, dplanes=3, blocknum=2, depth=5 - level, drop_path=drop_path)
        self.conv = Conv3x3(out_ch, 1, bias=True)
        self.cspn = CSPN(out_ch, pt=2 * (6 - level))
        # self.global_feat = GlobalPoint()
        self.sefuse = SEBlockFusion(out_ch, 64, 1)
        self.fuse_drop = DropPath(fuse_prob) if fuse_prob > 0. else nn.Identity()

    def pinv(self, S, K, xx, yy):
        fx, fy, cx, cy = K[:, 0:1, 0:1], K[:, 1:2, 1:2], K[:, 0:1, 2:3], K[:, 1:2, 2:3]
        S = S.view(S.shape[0], 1, -1)
        xx = xx.reshape(1, 1, -1)
        yy = yy.reshape(1, 1, -1)
        Px = S * (xx - cx) / fx
        Py = S * (yy - cy) / fy
        Pz = S
        Pxyz = torch.cat([Px, Py, Pz], dim=1).contiguous()
        return Pxyz

    def forward(self, context, fout, dout, to_res, sparse, global_feat, Kcam, c1=None, c2=None, c3=None):

        fout = self.upcat(context, fout, dout)

        Sp = self.wpool_s(sparse, fout)
        to_res = self.wpool_f(to_res, fout)

        if self.level == 0:
            weight, offset = self.weight_offset(to_res, fout)
            to_res = self.Post_process(to_res, weight, offset)

        Kp = Kcam.clone()
        Kp[:, :2] = Kp[:, :2] / 2 ** self.level

        B, _, height, width = to_res.shape
        xx, yy = torch.meshgrid(torch.arange(width, device=to_res.device), torch.arange(height, device=to_res.device),
                                indexing='xy')
        
        Pxyz = self.pinv(to_res, Kp, xx, yy).view(to_res.shape[0], 3, to_res.shape[2], to_res.shape[3])

        fout = self.fuse(fout, Pxyz)
        fout = self.sefuse(fout, global_feat)

        fout = self.fuse_drop(fout)
        res = self.conv(fout)
        dout = to_res + res
     
        dout = self.cspn(fout, dout, Sp)
        return fout, dout
    
    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        padding_mode = self._padding_mode
        downsample = None
        if norm_layer is None:
            bias = True
            norm_layer = nn.Identity
        else:
            bias = False
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                Conv1x1(self.inplanes, planes * block.expansion, stride, bias=bias),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, norm_layer=norm_layer, padding_mode=padding_mode,
                  drop_path=self._img_dprs[self._img_dpc]))
        self._img_dpc += 1
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer, padding_mode=padding_mode,
                                drop_path=self._img_dprs[self._img_dpc]))
            self._img_dpc += 1

        return nn.Sequential(*layers)


class Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        #self.conv = conv_module(in_channels, out_channels, kernel_size, **kwargs)
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, **kwargs)

    def forward(self, x):
        return F.conv2d(x, self.weight, bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups) 


class LayerNorm2d(nn.LayerNorm):
    def __init__(self, dim):
        super(LayerNorm2d, self).__init__(dim)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = super(LayerNorm2d, self).forward(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


def interpolate_float32(x, size=None, scale_factor=None, mode='nearest', align_corners=None):
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=False):
        return F.interpolate(x.float(), size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners)


class Re_block(nn.Module):

    def __init__(self, input, output) -> None:
        super(Re_block, self).__init__()
        midput = int(input / 4)
        self.re = nn.Sequential(
            nn.GroupNorm(4, input),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(input, midput, (1,1), 1),
            nn.GroupNorm(4, midput),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(midput, midput, (3,3), 1, 1),
            nn.GroupNorm(4, midput),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(midput, output, (1,1), 1)
        )
        if(input == output):
            self.resize = None
        else:
            self.resize = nn.Conv2d(input, output, (1,1), 1, 0)

    def forward(self, x):
        out = self.re(x)
        out = out + (x if self.resize == None else self.resize(x))
        return out


class mid_block(nn.Module):

    def __init__(self, midput) -> None:
        super(mid_block, self).__init__()
        self.mido = nn.Sequential(
            Re_block(midput, midput),
            Re_block(midput, midput)
        )

    def forward(self, x):
        out = self.mido(x)
        return out


class mid_adaptive_block(nn.Module):

    def __init__(self, input_num, output_num) -> None:
        super(mid_adaptive_block, self).__init__()
        self.mido = nn.Sequential(
            Re_block(input_num, output_num),
            Re_block(output_num, output_num)
        )

    def forward(self, x):
        out = self.mido(x)
        return out


class path_block(nn.Module):

    def __init__(self, midput, num_ch) -> None:
        super(path_block, self).__init__()
        self.mido = nn.Sequential()
        for i in range(num_ch):
            self.mido.add_module('path_block'+str(i), Re_block(midput, midput))

    def forward(self, x):
        out = self.mido(x)
        return out


class sparse_encoder(nn.Module):

    def __init__(self):
        super(sparse_encoder, self).__init__() 
        self.conv1 = nn.Conv2d(2, 64, kernel_size=5, stride=1, padding=2)
        self.encode1 = mid_block(64)
        self.encode2 = mid_block(96)
        self.encode3 = mid_block(128)
        self.encode4 = mid_block(256)
        self.encode5 = mid_block(256)
        self.mid_encode = mid_block(256)
        self.up_encode1 = mid_adaptive_block(256*2, 256)
        self.up_encode2 = mid_adaptive_block(256*2, 256)
        self.up_encode3 = mid_adaptive_block(128*2, 128)
        self.up_encode4 = mid_adaptive_block(96*2, 96)
        self.up_encode5 = mid_adaptive_block(96+64, 96)
        self.down1 = nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1)
        self.down2 = nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1)
        self.down3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.down4 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.up1 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up3 = nn.ConvTranspose2d(128, 96, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up4 = nn.ConvTranspose2d(96, 96, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, sparse_input):
        
        sparse_input_feat = self.conv1(sparse_input)

        sparse_feat_tmp1 = self.encode1(sparse_input_feat)
        sparse_feat_tmp2 = self.encode2(self.down1(sparse_feat_tmp1))
        sparse_feat_tmp3 = self.encode3(self.down2(sparse_feat_tmp2))
        sparse_feat_tmp4 = self.encode4(self.down3(sparse_feat_tmp3))
        sparse_feat_tmp5 = self.encode5(self.down4(sparse_feat_tmp4))

        sparse_feat_mid = self.mid_encode(sparse_feat_tmp5)

        sparse_feat5 = self.up_encode1(torch.cat((sparse_feat_mid, sparse_feat_tmp5), 1))
        sparse_feat4 = self.up_encode2(torch.cat((self.up1(sparse_feat5), sparse_feat_tmp4), 1))
        sparse_feat3 = self.up_encode3(torch.cat((self.up2(sparse_feat4), sparse_feat_tmp3), 1))
        sparse_feat2 = self.up_encode4(torch.cat((self.up3(sparse_feat3), sparse_feat_tmp2), 1))
        sparse_feat1 = self.up_encode5(torch.cat((self.up4(sparse_feat2), sparse_feat_tmp1), 1))

        return sparse_feat1, sparse_feat2, sparse_feat3, sparse_feat4, sparse_feat5


class rgb_encoder(nn.Module):

    def __init__(self):
        super(rgb_encoder, self).__init__() 
        self.conv1 = nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=1)
        self.encode1 = mid_block(96)
        self.encode2 = mid_block(96)
        self.encode3 = mid_block(128)
        self.encode4 = mid_block(256)
        self.encode5 = mid_block(256)
        self.mid_encode = mid_block(256)
        self.up_encode1 = mid_adaptive_block(256*2, 256)
        self.up_encode2 = mid_adaptive_block(256*2, 256)
        self.up_encode3 = mid_adaptive_block(128*2, 128)
        self.up_encode4 = mid_adaptive_block(96*2, 96)
        self.up_encode5 = mid_adaptive_block(96*2, 96)
        self.down1 = nn.Conv2d(96, 96, kernel_size=3, stride=2, padding=1)
        self.down2 = nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1)
        self.down3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.down4 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.up1 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up3 = nn.ConvTranspose2d(128, 96, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up4 = nn.ConvTranspose2d(96, 96, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, rgb_input):
        
        rgb_input_feat = self.conv1(rgb_input)

        rgb_feat_tmp1 = self.encode1(rgb_input_feat)
        rgb_feat_tmp2 = self.encode2(self.down1(rgb_feat_tmp1))
        rgb_feat_tmp3 = self.encode3(self.down2(rgb_feat_tmp2))
        rgb_feat_tmp4 = self.encode4(self.down3(rgb_feat_tmp3))
        rgb_feat_tmp5 = self.encode5(self.down4(rgb_feat_tmp4))

        rgb_feat_mid = self.mid_encode(rgb_feat_tmp5)

        rgb_feat5 = self.up_encode1(torch.cat((rgb_feat_mid, rgb_feat_tmp5), 1))
        rgb_feat4 = self.up_encode2(torch.cat((self.up1(rgb_feat5), rgb_feat_tmp4), 1))
        rgb_feat3 = self.up_encode3(torch.cat((self.up2(rgb_feat4), rgb_feat_tmp3), 1))
        rgb_feat2 = self.up_encode4(torch.cat((self.up3(rgb_feat3), rgb_feat_tmp2), 1))
        rgb_feat1 = self.up_encode5(torch.cat((self.up4(rgb_feat2), rgb_feat_tmp1), 1))

        return rgb_feat1, rgb_feat2, rgb_feat3, rgb_feat4, rgb_feat5


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1, tuning_mode=None):
        super(ResidualBlock, self).__init__()
  
        self.conv1 = Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.BatchNorm2d(planes)
        
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'layer':
            self.norm1 = LayerNorm2d(planes)
            self.norm2 = LayerNorm2d(planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = LayerNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.Sequential()

        if stride == 1 and in_planes == planes:
            self.downsample = None
        
        else:    
            self.downsample = nn.Sequential(
                Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)
            
    def forward(self, x):
        y = x
        y = self.conv1(y)
        y = self.norm1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.norm2(y)
        y = self.relu(y)

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)


class ConvGRU(nn.Module):
    def __init__(self, hidden_dim, input_dim, kernel_size=3, tuning_mode=None):
        super(ConvGRU, self).__init__()
        self.convz = Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size//2)
        self.convr = Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size//2)
        self.convq = Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size//2)

    def forward(self, h, cz, cr, cq, *x_list):
        x = torch.cat(x_list, dim=1)
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid((self.convz(hx) + cz))
        r = torch.sigmoid((self.convr(hx) + cr))
        q = torch.tanh((self.convq(torch.cat([r*h, x], dim=1)) + cq))

        h = (1-z) * h + z * q
        return h


def pool2x(x):
    return F.avg_pool2d(x, 3, stride=2, padding=1)

def pool4x(x):
    return F.avg_pool2d(x, 5, stride=4, padding=1)

def interp(x, dest):
    interp_args = {'mode': 'bilinear', 'align_corners': True}
    return interpolate_float32(x, dest.shape[2:], **interp_args)


class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim_depth=1, output_dim_norm=3, tuning_mode=None):
        super(FlowHead, self).__init__()
        self.conv1d = Conv2d(input_dim, hidden_dim // 2, 3, padding=1)
        self.conv2d = Conv2d(hidden_dim // 2, output_dim_depth, 3, padding=1)

        self.conv1n = Conv2d(input_dim-24, hidden_dim // 2, 3, padding=1)
        self.conv2n = Conv2d(hidden_dim // 2, output_dim_norm, 3, padding=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        depth = self.conv2d(self.relu(self.conv1d(x)))
        normal = self.conv2n(self.relu(self.conv1n(x[:,:-24])))
        return torch.cat((depth, normal), dim=1)


class EncodeHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim_depth=1, output_dim_norm=3, tuning_mode=None):
        super(EncodeHead, self).__init__()
        self.conv1d = Conv2d(output_dim_depth, hidden_dim // 2, 3, padding=1)
        self.conv2d = Conv2d(hidden_dim // 2, input_dim-30, 3, padding=1)

        self.conv1n = Conv2d(output_dim_norm, hidden_dim // 2, 3, padding=1)
        self.conv2n = Conv2d(hidden_dim // 2, 30, 3, padding=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        depth = self.conv2d(self.relu(self.conv1d(x[:,:1])))
        normal = self.conv2n(self.relu(self.conv1n(x[:,1:])))
        return torch.cat([depth, normal], dim=1)


class ConvBlock(nn.Module):
    # reimplementation of DPT
    def __init__(self, channels, tuning_mode=None):
        super(ConvBlock, self).__init__()

        self.act = nn.ReLU(inplace=False)
        self.conv1 = Conv2d(
            channels,
            channels,
            kernel_size=3, 
            stride=1,
            padding=1,
        )
        self.conv2 = Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x):
        out = self.act(x)
        out = self.conv1(out)
        out = self.act(out)
        out = self.conv2(out)
        return x + out


class ContextFeatureEncoder(nn.Module):
    '''
    Encoder features are used to:
        1. initialize the hidden state of the update operator 
        2. and also injected into the GRU during each iteration of the update operator
    '''
    def __init__(self, in_dim, output_dim, tuning_mode=None):
        '''
        in_dim     = [x4, x8, x16, x32]
        output_dim = [hindden_dims,   context_dims]
                    [[x4,x8,x16,x32],[x4,x8,x16,x32]]
        '''
        super().__init__()

        output_list = []
        for dim in output_dim:
            conv_out = nn.Sequential(
                ResidualBlock(in_dim[0], dim[0], 'layer', stride=1, tuning_mode=tuning_mode),
                Conv2d(dim[0], dim[0], 3, padding=1))
            output_list.append(conv_out)

        self.outputs04 = nn.ModuleList(output_list)

        output_list = []
        for dim in output_dim:
            conv_out = nn.Sequential(
                ResidualBlock(in_dim[1], dim[1], 'layer', stride=1, tuning_mode=tuning_mode),
                Conv2d(dim[1], dim[1], 3, padding=1))
            output_list.append(conv_out)

        self.outputs08 = nn.ModuleList(output_list)

        output_list = []
        for dim in output_dim:
            conv_out = nn.Sequential(
                ResidualBlock(in_dim[2], dim[2], 'layer', stride=1, tuning_mode=tuning_mode),
                Conv2d(dim[2], dim[2], 3, padding=1))
            output_list.append(conv_out)

        self.outputs16 = nn.ModuleList(output_list)

        # output_list = []
        # for dim in output_dim:
        #     conv_out = Conv2dLoRA(in_dim[3], dim[3], 3, padding=1)
        #     output_list.append(conv_out)

        # self.outputs32 = nn.ModuleList(output_list)

    def forward(self, encoder_features):
        # x_4, x_8, x_16, x_32 = encoder_features
        x_4, x_8, x_16, _, _ = encoder_features

        outputs04 = [f(x_4) for f in self.outputs04]
        outputs08 = [f(x_8) for f in self.outputs08]
        outputs16 = [f(x_16)for f in self.outputs16]
        # outputs32 = [f(x_32) for f in self.outputs32]

        return (outputs04, outputs08, outputs16)


class FuseBlock(nn.Module):
    # reimplementation of DPT
    def __init__(self, in_channels, out_channels, fuse=True, upsample=True, scale_factor=2, tuning_mode=None):
        super(FuseBlock, self).__init__()

        self.fuse = fuse
        self.scale_factor = scale_factor
        self.way_trunk = ConvBlock(in_channels, tuning_mode=tuning_mode)
        if self.fuse:
            self.way_branch = ConvBlock(in_channels, tuning_mode=tuning_mode)
        
        self.out_conv = Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.upsample = upsample

    def forward(self, x1, x2=None):
        if x2 is not None:
            x2 = self.way_branch(x2)
            x1 = x1 + x2

        out = self.way_trunk(x1)

        if self.upsample:
            out = interpolate_float32(
                out, scale_factor=self.scale_factor, mode="bilinear", align_corners=True
            )
        out = self.out_conv(out)
        return out


class DecoderFeature(nn.Module):
    def __init__(self, num_ch_dec=[96, 96, 128, 256, 256], tuning_mode=None):
        super(DecoderFeature, self).__init__()
        self.num_ch_dec = num_ch_dec

        self.upconv_3 = FuseBlock(
            self.num_ch_dec[4], 
            self.num_ch_dec[3], 
        fuse=False, upsample=True, tuning_mode=tuning_mode)
        
        self.upconv_2 = FuseBlock(
            self.num_ch_dec[3], 
            self.num_ch_dec[2],
        tuning_mode=tuning_mode)
        
        self.upconv_1 = FuseBlock(
            self.num_ch_dec[2], 
            self.num_ch_dec[1]+2,
        upsample=False, tuning_mode=tuning_mode)

        # self.upconv_0 = FuseBlock(
        #     self.num_ch_dec[1], 
        #     self.num_ch_dec[0] + 1,
        # )
    
    def forward(self, ref_feature):
        # x, x2, x1, x0 = ref_feature # 1/14 1/14 1/7 1/4
        x, x2, x1 = ref_feature
     
        x = self.upconv_3(x)     # 1/14
        x = self.upconv_2(x, x2) # 1/7
        x = self.upconv_1(x, x1) # 1/4
        # x = self.upconv_0(x, x0) # 4/7
        return x


def compute_depth_expectation(prob, depth_values):
    depth_values = depth_values.view(*depth_values.shape, 1, 1)
    depth = torch.sum(prob * depth_values, 1)
    return depth

def coords_grid(batch, ht, wd):
    # coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = [torch.zeros((ht, wd)) for _ in range(4)]
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)

def upflow4(flow, mode='bilinear'):
    new_size = (4 * flow.shape[2], 4 * flow.shape[3])
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=False):
        return  F.interpolate(flow, size=new_size, mode=mode, align_corners=True)

def norm_normalize(norm_out):
    norm_x, norm_y, norm_z = torch.split(norm_out, 1, dim=1)
    norm = torch.sqrt(norm_x ** 2.0 + norm_y ** 2.0 + norm_z ** 2.0) + 1e-10
    final_out = torch.cat([norm_x / norm, norm_y / norm, norm_z / norm], dim=1)
    return final_out


class BasicMultiUpdateBlock(nn.Module):
    def __init__(self, hidden_dims=[], out_dims=2, tuning_mode=None):
        super().__init__()
        self.n_gru_layers = 3
        
        encoder_output_dim = 4

        self.encode_head = EncodeHead(hidden_dims[2], hidden_dim=2*hidden_dims[2], tuning_mode=tuning_mode)
        self.gru08 = ConvGRU(hidden_dims[2], hidden_dims[2] + hidden_dims[1] * (self.n_gru_layers > 1), tuning_mode=tuning_mode)
        self.gru16 = ConvGRU(hidden_dims[1], hidden_dims[0] * (self.n_gru_layers == 3) + hidden_dims[2], tuning_mode=tuning_mode)
        self.gru32 = ConvGRU(hidden_dims[0], hidden_dims[1], tuning_mode=tuning_mode)
        self.flow_head = FlowHead(hidden_dims[2], hidden_dim=2*hidden_dims[2], tuning_mode=tuning_mode)

    def forward(self, net, inp, corr=None, flow=None, iter08=True, iter16=True, iter32=True, update=True):

        if iter32:
            net[2] = self.gru32(net[2], *(inp[2]), pool2x(net[1]))
        if iter16:
            if self.n_gru_layers > 2:
                net[1] = self.gru16(net[1], *(inp[1]), interp(pool2x(net[0]), net[1]), interp(net[2], net[1]))
            else:
                net[1] = self.gru16(net[1], *(inp[1]), interp(pool2x(net[0]), net[1]))
        if iter08:
            if corr is not None:
                motion_features = flow
            else:
                motion_features = flow
            motion_features = self.encode_head(motion_features)
            if self.n_gru_layers > 1:
                net[0] = self.gru08(net[0], *(inp[0]), motion_features, interp(net[1], net[0]))
            else:
                net[0] = self.gru08(net[0], *(inp[0]), motion_features)

        if not update:
            return net

        delta_flow = self.flow_head(net[0])

        return net, delta_flow
    

class DepthHead(nn.Module):

    def __init__(self, inter_dim, block=BasicBlock, norm_layer=nn.BatchNorm2d):
        super().__init__()

        self.convd1 = Basic2d(1, int(inter_dim*3/4), norm_layer=None, kernel_size=3, padding=1)
        self.convd2 = Basic2d(int(inter_dim*3/4), int(inter_dim*3/4), norm_layer=None, kernel_size=3, padding=1)

        self.convn1 = Basic2d(3, int(inter_dim/4), norm_layer=None, kernel_size=3, padding=1)
        self.convn2 = Basic2d(int(inter_dim/4), int(inter_dim/4), norm_layer=None, kernel_size=3, padding=1)

        self.convf1 = Basic2d(inter_dim, inter_dim, norm_layer=None, kernel_size=3, padding=1)
        self.convf2 = Basic2d(inter_dim, inter_dim, norm_layer=None, kernel_size=3, padding=1)

        self.convp1 = Basic2d(1, inter_dim, norm_layer=None, kernel_size=3, padding=1)
        self.convp2 = Basic2d(inter_dim, inter_dim, norm_layer=None, kernel_size=3, padding=1)

        self.convs1 = Basic2d(2, inter_dim, norm_layer=None, kernel_size=3, padding=1)
        self.convs2 = Basic2d(inter_dim, inter_dim, norm_layer=None, kernel_size=3, padding=1)

        self.conv = Basic2d(inter_dim * 4, inter_dim * 2, norm_layer=None, kernel_size=3, padding=1)
        self.ref = block(inter_dim * 2, inter_dim * 2, norm_layer=norm_layer, act=False)

        self.conv_bases = nn.Sequential(
            nn.Conv2d(inter_dim * 2, inter_dim, kernel_size=1, padding=0, stride=1),
            # nn.BatchNorm2d(bc),
            # nn.Tanh(),
            nn.Conv2d(inter_dim, 1, kernel_size=1, padding=0),
            # nn.BatchNorm2d(bases_size),
            # nn.Tanh()
        )

    def forward(self, context, depth, normal, dep_ip, sparse):

        d1 = self.convd1(depth)
        d2 = self.convd2(d1)

        n1 = self.convn1(normal)
        n2 = self.convn2(n1)

        f1 = self.convf1(context)
        f2 = self.convf2(f1)

        p1 = self.convp1(dep_ip)
        p2 = self.convp2(p1)

        s1 = self.convs1(sparse)
        s2 = self.convs2(s1)

        input_feature = torch.cat((d2, n2, f2, p2, s2), dim=1)
        input_feature = self.conv(input_feature)
        feature = self.ref(input_feature)

        out = self.conv_bases(feature)

        out = out + dep_ip

        return out


class Completion3D(nn.Module):
    
    def __init__(self) -> None:
        super(Completion3D, self).__init__()
        self.full_encoder = Encoder_LRRU()

        self.in_channels = [1024, 1024, 1024, 1024]
        self.feature_channels = [96, 96, 128, 256, 256] # [2/7, 1/7, 1/14, 1/14]
        self.decoder_channels = [96, 96, 128, 256, 256] # [-, 1/4, 1/7, 1/14, 1/14]
        self.use_cls_token = True
        self.up_scale = True
        self.num_register_tokens = 6
        self.min_val = 1.0
        self.max_val = 100.0
        self.regress_scale = 100.0

        self.tuning_mode = tuning_mode = None

        self.hidden_dims = self.context_dims = [96, 96, 128]
        self.n_gru_layers = 3
        self.n_downsample = 2
        self.iters = 6
        self.slow_fast_gru = True

        self.num_depth_regressor_anchor = 128 # 512
        self.used_res_channel = self.decoder_channels[1] # now, use 2/7 res
        self.depth_regressor = nn.Sequential(
            Conv2d(self.used_res_channel,
                    self.num_depth_regressor_anchor,
                    kernel_size=3,
                    padding=1),
            # nn.BatchNorm2d(self.num_depth_regressor_anchor),
            nn.ReLU(inplace=True),
            Conv2d(self.num_depth_regressor_anchor,
                    1,
                    kernel_size=1),
        )
        self.normal_predictor = nn.Sequential(
            Conv2d(self.used_res_channel,
                      128,
                      kernel_size=3,
                      padding=1),
            # nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            Conv2d(128, 128, kernel_size=1), nn.ReLU(inplace=True),
            Conv2d(128, 3, kernel_size=1),
        )

        self.context_feature_encoder = ContextFeatureEncoder(self.feature_channels, [self.hidden_dims, self.context_dims], tuning_mode=tuning_mode)
        self.context_zqr_convs = nn.ModuleList([Conv2d(self.context_dims[i], self.hidden_dims[i]*3, 3, padding=3//2) for i in range(self.n_gru_layers)])
        self.update_block = BasicMultiUpdateBlock(hidden_dims=self.hidden_dims[::-1], out_dims=4, tuning_mode=tuning_mode)

        self.relu = nn.ReLU(inplace=True)

        self.refine_depth1 = PMP(level=0, in_ch=64, out_ch=32, drop_path=0.1, fuse_prob=0., pool=False)
        self.refine_depth2 = PMP(level=1, in_ch=128, out_ch=64, drop_path=0.1, fuse_prob=0.)
        self.refine_depth3 = PMP(level=2, in_ch=128, out_ch=128, drop_path=0.1, fuse_prob=0.)
        self.refine_depth4 = PMP(level=3, in_ch=256, out_ch=128, drop_path=0.1, fuse_prob=0.)
        self.refine_depth5 = PMP(level=4, in_ch=256, out_ch=256, drop_path=0.1, fuse_prob=0., up=False)

        self.bridge5 = Basic2d(256, 256, norm_layer=nn.BatchNorm2d)
        self.bridge4 = Basic2d(256, 128, norm_layer=nn.BatchNorm2d)
        self.bridge3 = Basic2d(128, 128, norm_layer=nn.BatchNorm2d)
        self.bridge2 = Basic2d(96, 64, norm_layer=nn.BatchNorm2d)
        self.bridge1 = Basic2d(96, 32, norm_layer=nn.BatchNorm2d)
    
        self.global_feat = GlobalPoint()

        self.relu = nn.ReLU(inplace=True)

    def regress_depth(self, feature_map_d):
        depth = self.depth_regressor(feature_map_d)

        return depth

    def pred_normal(self, feature_map):
        normal_out = self.normal_predictor(feature_map)

        return norm_normalize(normal_out)
    
    def create_mesh_grid(self, height, width, batch, device="cuda", set_buffer=True):
        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=device),
                               torch.arange(0, width, dtype=torch.float32, device=device)], indexing='ij')
        meshgrid = torch.stack((x, y))
        meshgrid = meshgrid.unsqueeze(0).repeat(batch, 1, 1, 1)
        return meshgrid

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, _, H, W = img.shape

        coords0 = coords_grid(N, H, W).to(img.device)
        coords1 = coords_grid(N, H, W).to(img.device)

        return coords0, coords1

    def forward(self, img, sparse, stable, Kcam):
        stable_mask = torch.zeros_like(stable)
        stable_mask[stable > 0.0] = 1.0

        fuse_feat1, fuse_feat2, fuse_feat3, fuse_feat4, fuse_feat5 = self.full_encoder(img, sparse)

        encoder_features = [fuse_feat5, fuse_feat4, fuse_feat3, fuse_feat2, fuse_feat1]
        
        ## decode features to init-depth (and confidence)
        depth_pred = self.regress_depth(fuse_feat1) # regress bin for depth
        normal_pred = self.pred_normal(fuse_feat1) # mlp for normal

        depth_init = torch.cat((depth_pred, normal_pred), dim=1) # (N, 1+1+4, H, W)

        ## encoder features to context-feature for init-hidden-state and contex-features
        cnet_list = self.context_feature_encoder(encoder_features[::-1])
        net_list = [torch.tanh(x[0]) for x in cnet_list] # x_4, x_8, x_16 of hidden state
        inp_list = [torch.relu(x[1]) for x in cnet_list] # x_4, x_8, x_16 context features

        # Rather than running the GRU's conv layers on the context features multiple times, we do it once at the beginning 
        inp_list = [list(conv(i).split(split_size=conv.out_channels//3, dim=1)) for i,conv in zip(inp_list, self.context_zqr_convs)]

        coords0, coords1 = self.initialize_flow(net_list[0])

        if depth_init is not None:
            coords1 = coords1 + depth_init

        tmp_depth = depth_init[:,:1]
        tmp_depth = (1 - stable_mask) * tmp_depth + stable
        coords1[:, :1] = coords0[:, :1] + tmp_depth
        flow_predictions = [tmp_depth]
        normal_predictions = [norm_normalize(depth_init[:,1:].clone())]

        for itr in range(self.iters):
            # coords1 = coords1.detach()
            flow = coords1 - coords0
            if self.n_gru_layers == 3 and self.slow_fast_gru: # Update low-res GRU
                net_list = self.update_block(net_list, inp_list, iter32=True, iter16=False, iter08=False, update=False)
            if self.n_gru_layers >= 2 and self.slow_fast_gru:# Update low-res GRU and mid-res GRU
                net_list = self.update_block(net_list, inp_list, iter32=self.n_gru_layers==3, iter16=True, iter08=False, update=False)
            net_list, delta_flow = self.update_block(net_list, inp_list, None, flow, iter32=self.n_gru_layers==3, iter16=self.n_gru_layers>=2)
            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords0 + delta_flow                

            flow_tmp = coords1-coords0
            depth_tmp = flow_tmp[:, :1]
            depth_up = (1 - stable_mask) * depth_tmp + stable
            coords1[:, :1] = coords0[:, :1] + depth_up

            flow_predictions.append(depth_up)
            normal_predictions.append(norm_normalize(flow_tmp[:,1:].clone()))

        global_feat = self.global_feat(sparse[:,:1]*10.0, normal_predictions[-1], Kcam)

        output = []

        depth_pred1 = depth_up * 10.0
        # depth_pred2 = F.interpolate(depth_pred1, scale_factor=2**-1, mode='bilinear', align_corners=False)
        # depth_pred3 = F.interpolate(depth_pred2, scale_factor=2**-1, mode='bilinear', align_corners=False)
        # depth_pred4 = F.interpolate(depth_pred3, scale_factor=2**-1, mode='bilinear', align_corners=False)
        # depth_pred5 = F.interpolate(depth_pred4, scale_factor=2**-1, mode='bilinear', align_corners=False)

        feat_5 = self.bridge5((fuse_feat5*10.0).detach())
        feat_4 = self.bridge4((fuse_feat4*10.0).detach())
        feat_3 = self.bridge3((fuse_feat3*10.0).detach())
        feat_2 = self.bridge2((fuse_feat2*10.0).detach())
        feat_1 = self.bridge1((fuse_feat1*10.0).detach())

        control_point = sparse[:, :1] * 10.0

        fout_5, dout_5 = self.refine_depth5(feat_5, None, None, depth_pred1, control_point, global_feat, Kcam)
        output.append(F.interpolate(dout_5, scale_factor=2 ** 4, mode='bilinear', align_corners=True))

        fout_4, dout_4 = self.refine_depth4(feat_4, fout_5, dout_5, depth_pred1, control_point, global_feat, Kcam)
        output.append(F.interpolate(dout_4, scale_factor=2 ** 3, mode='bilinear', align_corners=True))

        fout_3, dout_3 = self.refine_depth3(feat_3, fout_4, dout_4, depth_pred1, control_point, global_feat, Kcam)
        output.append(F.interpolate(dout_3, scale_factor=2 ** 2, mode='bilinear', align_corners=True))

        fout_2, dout_2 = self.refine_depth2(feat_2, fout_3, dout_3, depth_pred1, control_point, global_feat, Kcam)
        output.append(F.interpolate(dout_2, scale_factor=2 ** 1, mode='bilinear', align_corners=True))

        fout_1, dout_1 = self.refine_depth1(feat_1, fout_2, dout_2, depth_pred1, control_point, global_feat, Kcam)
        output.append(dout_1)

        outputs=dict(
            prediction=output,
            predictions_list=flow_predictions,
            prediction_normal=normal_predictions[-1],
            predictions_normal_list=normal_predictions,
        )

        return outputs


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)

    Succ = Completion3D()
    Succ = Succ.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # lydebug.check_model(Succ, [(3, 256, 1216), (2, 256, 1216)])
    test_rgb = torch.randn(2, 3, 256, 1216).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    test_sparse = torch.randn(2, 2, 256, 1216).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # 初始化 Adam 优化器 
    optimizer = torch.optim.Adam(Succ.parameters(), lr=0.001) 
    for i in range(50): 
        # 前向传播 
        out = Succ(test_rgb, test_sparse)
        
        # 计算损失 
        loss = out["prediction"].mean() + out["confidence"].mean() + out["prediction_normal"].mean() 
        # 反向传播 
        optimizer.zero_grad() 
        # 梯度清零 
        loss.backward() 
        # 计算梯度 
        optimizer.step() 
