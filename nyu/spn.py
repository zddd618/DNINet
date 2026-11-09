import torch.nn as nn
import torch

from deeptool import lydebug, lyvision

from dcn_v2 import dcn_v2_conv

kernel_size = 3


def Conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def Conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

class Basic2d(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=None, kernel_size=3, padding=1):
        super().__init__()
        if norm_layer:
            conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                             stride=1, padding=padding, bias=False)
        else:
            conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                             stride=1, padding=padding, bias=True)
        self.conv = nn.Sequential(conv, )
        if norm_layer:
            self.conv.add_module('bn', norm_layer(out_channels))
        self.conv.add_module('relu', nn.ReLU(inplace=True))

    def forward(self, x):
        out = self.conv(x)
        return out

class Basic2dTrans(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                                       stride=2, padding=1, output_padding=1, bias=False)
        self.bn = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out

class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None, act=True):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = Conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.act = act

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        if self.act:
            out = self.relu(out)
        return out
    

class BasicDepthEncoder(nn.Module):

    def __init__(self, kernel_size, block=BasicBlock, bc=16, norm_layer=nn.BatchNorm2d):
        super(BasicDepthEncoder, self).__init__()
        self._norm_layer = norm_layer
        self.kernel_size = kernel_size
        self.num = kernel_size*kernel_size - 1
        self.idx_ref = self.num // 2

        self.convd1 = Basic2d(1, bc * 2, norm_layer=None, kernel_size=3, padding=1)
        self.convd2 = Basic2d(bc * 2, bc * 2, norm_layer=None, kernel_size=3, padding=1)

        self.convf1 = Basic2d(bc * 2, bc * 2, norm_layer=None, kernel_size=3, padding=1)
        self.convf2 = Basic2d(bc * 2, bc * 2, norm_layer=None, kernel_size=3, padding=1)

        self.conv = Basic2d(bc * 4, bc * 4, norm_layer=None, kernel_size=3, padding=1)
        self.ref = block(bc * 4, bc * 4, norm_layer=norm_layer, act=False)
        self.conv_weight = nn.Conv2d(bc * 4, self.kernel_size**2, kernel_size=1, stride=1, padding=0)
        self.conv_offset = nn.Conv2d(bc * 4, 2*(self.kernel_size**2 - 1), kernel_size=1, stride=1, padding=0)

    def forward(self, depth, context):
        B, _, H, W = depth.shape

        d1 = self.convd1(depth)
        d2 = self.convd2(d1)

        f1 = self.convf1(context)
        f2 = self.convf2(f1)

        input_feature = torch.cat((d2, f2), dim=1)
        input_feature = self.conv(input_feature)
        feature = self.ref(input_feature)
        weight = torch.sigmoid(self.conv_weight(feature))
        offset = self.conv_offset(feature)

        # Add zero reference offset
        offset = offset.view(B, self.num, 2, H, W)
        list_offset = list(torch.chunk(offset, self.num, dim=1))
        list_offset.insert(self.idx_ref,
                           torch.zeros((B, 1, 2, H, W)).type_as(offset))
        offset = torch.cat(list_offset, dim=1).view(B, -1, H, W)

        return weight, offset

class pa_spn(nn.Module):

    def __init__(self):
        super().__init__()

        self.dkn_residual = True

        self.w = nn.Parameter(torch.ones((1, 1, kernel_size, kernel_size)))
        self.b = nn.Parameter(torch.zeros(1))
        self.stride = 1
        self.padding = int((kernel_size - 1) / 2)
        self.dilation = 1
        self.deformable_groups = 1
        # self.im2col_step = 64

    def forward(self, depth, weight, offset):

        if self.dkn_residual:
            weight = weight - torch.mean(weight, 1).unsqueeze(1).expand_as(weight)
        else:
            weight = weight / torch.sum(weight, 1).unsqueeze(1).expand_as(weight)

        output = dcn_v2_conv.apply(
            depth, offset, weight, self.w, self.b, self.stride, self.padding,
            self.dilation, self.deformable_groups)

        if self.dkn_residual:
            output = output + depth

        return output
