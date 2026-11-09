import os

from deeptool import lyconfig, lydebug
from deeptool.lydata import KITTIDC, NYUBIG

exam_name = os.path.basename(os.getcwd())

import torch

def calculate_coefficients(height, width, focal_length, cx, cy):
    # 生成像素网格
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width))
    y = y.float()
    x = x.float()
    
    # 计算系数
    # coefficients = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2 + focal_length ** 2) / focal_length
    coefficients = focal_length / torch.sqrt((x - cx) ** 2 + (y - cy) ** 2 + focal_length ** 2)
    
    return coefficients

# 示例参数
height = 256
width = 1216
ori_height = 352
ori_width = 1216
focal_length = 700.0
cx = (width - 1.0) / 2.0
cy = (2*height - ori_height - 1.0) / 2.0

# 计算系数图
coefficients = calculate_coefficients(height, width, focal_length, cx, cy)
scale = 1.0 / torch.min(coefficients)

# 移动到CUDA
# coefficients = coefficients

def build_it():
    lydebug.check_directory('./saved/test')
    lydebug.check_directory('./saved/train')
    lydebug.check_directory('./saved/logs')
    lydebug.check_directory(lyconfig.checkpoint_path)
    lydebug.check_directory(lyconfig.history_checkpoint_path + exam_name)

    # DIODE.check_states()
    # HRWSI.check_states()
    # NYUBIG.check_states()
    KITTIDC.check_states()

if __name__ == '__main__':
    build_it()
    print(coefficients)
    print(scale)
