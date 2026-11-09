from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
import torch
import random
import numpy as np
import importlib
import build_essential
import cv2

from deeptool import lydebug, lyvision
import deeptool.lydata.utils as lyutils
# from deeptool.lydata.HOLE import HOLE

from remove_out import outlier_removal
from fill import fill_in_fast

input_size = 320

def Crop(img, h_init, w_init, crop_h, crop_w):
    return TF.crop(img, h_init, w_init, crop_h, crop_w)

def Hflip(img, flip):
    if flip > 0.5:
        return TF.hflip(img)
    else:
        return img

def ColorJitter(img):
    img_np = np.array(img)
    pca = compute_pca(img_np)
    img_np = add_pca_jitter(img_np, pca)
    img = torch.from_numpy(img_np).type(torch.float32)

    return img

def compute_pca(image):
    reshaped_data = image.reshape(-1, 3)
    reshaped_data = (reshaped_data).astype(np.float32)
    covariance = np.cov(reshaped_data.T)
    e_vals, e_vecs = np.linalg.eigh(covariance)
    pca = np.sqrt(e_vals) * e_vecs
    return pca

def add_pca_jitter(img_data, pca):
    new_img_data = np.copy(img_data).astype(np.float32)
    magnitude = np.random.randn(3) * 0.15
    noise = (pca * magnitude).sum(axis=1)

    new_img_data = new_img_data + noise[:, np.newaxis, np.newaxis]
    np.clip(new_img_data, 0.0, 1.0, out=new_img_data)

    return new_img_data

def Rotation(img, degree):
    return TF.rotate(img, angle=degree)

def Resize(img, size, mode=None):
    if mode:
        return TF.resize(img, size, mode)
    else:
        return TF.resize(img, size)

class G2_data(Dataset):

    def __init__(self, name, dataset_name_list) -> None:
        super(G2_data, self).__init__()

        self.kind = name
        self.dataset = []

        for i in dataset_name_list:
            dataset_name = importlib.import_module("deeptool.lydata." + i)
            dataset_real = dataset_name.DATASET(self.kind)
            self.dataset.append(dataset_real)
        
        self.length = [0] + [s.measure() for s in self.dataset]
        for i in range(1, len(self.length)):
            self.length[i] += self.length[i - 1]    
        # self.hole = HOLE("train")

        self.crop = transforms.Compose([
            transforms.RandomCrop((input_size, input_size)),
            transforms.RandomHorizontalFlip(0.5)
        ])
        # self.hole_trans = transforms.Compose([
        #     transforms.RandomCrop((input_size, input_size)),
        #     transforms.RandomAffine(degrees=180, translate=(0.5, 0.5), scale=(0.5, 4.0), shear=60, fill=1),
        #     transforms.RandomHorizontalFlip(0.5),
        #     transforms.RandomVerticalFlip(0.5)
        # ])
        self.rgb_trans = transforms.Compose([
            transforms.ColorJitter(0.2, 0.2, 0.2)
        ])

        self.height, self.width = 256, 1216
        self.ori_height, self.ori_width = 352, 1216
        
        self.fx = torch.tensor(700.0)
        self.fy = torch.tensor(700.0)
        self.cx = torch.tensor(self.width / 2).float()
        self.cy = torch.tensor((2*self.height - self.ori_height) / 2).float()

        u = torch.arange(0, self.width).repeat(self.height, 1).float()
        v = torch.arange(0, self.height).repeat(self.width, 1).float()
        u = u.view(1, self.height, self.width)
        v = v.transpose(1, 0).view(1, self.height, self.width)

        self.u = (u - self.cx) / self.fx
        self.v = (v - self.cy) / self.fy

    def build_sample_matrix(self, rate, input_size=input_size):
        if rate == 0.0:
            matrix = torch.ones(input_size)
        elif rate == 1.0:
            matrix = torch.zeros(input_size)
        else:
            matrix = torch.zeros(input_size).uniform_(0.0, 1.0)
            matrix[matrix <= rate] = 0
            matrix[matrix > rate] = 1

        return matrix

    def distort(self, gt, p_blur=-1.0, p_noise=0.4):
        distorted_gt = gt.clone()
        input_size = gt.shape[1:]
        if random.random() < p_noise:
            gaussian = torch.ones(input_size).normal_(0, random.uniform(0.01, 0.1))
            matrix = self.build_sample_matrix(random.random()*0.6+0.2, input_size)
            distorted_gt = distorted_gt + matrix * gaussian

        if random.random() < p_blur:
            factor = 2 ** random.randint(1, 5)
            trans = transforms.Compose([
                transforms.Resize((int(input_size[0] * 1.0 / factor), int(input_size[1] * 1.0 / factor)),
                                  interpolation=transforms.InterpolationMode.NEAREST),
                transforms.Resize(input_size, interpolation=transforms.InterpolationMode.NEAREST)
            ])
            distorted_gt = trans(distorted_gt)
        distorted_gt = torch.clamp(distorted_gt, -1.0, 1.0)

        return distorted_gt

    def data_pipeline(self, img, gt_ori, depth, normal, depth_stable, dep_ip_torch):

        normal = normal.permute(1, 2, 0).cpu().numpy() * 2.0 - 1.0
        resized_normal = cv2.resize(normal, (1242, 375), interpolation=cv2.INTER_LINEAR)
        normal = torch.from_numpy(resized_normal).permute(2, 0, 1)
        normal = Crop(normal, 55, 13, 320, 1216)

        flip = torch.FloatTensor(1).uniform_(0, 1).item()
        degree = torch.FloatTensor(1).uniform_(-5.0, 5.0).item()
 
        img = ColorJitter(img)

        depth = Hflip(depth, flip)
        gt_ori = Hflip(gt_ori, flip)
        img = Hflip(img, flip)
        normal = Hflip(normal, flip)
        depth_stable = Hflip(depth_stable, flip)
        dep_ip_torch = Hflip(dep_ip_torch, flip)

        depth = Rotation(depth, degree)
        gt_ori = Rotation(gt_ori, degree)
        img = Rotation(img, degree)
        normal = Rotation(normal, degree)
        depth_stable = Rotation(depth_stable, degree)
        dep_ip_torch = Rotation(dep_ip_torch, degree)

        width_ = depth.shape[2]
        height_ = depth.shape[1]
        rwidth, rheight = 1216, 256
        h_init = height_ - rheight
        w_init = int(torch.FloatTensor(1).uniform_(0, (width_-rwidth)//2).item())
        depth = Crop(depth, h_init, w_init, rheight, rwidth)
        gt_ori = Crop(gt_ori, h_init, w_init, rheight, rwidth)
        img = Crop(img, h_init, w_init, rheight, rwidth)
        normal = Crop(normal, h_init, w_init, rheight, rwidth)
        depth_stable = Crop(depth_stable, h_init, w_init, rheight, rwidth)
        dep_ip_torch = Crop(dep_ip_torch, h_init, w_init, rheight, rwidth)

        img = TF.normalize(img, (90.995/255.0, 96.2278/255.0, 94.3213/255.0), (79.2382/255.0, 80.5267/255.0, 82.1483/255.0), inplace=True)

        rgb = img
        gt = gt_ori
        
        input_hole = torch.ones_like(gt)
        output_hole = torch.zeros_like(gt)
        output_hole[gt > 0.] = 1

        input_hole[depth == 0] = 0
        depth *= input_hole

        # depth = self.distort(depth, input_hole)
        # normal = self.distort(normal)

        # depth = depth / build_essential.coefficients
        # gt = gt / build_essential.coefficients

        depth = torch.cat([depth, input_hole], 0)
        # depth_x = depth * self.u
        # depth_y = depth * self.v

        # depth = torch.cat([depth, depth_x, depth_y], 0)

        # gt_x = gt * self.u
        # gt_y = gt * self.v
        
        # gt = torch.cat([gt, gt_x, gt_y], 0)

        return [rgb, depth, depth_stable, dep_ip_torch], (input_hole, output_hole), gt, normal

    def __getitem__(self, index):
        dataset_select = 0
        for i in range(len(self.length)):
            if index < self.length[i]:
                dataset_select = i - 1
                break
        
        if "train" in self.kind:
            data_dict = self.dataset[dataset_select].select(index - self.length[dataset_select], ["img", "gt", "raw", "normal"])

            img = data_dict["img"]
            gt_ori = data_dict["gt"]
            depth = data_dict["raw"]
            normal = data_dict["normal"]
            
            depth_np = depth.clone().numpy() * 65535.0 / 256.0
            depth_np = depth_np.squeeze(0)

            depth_np, _ = outlier_removal(depth_np)

            depth_np = depth_np / 65535.0 * 256.0

            depth_stable = torch.from_numpy(depth_np).float().unsqueeze(0)
            depth_stable[depth_stable <= 0.] = 0.

            dep_np = depth.clone().numpy() * 65535.0 / 256.0
            dep_np = dep_np.squeeze(0)
            dep_np_ip = np.copy(dep_np)

            dep_ip = fill_in_fast(dep_np_ip, max_depth=100.0,
                                extrapolate=True, blur_type='gaussian')

            dep_ip_torch = torch.from_numpy(dep_ip)
            dep_ip_torch = dep_ip_torch.to(dtype=torch.float32) / 65535.0 * 256.0

            depth = depth * 2.0
            depth_stable = depth_stable * 2.0
            dep_ip_torch = dep_ip_torch * 2.0
            gt_ori = gt_ori * 2.0 * 128.0

            input, hole, gt, normal = self.data_pipeline(img, gt_ori, depth, normal, depth_stable, dep_ip_torch)

            return input, hole, gt, normal
        elif "val" in self.kind:
            data_dict = self.dataset[dataset_select].select(index - dataset_select, ["img", "gt", "gt_ori"])

            img = data_dict["img"]
            gt_ori = data_dict["gt_ori"]
            gt = data_dict["gt"]

            width_ = img.shape[2]
            height_ = img.shape[1]
            rwidth, rheight = 1216, 256
            h_init = height_ - rheight
            w_init = 0
            gt = Crop(gt, h_init, w_init, rheight, rwidth)
            gt_ori = Crop(gt_ori, h_init, w_init, rheight, rwidth)
            img = Crop(img, h_init, w_init, rheight, rwidth)

            img = TF.normalize(img, (90.995/255.0, 96.2278/255.0, 94.3213/255.0), (79.2382/255.0, 80.5267/255.0, 82.1483/255.0), inplace=True)

            gt_np = gt.clone().numpy() * 65535.0 / 256.0
            gt_np = gt_np.squeeze(0)

            gt_np, _ = outlier_removal(gt_np)

            gt_np = gt_np / 65535.0 * 256.0

            gt_stable = torch.from_numpy(gt_np).float().unsqueeze(0)
            gt_stable[gt_stable <= 0.] = 0.

            dep_np = gt.clone().numpy() * 65535.0 / 256.0
            dep_np = dep_np.squeeze(0)
            dep_np_ip = np.copy(dep_np)

            dep_ip = fill_in_fast(dep_np_ip, max_depth=100.0,
                                extrapolate=True, blur_type='gaussian')

            dep_ip_torch = torch.from_numpy(dep_ip)
            dep_ip_torch = dep_ip_torch.to(dtype=torch.float32) / 65535.0 * 256.0

            gt = gt * 2.0
            gt_stable = gt_stable * 2.0
            dep_ip_torch = dep_ip_torch * 2.0

            input_hole = torch.zeros_like(gt)
            input_hole[gt > 0.] = 1.

            # gt = gt / build_essential.coefficients

            # gt_x = self.u * gt
            # gt_y = self.v * gt

            gt = torch.cat([gt, input_hole], 0)

            input = [img, gt, gt_stable, dep_ip_torch]

            return input, gt_ori
        else:
            data_dict = self.dataset[dataset_select].select(index - dataset_select, ["img", "gt"])

            img = data_dict["img"]
            gt = data_dict["gt"]
            jpg_name = data_dict["path"]

            gt = gt[:, 96:, :]
            img = img[:, 96:, :]

            img = TF.normalize(img, (90.995/255.0, 96.2278/255.0, 94.3213/255.0), (79.2382/255.0, 80.5267/255.0, 82.1483/255.0), inplace=True)

            gt_np = gt.clone().numpy() * 65535.0 / 256.0
            gt_np = gt_np.squeeze(0)

            gt_np, _ = outlier_removal(gt_np)

            gt_np = gt_np / 65535.0 * 256.0

            gt_stable = torch.from_numpy(gt_np).float().unsqueeze(0)
            gt_stable[gt_stable <= 0.] = 0.

            gt = gt * 2.0
            gt_stable = gt_stable * 2.0

            input_hole = torch.zeros_like(gt)
            input_hole[gt > 0.] = 1.

            # gt_x = self.u * gt
            # gt_y = self.v * gt

            gt = torch.cat([gt, input_hole], 0)

            input = [img, gt, gt_stable]

            return input, jpg_name

    def __len__(self):
        return self.length[-1]


if __name__ == "__main__":
    G2_dataset = G2_data("train-full", ["KITTIDC"])
    print(torch.max(G2_dataset[0][0][0]))
    print(torch.min(G2_dataset[0][0][0]))
    print(torch.max(G2_dataset[1][0][1][:1, :, :]))
    print(torch.min(G2_dataset[1][0][1][:1, :, :]))
    print(torch.max(G2_dataset[1][2]))
    print(torch.min(G2_dataset[1][2]))
