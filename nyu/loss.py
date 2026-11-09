import torch.nn as nn
import torch.nn.functional as func
import numpy as np
import math
import random
import torch

from deeptool import lydebug
from depth_to_normal import get_surface_normalv2 as get_surface_normal

alpha = 0.000001

class G2_loss(nn.Module):
    def __init__(self) -> None:
        super(G2_loss, self).__init__()

        self.kenel_x = torch.FloatTensor([[[[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]]]])
        self.weight_x = nn.Parameter(data=self.kenel_x, requires_grad=False)
        self.kenel_y = torch.FloatTensor([[[[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]]]])
        self.weight_y = nn.Parameter(data=self.kenel_y, requires_grad=False)
        # self.loss = nn.SmoothL1Loss(reduction="sum", beta=0.1)
        self.loss = nn.L1Loss(reduction="sum")
        self.lossl2 = nn.MSELoss(reduction="sum")
        self.loss_normal = nn.MSELoss(reduction="mean")

        # self.height, self.width = 256, 1216
        # self.ori_height, self.ori_width = 352, 1216
        
        # self.fx = torch.tensor(721.0)
        # self.fy = torch.tensor(721.0)
        # self.cx = torch.tensor(self.width / 2).float()
        # self.cy = torch.tensor((2*self.height - self.ori_height) / 2).float()

        # u = torch.arange(0, self.width).repeat(self.height, 1).float()
        # v = torch.arange(0, self.height).repeat(self.width, 1).float()
        # u = u.view(1, self.height, self.width)
        # v = v.transpose(1, 0).view(1, self.height, self.width)

        # self.u = (u - self.cx) / self.fx
        # self.v = (v - self.cy) / self.fy

        # self.u = self.u.cuda()
        # self.v = self.v.cuda()

    def gradient_loss(self, sparsein, sparsegoal, hole, lowerrank):
        # Gradient loss
        result = torch.FloatTensor([0.])
        result = result.cuda()
        dif = (sparsein - sparsegoal) * hole
        for i in range(lowerrank):
            factor = 1.0/(2 ** i)
            element = func.interpolate(dif, scale_factor=factor, recompute_scale_factor=True)
            dif_x = torch.sum(torch.abs(func.conv2d(element, self.weight_x, None, 1, 0)))
            dif_y = torch.sum(torch.abs(func.conv2d(element, self.weight_y, None, 1, 0)))
            result = result + dif_x + dif_y

        result = result / (torch.sum(hole) + alpha)
        return result

    def relative_loss(self, input, gt, hole):
        num = torch.sum(hole, dim=(1, 2, 3))
        num[num == 0.] = alpha

        input_mean = (torch.sum(input * hole, dim=(1, 2, 3)) / num).view(input.shape[0], 1, 1, 1)
        input_std = (torch.sum(torch.abs((input - input_mean) * hole), dim=(1, 2, 3)) / num).view(input.shape[0], 1, 1, 1) + alpha
    
        sta_input = (input - input_mean) / input_std

        relative = self.loss(sta_input * hole, gt * hole).cuda()
        relative = relative / (torch.sum(hole) + alpha)

        return relative
    
    def forward(self, pred, gt, hole):

        pred_1, pred_2, pred_3, pred_4, pred_5 = pred["prediction"]
        gt = gt * 10.0

        pred_1 = pred_1[:, :, 14:242, 8:312]
        pred_2 = pred_2[:, :, 14:242, 8:312]
        pred_3 = pred_3[:, :, 14:242, 8:312]
        pred_4 = pred_4[:, :, 14:242, 8:312]
        pred_5 = pred_5[:, :, 14:242, 8:312]
        gt = gt[:, :, 14:242, 8:312]
        hole = [hole[0][:, :, 14:242, 8:312], hole[1][:, :, 14:242, 8:312]]

        loss_1 = self.loss(pred_1 * hole[1], gt * hole[1])
        # loss_1 = loss_1 + self.lossl2(pred_1 * hole[1], gt * hole[1])
        loss_1 = loss_1 / (torch.sum(hole[1]) + alpha)

        loss_2 = self.loss(pred_2 * hole[1], gt * hole[1])
        # loss_2 = loss_2 + self.lossl2(pred_2 * hole[1], gt * hole[1])
        loss_2 = loss_2 / (torch.sum(hole[1]) + alpha)

        loss_3 = self.loss(pred_3 * hole[1], gt * hole[1])
        # loss_3 = loss_3 + self.lossl2(pred_3 * hole[1], gt * hole[1])
        loss_3 = loss_3 / (torch.sum(hole[1]) + alpha)

        loss_4 = self.loss(pred_4 * hole[1], gt * hole[1])
        # loss_4 = loss_4 + self.lossl2(pred_4 * hole[1], gt * hole[1])
        loss_4 = loss_4 / (torch.sum(hole[1]) + alpha)

        loss_5 = self.loss(pred_5 * hole[1], gt * hole[1])
        # loss_5 = loss_5 + self.lossl2(pred_5 * hole[1], gt * hole[1])
        loss_5 = loss_5 / (torch.sum(hole[1]) + alpha)

        loss = math.pow(4,-4)*loss_1 + math.pow(4,-3)*loss_2 + math.pow(4,-2)*loss_3 + math.pow(4,-1)*loss_4 + loss_5
        # loss = loss_5

        return loss, loss_1, loss_2, loss_3, loss_4, loss_5
    
    # def all_loss(self, pred, gt, sta_gt, hole):

    #     loss_6 = self.loss(pred * hole[1], gt * hole[1])
    #     loss_6 = loss_6 / (torch.sum(hole[1]) + alpha)

    #     loss_6_l2 = self.lossl2(pred * hole[1], gt * hole[1])
    #     loss_6_l2 = loss_6_l2 / (torch.sum(hole[1]) + alpha)
        
    #     result = self.gradient_loss(pred, gt, hole[1], 4)

    #     relative = self.relative_loss(pred, sta_gt, hole[1])

    #     loss = loss_6 + 1.5*loss_6_l2

    #     return loss

    # def random_crop_loss(self, pred, gt, hole, num_crops=20):
    #     b, c, h, w = pred.size()
    #     loss_list = []
    #     final_loss = 0

    #     for _ in range(num_crops):
    #         crop_h = random.randint(int(h * 0.25), int(h * 0.5))
    #         crop_w = random.randint(int(w * 0.25), int(w * 0.5))
            
    #         top = random.randint(0, h - crop_h)
    #         left = random.randint(0, w - crop_w)

    #         cropped_pred = pred[:, :, top:top + crop_h, left:left + crop_w]
    #         cropped_gt = gt[:, :, top:top + crop_h, left:left + crop_w]
    #         cropped_hole = hole[1][:, :, top:top + crop_h, left:left + crop_w]

    #         num = torch.sum(cropped_hole, dim=(1, 2, 3))
    #         num[num == 0.] = alpha

    #         gt_mean = (torch.sum(cropped_gt * cropped_hole, dim=(1, 2, 3)) / num).view(b, 1, 1, 1)
    #         gt_std = (torch.sum(torch.abs((cropped_gt - gt_mean) * cropped_hole), dim=(1, 2, 3)) / num).view(b, 1, 1, 1) + alpha

    #         sta_gt = (cropped_gt - gt_mean) / gt_std

    #         pred_mean = (torch.sum(cropped_pred * cropped_hole, dim=(1, 2, 3)) / num).view(b, 1, 1, 1)
    #         pred_std = (torch.sum(torch.abs((cropped_pred - pred_mean) * cropped_hole), dim=(1, 2, 3)) / num).view(b, 1, 1, 1) + alpha

    #         sta_pred = (cropped_pred - pred_mean) / pred_std

    #         loss_item = self.loss(sta_pred * cropped_hole, sta_gt * cropped_hole) / (torch.sum(cropped_hole) + alpha)

    #         loss_list.append(loss_item)
    #         final_loss += loss_item

    #     final_loss /= num_crops

    #     return final_loss, loss_list
    
    # def consistency_loss(self, pred, normal_pred):
    #     pred = pred * 256.0 / 160.0
    #     pred_x = pred * self.u
    #     pred_y = pred * self.v

    #     full_in = torch.cat((pred_x, pred_y, pred), dim=1)
    #     full_in = full_in.permute(0, 2, 3, 1).contiguous()

    #     normal, normal_mask = get_surface_normal(full_in)
    #     normal_loss = self.lossl2(normal*normal_mask, normal_pred*normal_mask) / (torch.sum(normal_mask) + alpha)
    #     return normal_loss

    # def forward(self, pred, gt, hole, normal):

    #     pred_6 = pred["prediction"] * 80.0

    #     normal_pred = pred["prediction_normal"][:, :3, :, :]

    #     num = torch.sum(hole[1], dim=(1, 2, 3))
    #     num[num == 0.] = alpha

    #     gt_mean = (torch.sum(gt * hole[1], dim=(1, 2, 3)) / num).view(gt.shape[0], 1, 1, 1)
    #     gt_std = (torch.sum(torch.abs((gt - gt_mean) * hole[1]), dim=(1, 2, 3)) / num).view(gt.shape[0], 1, 1, 1) + alpha

    #     sta_gt = (gt - gt_mean) / gt_std

    #     loss_6 = self.all_loss(pred_6, gt, sta_gt, hole)

    #     loss_normal = self.loss_normal(normal_pred, normal)

    #     loss_random, _ = self.random_crop_loss(pred_6, gt, hole)

    #     #loss_consistency = self.consistency_loss(pred_6, normal_pred)

    #     loss = 3.0*loss_6 + 2.0*loss_random + loss_normal

    #     return loss, loss_6, loss_random, loss_normal, loss_6
        

if __name__ == "__main__":
    loss = G2_loss()
    input = torch.randn(4, 1, 320, 320)
    output = torch.randn(4, 1, 320, 320)
    hole = [torch.randn(4, 1, 320, 320), torch.randn(4, 1, 320, 320)]
    fgoal = torch.randn(4, 1, 1, 1)
    sgoal = torch.randn(4, 1, 1, 1)
    f = torch.randn(4, 1, 320, 320)
    s = torch.randn(4, 1, 320, 320)
    crop_seg = torch.randn(4, 1, 320, 320)
    tmp_s_goal = torch.randn(4, 1, 1, 1)
    result = loss(input, output, hole, fgoal, sgoal, f, s, crop_seg, tmp_s_goal)
    print(result)
    print("Test is OK!")
