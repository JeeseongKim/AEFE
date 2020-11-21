## loss function

import torch
from torch import nn
import math
import time

class loss_concentration(nn.Module):
    def __init__(self):
        super(loss_concentration, self).__init__()

    def forward(self, softmask):
        dmap_sz_0, dmap_sz_1, _, _ = softmask.shape #(b, k, 96, 128)

        #var = torch.var(softmask)
        #conc_loss = torch.tensor(2 * math.pi * torch.exp(var))

        var_x = torch.var(torch.var(softmask, 1))
        var_y = torch.var(torch.var(softmask, 0))
        conc_loss = torch.tensor(2 * math.pi * (torch.exp(var_x) + torch.exp(var_y)))


        """
        for b in range(dmap_sz_0):
            for k in range(dmap_sz_1):
                cur_std_xy = DetectionMap[b, k, :, :] / zeta[b, k] #(96,128)
                std_x = torch.var(torch.var(cur_std_xy, 1))
                std_y = torch.var(torch.var(cur_std_xy, 0))
                #print("std_x: ", std_x.item(), "std_y: ", std_y.item())
                #cur_loss = 2 * math.pi * (torch.exp(std_x) + torch.exp(std_y))
                cur_loss = 2 * math.pi * (torch.log(std_x) + torch.log(std_y))
                conc_loss = conc_loss + cur_loss
        """
        return conc_loss

class loss_separation(nn.Module):
    def __init__(self):
        super(loss_separation, self).__init__()
        self.std_sep = 0.05  # 0.04 ~ 0.08
        self.std_sep2 = self.std_sep ** 2

    def forward(self, keypoints):
        #start = time.time()
        sep_loss = 0
        kp_sz_0, kp_sz_1, _ = keypoints.shape
        #for b in range(kp_sz_0):
        for i in range(kp_sz_1):
            for j in range(kp_sz_1):
                if(i != j):
                    #print("keypoints: ", "1:", keypoints[:, i, :] , "  2: ",  keypoints[:, j, :])
                    inp_ = keypoints[:, i, :] - keypoints[:, j, :]
                    #print("111", inp_)
                    inp_xy_norm_sum_sqrt = math.sqrt((inp_ ** 2).sum())
                    #print("111", (inp_ ** 2))
                    #print("111", (inp_ ** 2).sum())
                    #print("222", inp_xy_norm_sum_sqrt)
                    cur_loss = torch.exp(torch.tensor(-0.001 * inp_xy_norm_sum_sqrt / self.std_sep2))
                    #print("333", (torch.tensor(-0.001 * inp_xy_norm_sum_sqrt / self.std_sep2)).item())
                    #print("444", cur_loss)
                    sep_loss = sep_loss + cur_loss
                    #print("555", sep_loss)
        #print("sep_loss_time", time.time() - start)
        return sep_loss

class loss_reconstruction(nn.Module):
    def __init__(self):
        super(loss_reconstruction, self).__init__()
        self.std_color = 0.05
        self.std_color_2 = self.std_color ** 2

    def forward(self, reconImg, aefe_input):

        recon_loss = (((aefe_input - reconImg) ** 2) / self.std_color_2).sum() + torch.log(torch.tensor(2 * math.pi * self.std_color_2))

        return recon_loss


