## loss function

import torch
from torch import nn
import math

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

    def forward(self, keypoints):
        std_sep = 0.05 #0.04 ~ 0.08
        sep_loss = 0
        tmp = torch.zeros(1)

        for b in range(keypoints.shape[0]):
            for i in range(keypoints.shape[1]):
                for j in range(keypoints.shape[1]):
                    if(i != j):
                        inp_x = keypoints[b, i, 0] - keypoints[b, j, 0]
                        inp_y = keypoints[b, i, 1] - keypoints[b, j, 1]
                        inp_xy_norm = math.sqrt(inp_x * inp_x + inp_y * inp_y)
                        #print("inp_x  ", inp_x)
                        #print("inp_y  ", inp_y)
                        tmp[0] = -0.5 * inp_xy_norm / std_sep * std_sep
                        cur_loss = torch.exp(tmp)
                        sep_loss = sep_loss + cur_loss
        #print(sep_loss)
        return sep_loss

class loss_reconstruction(nn.Module):
    def __init__(self):
        super(loss_reconstruction, self).__init__()

    def forward(self, reconImg, aefe_input):
        std_color = 0.05
        std_color_2 = std_color ** 2

        recon_loss = (((aefe_input - reconImg) ** 2) / std_color_2).sum() + torch.log(torch.tensor(2 * math.pi * std_color_2))

        return recon_loss


