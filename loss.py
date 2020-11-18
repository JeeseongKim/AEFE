## loss function

import torch
from torch import nn
import math

class loss_concentration(nn.Module):
    def __init__(self):
        super(loss_concentration, self).__init__()

    def forward(self, DetectionMap, zeta):
        conc_loss = 0
        get_std = torch.zeros_like(DetectionMap)
        std_x = torch.zeros(DetectionMap.shape[0], DetectionMap.shape[1])
        std_y = torch.zeros(DetectionMap.shape[0], DetectionMap.shape[1])
        #dev = torch.zeros(DetectionMap.shape[0])
        for b in range(DetectionMap.shape[0]):
            for k in range(DetectionMap.shape[1]):
                get_std[b, k, :, :] = DetectionMap[b, k, :, :] / zeta[b, k]
                std_x[b, k] = torch.var(torch.var(get_std[b, k, :, :], 1))
                #print("std_x", std_x[b, k])
                std_y[b, k] = torch.var(torch.var(get_std[b, k, :, :], 0))
                #print("std_y", std_y[b, k])
                #dev[k] = 0.5 * (std_x[k] ** 2 + std_y[k] ** 2)
                cur_loss = 2 * math.pi * math.exp(std_x[b, k]) + math.exp(std_y[b, k])
                #cur_loss = std_x[b, k] + std_y[b, k]
                #print(cur_loss)
                conc_loss = conc_loss + cur_loss
        #print(conc_loss)
        conc_loss = torch.tensor(conc_loss)
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

