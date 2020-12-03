## loss function

import torch
from torch import nn
import math
import time
import torch.nn.functional as F

class loss_concentration(nn.Module):
    def __init__(self):
        super(loss_concentration, self).__init__()

    def forward(self, softmask):
        dmap_sz_0, dmap_sz_1, _, _ = softmask.shape #(b, k, 96, 128)

        #var_x = torch.var(torch.var(softmask, 1))
        #var_y = torch.var(torch.var(softmask, 0))
        var_x = torch.mean(torch.var(softmask, 1))
        var_y = torch.mean(torch.var(softmask, 0))

        conc_loss = torch.tensor(2 * torch.tensor(math.pi) * torch.tensor(math.e) * ((var_x + var_y)**0.5))
        #print("var_x", var_x)
        #print("var_y", var_y)
        #print("333", ((var_x + var_y)**0.5))
        #print("***", conc_loss)

        return conc_loss

class loss_separation(nn.Module):
    def __init__(self):
        super(loss_separation, self).__init__()
        self.std_sep = 0.06  # 0.04 ~ 0.08
        self.inv_std_sep = 1 / (self.std_sep ** 2)

    def forward(self, keypoints):
        sep_loss = 0
        kp_sz_0, kp_sz_1, _ = keypoints.shape
        '''
        for i in range(kp_sz_1):
            for j in range(kp_sz_1):
                if(i != j):
                    inp_ = keypoints[:, i, :] - keypoints[:, j, :]
                    inp_xy_norm_sum_sqrt = math.sqrt((inp_ ** 2).sum())

                    cur_loss = torch.exp(torch.tensor(-0.001 * inp_xy_norm_sum_sqrt / self.std_sep2))
                    sep_loss = sep_loss + cur_loss
        '''

        for i in range(kp_sz_1):
            cur_loss = F.mse_loss(keypoints[:, i, :].unsqueeze(1), keypoints)
            sep_loss = sep_loss + torch.exp(- cur_loss)
            sep_loss = sep_loss + cur_loss

        sep_loss_output = sep_loss

        return sep_loss_output

class loss_reconstruction(nn.Module):
    def __init__(self):
        super(loss_reconstruction, self).__init__()
        self.std_color = 0.05
        self.std_color_2 = self.std_color ** 2

    def forward(self, reconImg, aefe_input):

        recon_loss = (((aefe_input - reconImg) ** 2) / self.std_color_2).sum() + torch.log(torch.tensor(2 * math.pi * self.std_color_2))

        return recon_loss


