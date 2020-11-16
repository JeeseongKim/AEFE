import torch
from torch import nn
import matplotlib.pyplot as plt
from PIL import Image
import glob
from torch.autograd import Variable
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
import os
import cv2
import numpy as np
from model.layers import Conv, Residual, Hourglass, UnFlatten, Merge, Pool
from model.Heatmap import HeatmapLoss, GenerateHeatmap
from StackedHourglass import StackedHourglassForKP
import time
pre_width = 128
pre_height = 96

scaleFactor = 1

class DetectionConfidenceMap2keypoint(nn.Module):
    def __init__(self, shgs_out, shgs_out_dim):
        super(DetectionConfidenceMap2keypoint, self).__init__()

    def forward(self, combined_hm_preds, batch_size, num_of_kp):
        combined_hm_preds = combined_hm_preds.cuda()
        start_ = time.time()
        _, inp_channel, img_height, img_width = combined_hm_preds.shape

        get_kp_x = 0
        get_kp_y = 0

        keypoint = torch.ones(batch_size, inp_channel, 2)
        R_k = combined_hm_preds

        softmax = torch.nn.Softmax()
        map_val_all = softmax(R_k) #(b, k, 96, 128)
        map_val_all = map_val_all.cuda()
        get_zeta = torch.zeros(batch_size, num_of_kp)


        for b in range(R_k.shape[0]):
            #tmp_get_zeta = map_val_all[b, :, :, :] #(k,96,128)
            get_zeta[b, :] = map_val_all[b, :, :, :].view(map_val_all[b, :, :, :].shape[0], -1).sum(1).cuda()
            new_start = time.time()
            for k in range(inp_channel):
                for i in range(img_height):
                    for j in range(img_width):
                        get_kp_x += j * map_val_all[b, k, i, j]
                        get_kp_y += i * map_val_all[b, k, i, j]

                keypoint[b, k, 0] = (int)(torch.round((get_kp_x / get_zeta[b, k])))
                keypoint[b, k, 1] = (int)(torch.round((get_kp_y / get_zeta[b, k])))

                ##Hard Constraint on Keypoints##
                if((keypoint[b, k, 0] > pre_width) or (keypoint[b, k, 0] < 0)):
                #if ((keypoint[b, k, 0] > pre_width)):
                    keypoint[b, k, 0] = (int)(pre_width * 0.5)
                if ((keypoint[b, k, 1] > pre_height) or (keypoint[b, k, 1] < 0)):
                #if ((keypoint[b, k, 1] > pre_height)):
                    keypoint[b, k, 1] = (int)(pre_height * 0.5)

            print("kp_3 time: ", time.time() - new_start)
        print("kp_4 time: ", time.time() - start_)
                #print("my keypoint is", "[", keypoint[b, k, 0].item(), ",", keypoint[b, k, 1].item(), "]")

        return map_val_all, keypoint, get_zeta

class ReconDetectionConfidenceMap(nn.Module):
    def __init__(self):
        super(ReconDetectionConfidenceMap, self).__init__()

    def forward(self, keypoints, reconDetectionMap_std_x, reconDetectionMap_std_y, correlation, DetectionMap):
        scoreMap = torch.ones(DetectionMap.shape[0], DetectionMap.shape[1], DetectionMap.shape[2], DetectionMap.shape[3])
        reconDetectionMap = torch.ones(DetectionMap.shape[0], DetectionMap.shape[1], DetectionMap.shape[2], DetectionMap.shape[3])
        _, keypoints_num_sz, _ = keypoints.shape
        cov = torch.zeros(2, 2)
        cov[0, 0] = reconDetectionMap_std_x ** 2
        cov[0, 1] = correlation * reconDetectionMap_std_x * reconDetectionMap_std_y
        cov[1, 0] = correlation * reconDetectionMap_std_x * reconDetectionMap_std_y
        cov[1, 1] = reconDetectionMap_std_y ** 2
        gaussian_distribution_den = 1 / (2 * np.pi * reconDetectionMap_std_x * reconDetectionMap_std_y * np.sqrt(1 - (correlation * correlation)))
        gaussian_distribution_num_1 = -1 / (2 * (1 - (correlation * correlation)))
        cur_rk_all = torch.zeros(DetectionMap.shape[0], DetectionMap.shape[2], DetectionMap.shape[3])
        for b in range(DetectionMap.shape[0]):
            for k in range(keypoints_num_sz):
                for i in range(DetectionMap.shape[2]): #height (y)
                    for j in range(DetectionMap.shape[3]): #width (x)
                        gaussian_distribution_num_2 = ((i-keypoints[b, k, 1])**2 / reconDetectionMap_std_y**2) + ((j - keypoints[b, k, 0])**2 / reconDetectionMap_std_x**2) - (2*correlation*(i - keypoints[b, k, 1])*(j - keypoints[b, k, 0])/(reconDetectionMap_std_x*reconDetectionMap_std_y))
                        gaussian_distribution = gaussian_distribution_den * torch.exp(gaussian_distribution_num_1*gaussian_distribution_num_2)
                        scoreMap[b, k, i, j] = gaussian_distribution.item()
                        if((scoreMap[b, k, i, j]<0) or (scoreMap[b, k, i, j]>1)):
                            print("---ScoreMap is Out of Range---")

        #print("scoreMap", scoreMap.shape) #(b, n, 96, 128)
        for b in range(DetectionMap.shape[0]):
            cur_rk_all[b, :, :] = torch.sum(scoreMap[b, :, :, :], dim=0)  # (b, 96,128)
        #print("****", cur_rk_all)
        #print("****", cur_rk_all.shape)
        epsilon_scale = 1e-5
        for b in range(DetectionMap.shape[0]):
            for i in range(DetectionMap.shape[2]):
                for j in range(DetectionMap.shape[3]):
                    for k in range(keypoints_num_sz):
                        reconDetectionMap[b, k, i, j] = (scoreMap[b, k, i, j] /(cur_rk_all[b, i, j] + epsilon_scale)).item()
                        #print("---")
                        #print(cur_rk_all[i, j].item())
                        #print("reconDetectionMap", reconDetectionMap[k, i, j])
        #print(reconDetectionMap.shape) #(b, n, 96, 128)
        #print(reconDetectionMap)
        return reconDetectionMap

class create_softmask (nn.Module):
    def __init__(self):
        super(create_softmask, self).__init__()

    def forward(self, dev_x, dev_y, keypoints, DetectionMap, zeta):
        """
        get_std = torch.zeros_like(DetectionMap)
        std_x = torch.zeros_like(DetectionMap)
        std_y = torch.zeros_like(DetectionMap)
        dev = torch.zeros_like(DetectionMap)
        cov_softmask = torch.zeros(DetectionMap.shape[0], 2, 2)
        softmask = torch.zeros(DetectionMap.shape[0], DetectionMap.shape[1], DetectionMap.shape[2])
        correlation = 0.5

        for k in range(DetectionMap.shape[0]):
            get_std[k] = DetectionMap[k, :, :]/zeta[k]
            std_x[k] = torch.var(torch.var(get_std, 1))
            std_y[k] = torch.var(torch.var(get_std, 0))
            dev[k] = 0.5*(std_x[k]**2 + std_y[k]**2)
            cov_softmask[k, 0, 0] = std_x[k] ** 2
            cov_softmask[k, 0, 1] = correlation * std_x[k] * std_y[k]
            cov_softmask[k, 1, 0] = cov_softmask[k, 0, 1]
            cov_softmask[k, 1, 1] = std_y[k] ** 2
            gaussian_distribution_den = 1 / (2 * np.pi * std_x[k] * std_y[k] * np.sqrt(1 - (correlation * correlation)))
            gaussian_distribution_num_1 = -1 / (2 * (1 - (correlation * correlation)))

            for i in range(DetectionMap.shape[1]):
                for j in range(DetectionMap.shape[2]):
                    gaussian_distribution_num_2 = ((i-keypoints[k, 1])**2 / std_y[k] ** 2) + ((j - keypoints[k, 0])**2 / std_x[k] **2) - (2*correlation*(i - keypoints[k, 1])*(j - keypoints[k, 0])/(std_x[k] *std_y[k]))
                    gaussian_distribution_softmask = gaussian_distribution_den * torch.exp(gaussian_distribution_num_1*gaussian_distribution_num_2)
                    softmask[k, i, j] = gaussian_distribution_softmask.item()

                    if ((softmask[k, i, j] < 0) or (softmask[k, i, j] > 1)):
                        print("---OUT OF RANGE---")

        softmask = softmask / (DetectionMap.shape[1] * DetectionMap.shape[2])
        """

        softmask = torch.zeros_like(DetectionMap)
        for b in range(DetectionMap.shape[0]):
            for k in range(DetectionMap.shape[1]):
                softmask[b, k, :, :] = DetectionMap[b, k, :, :] / zeta[b, k]

        #print("softmask", softmask.shape)
        return softmask