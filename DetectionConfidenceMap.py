import torch
from torch import nn
import numpy as np
import time

pre_width = 128
pre_height = 96

class DetectionConfidenceMap2keypoint(nn.Module):
    def __init__(self, shgs_out, shgs_out_dim):
        super(DetectionConfidenceMap2keypoint, self).__init__()

    def forward(self, combined_hm_preds, cur_batch, num_of_kp):
        start_ = time.time()
        _, inp_channel, img_height, img_width = combined_hm_preds.shape

        keypoint = torch.ones(cur_batch, inp_channel, 2)
        R_k = combined_hm_preds

        softmax = torch.nn.Softmax()
        map_val_all = softmax(R_k) #(b, k, 96, 128)
        #get_zeta = torch.zeros(batch_size, num_of_kp) #(b,k)

        get_zeta = map_val_all.view(map_val_all.shape[0], map_val_all.shape[1], -1).sum(2) #(b, k)
        #my_i = torch.tensor([i for i in range(img_height)])
        #my_j = torch.tensor([j for j in range(img_width)])
        """
        R_k_shape_0 = R_k.shape[0]
        for b in range(R_k_shape_0):
            for k in range(inp_channel):
                get_kp_x = 0
                get_kp_y = 0
                for i in range(img_height):
                    for j in range(img_width):
                        get_kp_x += j * map_val_all[b, k, i, j]
                        get_kp_y += i * map_val_all[b, k, i, j]
                keypoint[b, k, 0] = int(torch.round((get_kp_x / get_zeta[b, k])))
                keypoint[b, k, 1] = int(torch.round((get_kp_y / get_zeta[b, k])))

                if ((keypoint[b, k, 0] > pre_width)): #or (keypoint[b, k, 0] < 0)):
                    keypoint[b, k, 0] = int(pre_width * 0.5)
                if ((keypoint[b, k, 1] > pre_height)): #or (keypoint[b, k, 1] < 0)):
                    keypoint[b, k, 1] = int(pre_height * 0.5)
        """
        R_k_shape_0 = R_k.shape[0]

        get_kp_x = 0
        get_kp_y = 0

        for i in range(img_height):
            for j in range(img_width):
                get_kp_x += j * map_val_all[:, :, i, j] #(b,k)
                get_kp_y += i * map_val_all[:, :, i, j] #(b,k)
        #print("111", get_kp_x.shape)
        #print("222", get_kp_y.shape)

        start_get = time.time()
        R_k_shape_0 = R_k.shape[0]
        for b in range(R_k_shape_0):
            for k in range(inp_channel):
                keypoint[b, k, 0] = int(torch.round((get_kp_x[b, k] / get_zeta[b, k])))
                keypoint[b, k, 1] = int(torch.round((get_kp_y[b, k] / get_zeta[b, k])))
                #print("kp:", keypoint[b, k, 0], "  ", keypoint[b, k, 1])
                if ((keypoint[b, k, 0] > pre_width)):  # or (keypoint[b, k, 0] < 0)):
                    keypoint[b, k, 0] = int(pre_width * 0.5)
                if ((keypoint[b, k, 1] > pre_height)):  # or (keypoint[b, k, 1] < 0)):
                    keypoint[b, k, 1] = int(pre_height * 0.5)


        #print("kp_gen time: ", time.time() - start_get)
        return map_val_all, keypoint, get_zeta

class ReconDetectionConfidenceMap(nn.Module):
    def __init__(self):
        super(ReconDetectionConfidenceMap, self).__init__()

    def forward(self, batch_size, keypoints, reconDetectionMap_std_x, reconDetectionMap_std_y, correlation, DetectionMap):
        start = time.time()
        scoreMap = torch.ones(DetectionMap.shape[0], DetectionMap.shape[1], DetectionMap.shape[2], DetectionMap.shape[3])
        #print("-----scoreMap.shape-----: ", scoreMap.shape)
        reconDetectionMap = torch.ones(DetectionMap.shape[0], DetectionMap.shape[1], DetectionMap.shape[2], DetectionMap.shape[3])
        _, keypoints_num_sz, _ = keypoints.shape

        epsilon_scale = 1e-5

        cov = torch.zeros(2, 2)
        cov[0, 0] = reconDetectionMap_std_x ** 2
        cov[0, 1] = correlation * reconDetectionMap_std_x * reconDetectionMap_std_y
        cov[1, 0] = cov[0, 1]
        cov[1, 1] = reconDetectionMap_std_y ** 2

        gaussian_distribution_den = 1 / (2 * np.pi * reconDetectionMap_std_x * reconDetectionMap_std_y * np.sqrt(1 - (correlation ** 2)))
        gaussian_distribution_num_1 = -1 / (2 * (1 - (correlation ** 2)))
        #print("ReconDetectionMap Setup time: ", time.time() - start)

        start2 = time.time()
        """
        DetectionMap_sz_0, DetectionMap_sz_1, DetectionMap_sz_2, DetectionMap_sz_3 = DetectionMap.shape
        for b in range(DetectionMap_sz_0):
            for k in range(keypoints_num_sz):
                my_y = keypoints[b, k, 1]
                my_x = keypoints[b, k, 0]
                for i in range(DetectionMap_sz_2): #height (y)
                    for j in range(DetectionMap_sz_3): #width (x)
                        gaussian_distribution_num_2 = ((i-my_y)**2 / reconDetectionMap_std_y**2) + ((j - my_x)**2 / reconDetectionMap_std_x**2) - (2*correlation*(i - my_y)*(j - my_x)/(reconDetectionMap_std_x*reconDetectionMap_std_y))
                        scoreMap[b, k, i, j] = gaussian_distribution_den * torch.exp(gaussian_distribution_num_1*gaussian_distribution_num_2)
        """

        DetectionMap_sz_0, DetectionMap_sz_1, DetectionMap_sz_2, DetectionMap_sz_3 = DetectionMap.shape
        for i in range(DetectionMap_sz_2): #height (y)
            for j in range(DetectionMap_sz_3): #width (x)
                gaussian_distribution_num_2 = ((i - keypoints[:, :, 1])**2 / reconDetectionMap_std_y**2) + ((j - keypoints[:, :, 0])**2 / reconDetectionMap_std_x**2) - (2*correlation*(i - keypoints[:, :, 1])*(j - keypoints[:, :, 0])/(reconDetectionMap_std_x*reconDetectionMap_std_y))
                #print("****", (gaussian_distribution_den * torch.exp(gaussian_distribution_num_1 * gaussian_distribution_num_2)).shape)
                scoreMap[:, :, i, j] = gaussian_distribution_den * torch.exp(gaussian_distribution_num_1 * gaussian_distribution_num_2)

        #print("Gaussian distribution generation time: ", time.time() - start2)

        cur_rk_all = scoreMap.sum(1) #(b, 96, 128)

        #cur_rk_all = torch.zeros(DetectionMap_sz_0, DetectionMap_sz_2, DetectionMap_sz_3)
        #for b in range(DetectionMap_sz_0):
        #    cur_rk_all[b, :, :] = torch.sum(scoreMap[b, :, :, :], dim=0)  # (b, 96,128)
        start3 = time.time()
        for k in range(keypoints_num_sz):
            reconDetectionMap[:, k, :, :] = scoreMap[:, k, :, :]/ (cur_rk_all+epsilon_scale)
        #print("get reconstructed detection map: ", time.time() - start3)

        """
        for b in range(DetectionMap_sz_0):
            for i in range(DetectionMap_sz_2):
                for j in range(DetectionMap_sz_3):
                    for k in range(keypoints_num_sz):
                        reconDetectionMap[b, k, i, j] = (scoreMap[b, k, i, j] /(cur_rk_all[b, i, j] + epsilon_scale))
        """

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
        detectionmap_sz_0, detectionmap_sz_1, _, _ = DetectionMap.shape
        for b in range(detectionmap_sz_0):
            for k in range(detectionmap_sz_1):
                softmask[b, k, :, :] = DetectionMap[b, k, :, :] / zeta[b, k]

        #print("softmask", softmask.shape)
        return softmask