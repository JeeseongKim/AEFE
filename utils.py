import torch
from torch import nn
import random
import math
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import glob
from torchvision import transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import os
import cv2


class make_transformation_M (nn.Module):
    def __init__(self):
        super(make_transformation_M, self).__init__()

    def forward(self):
        dx = random.uniform(-10.0, 10.0)
        dy = random.uniform(-10.0, 10.0)
        theta = random.uniform(-math.pi, math.pi)

        transformation_g = torch.zeros(4, 4)
        transformation_g[0, 0] = math.cos(theta)
        transformation_g[0, 1] = -math.sin(theta)
        transformation_g[0, 3] = dx

        transformation_g[1, 0] = math.sin(theta)
        transformation_g[1, 1] = math.cos(theta)
        transformation_g[1, 3] = dy

        transformation_g[2, 2] = 1.0
        transformation_g[3, 3] = 1.0

        return transformation_g

class my_dataset(Dataset):
    def __init__(self):
        self.img_filename = []
        self.dataset_img = []
        self.dataset_filename = []
        #whole_flist = sorted(glob.glob('./data/*.jpg'))
        input_height = 192
        input_width = 256

        for filename in (sorted(glob.glob('./data/*.jpg'))):
            im = Image.open(filename)
            img_rsz = cv2.resize(np.array(im), (input_width, input_height)) #(192,256,3)
            img_tensor_input = transforms.ToTensor()(img_rsz).unsqueeze(0)  # (1,3,192,256)

            self.dataset_img.append(img_tensor_input)
            self.dataset_filename.append(filename.split('.')[1].split('/')[2])

        self.len = len(self.dataset_img)

    def __getitem__(self, index):
        return self.dataset_img[index], self.dataset_filename[index]

    def __len__(self):
        return len(self.dataset_img)