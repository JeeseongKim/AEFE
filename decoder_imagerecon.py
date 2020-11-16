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

"""class recon_img(nn.Module):
        def __init__(self):

        def forward(self):
        recon_img = []

        return recon_img"""