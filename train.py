import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from torch.autograd import Variable
import time
from model.layers import Linear, dec_Linear
from StackedHourglass import StackedHourglassForKP, StackedHourglassImgRecon
from DetectionConfidenceMap import DetectionConfidenceMap2keypoint, ReconDetectionConfidenceMap, create_softmask
from loss import loss_concentration, loss_separation, loss_reconstruction
from utils import my_dataset, saveKPimg

from torch.multiprocessing import Process

import cv2
import numpy as np

import warnings
warnings.filterwarnings("ignore")

from torch.utils.data import TensorDataset, DataLoader

import visdom
vis = visdom.Visdom()
plot = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]))

torch.multiprocessing.set_start_method('spawn', force=True)
#########################################parameter#########################################
h0 = 480
w0 = 640

input_height = 192
input_width = 256

num_of_kp = 200
feature_dimension = 128

num_epochs = 100
batch_size = 4

stacked_hourglass_inpdim_kp = input_width
stacked_hourglass_oupdim_kp = num_of_kp #number of my keypoints

stacked_hourglass_inpdim_desc = input_width
stacked_hourglass_oupdim_desc = num_of_kp # same as stacked_hourglass_oupdim_kp = 16

pre_width = 128
pre_height = 96

num_nstack = 2
###########################################################################################
concat_recon = []
dtype = torch.FloatTensor
######################################################################################################################################################################################
#@profile
def train():

    model_start = time.time()

    model_StackedHourglassForKP = StackedHourglassForKP(nstack=num_nstack, inp_dim=stacked_hourglass_inpdim_kp, oup_dim=stacked_hourglass_oupdim_kp, bn=False, increase=0)
    model_StackedHourglassForKP = nn.DataParallel(model_StackedHourglassForKP)
    model_StackedHourglassForKP.cuda()
    optimizer_StackedHourglass_kp = torch.optim.Adam(model_StackedHourglassForKP.parameters(), lr=1e-3, weight_decay=2e-4)

    model_feature_descriptor = Linear(img_width=pre_width, img_height=pre_height, feature_dimension=feature_dimension)
    model_feature_descriptor = nn.DataParallel(model_feature_descriptor)
    model_feature_descriptor.cuda()
    optimizer_Wk_ = torch.optim.Adam(model_feature_descriptor.parameters(), lr=1e-3, weight_decay=2e-4)

    model_dec_feature_descriptor = dec_Linear(feature_dimension=feature_dimension, img_width=pre_height, img_height=pre_width)
    model_dec_feature_descriptor = nn.DataParallel(model_dec_feature_descriptor)
    model_dec_feature_descriptor.cuda()
    optimizer_decfeatureDescriptor = torch.optim.Adam(model_dec_feature_descriptor.parameters(), lr=1e-3, weight_decay=2e-4)

    model_StackedHourglassImgRecon = StackedHourglassImgRecon(num_of_kp=num_of_kp, nstack=num_nstack, inp_dim=stacked_hourglass_inpdim_kp, oup_dim=3, bn=False, increase=0)
    model_StackedHourglassImgRecon = nn.DataParallel(model_StackedHourglassImgRecon)
    model_StackedHourglassImgRecon.cuda()
    optimizer_ImgRecon = torch.optim.Adam(model_StackedHourglassImgRecon.parameters(), lr=1e-4, weight_decay=1e-5)

    Batchnorm2d = torch.nn.BatchNorm2d(num_of_kp)
    Batchnorm2d = nn.DataParallel(Batchnorm2d)
    Batchnorm2d.cuda()

    recon_Batchnorm2d = torch.nn.BatchNorm2d(num_of_kp * 2)
    recon_Batchnorm2d = nn.DataParallel(recon_Batchnorm2d)
    recon_Batchnorm2d.cuda()

    dataset = my_dataset()
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    save_loss = []
    print("***", time.time() - model_start) #7.26

    for epoch in range(num_epochs):
        print("===epoch=== ", epoch)
        running_loss = 0
        for i, data in enumerate(train_loader):
            start = time.time()
            #torch.cuda.empty_cache()
            input_img, cur_filename, kp_img = data
            #print("data loader time: ", time.time() - start)
            input_img = Variable(input_img, requires_grad=True).cuda()
            aefe_input = input_img[:, 0, :, :, :] #(b, 3, 192, 256)
            cur_batch = aefe_input.shape[0]

            start2 = time.time()
            combined_hm_preds = model_StackedHourglassForKP(aefe_input)
            #combined_hm_preds = Variable(combined_hm_preds.type(dtype), requires_grad=True) #(b, n_stack, n, 96, 128)
            combined_hm_preds = combined_hm_preds[:, num_nstack-1, :, :, :] # => last layer #(b,n,96,128)
            #combined_hm_preds = Batchnorm2d(combined_hm_preds)
            #print("stacked hourglass model: ", time.time() - start2)

            #start3 = time.time()
            fn_DetectionConfidenceMap2keypoint = DetectionConfidenceMap2keypoint(shgs_out=combined_hm_preds, shgs_out_dim=stacked_hourglass_oupdim_kp)
            DetectionMap, keypoints, zeta = fn_DetectionConfidenceMap2keypoint(combined_hm_preds, cur_batch, num_of_kp)
            keypoints = keypoints.cuda()
            #print("KPtime: ", time.time() - start3)

            ############################## My Loss Function ############################
            #start4 = time.time()
            fn_loss_concentration = loss_concentration().cuda()
            #cur_conc_loss = fn_loss_concentration(DetectionMap, zeta)

            fn_loss_separation = loss_separation().cuda()
            cur_sep_loss = fn_loss_separation(keypoints)

            #fn_loss_recon = nn.MSELoss()#loss_reconstruction().cuda()
            #print("loss time: ", time.time() - start4)
            #############################################################################
            #start5 = time.time()
            fn_softmask = create_softmask()
            softmask = fn_softmask(DetectionMap, zeta) #(b,k,96,128)
            #print("softmask time: ", time.time() - start5)

            #start6 = time.time()
            cur_conc_loss = fn_loss_concentration(softmask)
            #print("conc loss: ", time.time() - start6)

            #start7 = time.time()
            get_descriptors = combined_hm_preds
            #get_descriptors = Variable(get_descriptors.type(dtype), requires_grad=True)  # (b, n_stack, n, 96, 128)

            leakyrelu4descriptors = torch.nn.LeakyReLU(0.01).cuda()
            #relu4descriptors = torch.nn.ReLU()
            get_descriptors = leakyrelu4descriptors(get_descriptors) #(b,k,96,128)

            Wk_raw = get_descriptors * DetectionMap
            #Wk_rsz = Wk_raw.view(Wk_raw.shape[0], Wk_raw.shape[1], Wk_raw.shape[2] * Wk_raw.shape[3]).cuda()
            Wk_rsz = Wk_raw.view(Wk_raw.shape[0], Wk_raw.shape[1], Wk_raw.shape[2] * Wk_raw.shape[3])
            Wk_ = model_feature_descriptor(Wk_rsz)
            #Wk_ = Variable(Wk_.type(dtype), requires_grad=True)

            mul_all_torch = (softmask*get_descriptors).sum(dim=[2,3]).unsqueeze(2)
            my_descriptor = (mul_all_torch * Wk_) #(b, 16, 128)
            #print("descriptor time: ", time.time() - start7)

            #start8 = time.time()
            fn_reconDetectionMap = ReconDetectionConfidenceMap()
            reconDetectionMap = fn_reconDetectionMap(cur_batch, keypoints, DetectionMap)
            #print("reconDetectionMap time: ", time.time() - start8)

            #start9 = time.time()
            til_Wk = model_dec_feature_descriptor(my_descriptor) #(b, 16, 128)
            #til_Wk = Variable(til_Wk.type(dtype), requires_grad=True).cuda(1)
            #til_Wk = Variable(til_Wk.type(dtype), requires_grad=True)
            dec_Wk = til_Wk.view(Wk_raw.shape[0], Wk_raw.shape[1], Wk_raw.shape[2], Wk_raw.shape[3])
            #relu4dec_Wk = torch.nn.ReLU()
            dec_Wk = leakyrelu4descriptors(dec_Wk)
            reconFeatureMap = dec_Wk * reconDetectionMap
            #print("reconFeatureMap time: ", time.time() - start9)

            #start10 = time.time()
            # reconDetectionMap shape [b, n, 96, 128]
            # reconFeatureMap shape [b, n, 96, 128]
            concat_recon = torch.cat((reconDetectionMap, reconFeatureMap), 1) #(b, 2n, 96, 128) channel-wise concatenation
            #concat_recon = recon_Batchnorm2d(concat_recon)
            #print("concat time: ", time.time() - start10)

            #start11 = time.time()
            #reconImg = model_StackedHourglassImgRecon(concat_recon).cuda(1)
            #reconImg = Variable(reconImg.type(dtype), requires_grad=True).cuda(1) #(b, 8, 3, 192, 256)
            reconImg = model_StackedHourglassImgRecon(concat_recon)
            #reconImg = Variable(reconImg.type(dtype), requires_grad=True) #(b, 8, 3, 192, 256)
            reconImg = reconImg[:, num_nstack-1, :, :, :] #(b,3,192,256)

            my_sigmoid = nn.Sigmoid()
            reconImg = my_sigmoid(reconImg)
            reconImg = 0.5 * torch.log(1 + 2 * torch.exp(reconImg))
            #print("Image reconstruction time: ", time.time() - start11)
            #recon_visdom_show = vis.image(reconImg)

            # ================Backward================
            #start12 = time.time()
            #print("---Optimizer---")
            optimizer_StackedHourglass_kp.zero_grad()
            optimizer_Wk_.zero_grad()
            optimizer_decfeatureDescriptor.zero_grad()
            optimizer_ImgRecon.zero_grad()

            cur_recon_loss = F.mse_loss(reconImg, aefe_input.detach())
            #print("**", cur_recon_loss)
            #loss = 3 * cur_conc_loss.cuda() + 1 * cur_sep_loss.cuda() + 1e-7 * cur_recon_loss.cuda()
            loss = cur_recon_loss

            print("Loss=>", "concentration loss: ", cur_conc_loss.item(), ",", "separation loss: ", cur_sep_loss.item(), "," , "cur_recon_loss: ", cur_recon_loss.item())

            loss.backward()

            optimizer_StackedHourglass_kp.step()
            optimizer_Wk_.step()
            optimizer_decfeatureDescriptor.step()
            optimizer_ImgRecon.step()

            running_loss = running_loss + loss.item()
            #my_iteration = my_iteration + 1
            #print("back propagation time: ", time.time() - start12)
            #print("***", time.time() - start)
            #torch.cuda.empty_cache()

            #start13 = time.time()

            if (epoch % 10 == 0):
                fn_save_kpimg = saveKPimg()
                fn_save_kpimg(kp_img, keypoints, epoch, cur_filename)
                img_save_filename = ("SaveReconstructedImg/recon_%s_epoch_%s.jpg" % (cur_filename, epoch))
                save_image(reconImg, img_save_filename)
            #print("im save time: ", time.time() - start13)
        print('epoch [{}/{}], loss:{:.4f} '.format(epoch + 1, num_epochs, running_loss))
        vis.line(Y=[running_loss], X=np.array([epoch]), win=plot, update='append')

        save_loss.append(running_loss)
        #plot_recon_img = vis.image(reconImg)

    plt.plot(save_loss)

##########################################################################################################################
if __name__ == '__main__':
    torch.cuda.empty_cache()
    train()

# torch.save(model.state_dict(), './StackedHourGlass.pth')
##########################################################################################################################