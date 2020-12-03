import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from torch.autograd import Variable
import time
from model.layers import Linear, dec_Linear
from StackedHourglass import StackedHourglassForKP, StackedHourglassImgRecon
from DetectionConfidenceMap import DetectionConfidenceMap2keypoint, ReconDetectionConfidenceMap, create_softmask, LinearReconDetectionMap
from loss import loss_concentration, loss_separation, loss_reconstruction
from utils import my_dataset, saveKPimg
import os
import numpy as np
from parallel import DataParallelModel, DataParallelCriterion
import seaborn
from tqdm import tqdm
import argparse


import warnings
warnings.filterwarnings("ignore")

from torch.utils.data import TensorDataset, DataLoader

import visdom
vis = visdom.Visdom()
plot_all = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]), opts=dict(title='All Loss'))
plot_recon = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]), opts=dict(title='Reconstruction Loss'))
plot_sep = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]), opts=dict(title='Separation Loss'))
plot_conc = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]), opts=dict(title='Concentration Loss'))

plot_all_iter = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]), opts=dict(title='All Loss per iter'))
plot_recon_iter = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]), opts=dict(title='Reconstruction Loss per iter'))
plot_sep_iter = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]), opts=dict(title='Separation Loss per iter'))
plot_conc_iter = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]), opts=dict(title='Concentration Loss per iter'))

torch.multiprocessing.set_start_method('spawn', force=True)
#########################################parameter#########################################
h0 = 480
w0 = 640

input_width = 208
#input_width = 128

num_of_kp = 200
feature_dimension = 128

num_epochs = 100
batch_size = 4

stacked_hourglass_inpdim_kp = input_width
stacked_hourglass_oupdim_kp = num_of_kp #number of my keypoints

#lg data
#pre_width = 128
#pre_height = 96

pre_width = 208
pre_height = 64

num_nstack = 6

learning_rate = 1e-4 #1e-3
weight_decay = 1e-5 #1e-5 #5e-4
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
    optimizer_StackedHourglass_kp = torch.optim.Adam(model_StackedHourglassForKP.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model_feature_descriptor = Linear(img_width=pre_width, img_height=pre_height, feature_dimension=feature_dimension)
    model_feature_descriptor = nn.DataParallel(model_feature_descriptor)
    model_feature_descriptor.cuda()
    optimizer_Wk_ = torch.optim.Adam(model_feature_descriptor.parameters(),  lr=learning_rate, weight_decay=weight_decay)

    model_detection_map = LinearReconDetectionMap(img_width=pre_width, img_height=pre_height, num_of_kp=num_of_kp)
    model_detection_map = nn.DataParallel(model_detection_map)
    model_detection_map.cuda()
    optimizer_reconDetection = torch.optim.Adam(model_detection_map.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model_dec_feature_descriptor = dec_Linear(feature_dimension=feature_dimension, img_width=pre_height, img_height=pre_width)
    model_dec_feature_descriptor = nn.DataParallel(model_dec_feature_descriptor)
    model_dec_feature_descriptor.cuda()
    optimizer_decfeatureDescriptor = torch.optim.Adam(model_dec_feature_descriptor.parameters(),  lr=learning_rate, weight_decay=weight_decay)

    model_StackedHourglassImgRecon = StackedHourglassImgRecon(num_of_kp=num_of_kp, nstack=num_nstack, inp_dim=stacked_hourglass_inpdim_kp, oup_dim=3, bn=False, increase=0)
    model_StackedHourglassImgRecon = nn.DataParallel(model_StackedHourglassImgRecon)
    model_StackedHourglassImgRecon.cuda()
    optimizer_ImgRecon = torch.optim.Adam(model_StackedHourglassImgRecon.parameters(), lr=learning_rate, weight_decay=weight_decay)

    ###################################################################################################################
    if os.path.exists("./SaveModelCKPT/train_model.pth"):
        checkpoint = torch.load("./SaveModelCKPT/train_model.pth")
        model_StackedHourglassForKP.load_state_dict(checkpoint['model_StackedHourglassForKP'])
        model_feature_descriptor.load_state_dict(checkpoint['model_feature_descriptor'])
        model_detection_map.load_state_dict(checkpoint['model_detection_map'])
        model_dec_feature_descriptor.load_state_dict(checkpoint['model_dec_feature_descriptor'])
        model_StackedHourglassImgRecon.load_state_dict(checkpoint['model_StackedHourglassImgRecon'])

        optimizer_StackedHourglass_kp.load_state_dict(checkpoint['optimizer_StackedHourglass_kp'])
        optimizer_Wk_.load_state_dict(checkpoint['optimizer_Wk_'])
        optimizer_reconDetection.load_state_dict(checkpoint['optimizer_reconDetection'])
        optimizer_decfeatureDescriptor.load_state_dict(checkpoint['optimizer_decfeatureDescriptor'])
        optimizer_ImgRecon.load_state_dict(checkpoint['optimizer_ImgRecon'])
    ###################################################################################################################

    """
    Batchnorm2d = torch.nn.BatchNorm2d(num_of_kp)
    Batchnorm2d = nn.DataParallel(Batchnorm2d)
    Batchnorm2d.cuda()

    recon_Batchnorm2d = torch.nn.BatchNorm2d(num_of_kp * 2)
    recon_Batchnorm2d = nn.DataParallel(recon_Batchnorm2d)
    recon_Batchnorm2d.cuda()
    """
    dataset = my_dataset()
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    save_loss = []
    print("***", time.time() - model_start) #7.26 ~ 63
    my_iteration = 0

    for epoch in tqdm(range(num_epochs)):
        print("\n===epoch=== ", epoch)
        running_loss = 0
        running_recon_loss = 0
        running_sep_loss = 0
        running_conc_loss = 0
        for i, data in enumerate(tqdm(train_loader)):
            start = time.time()
            input_img, cur_filename, kp_img = data
            #print("data loader time: ", time.time() - start)
            aefe_input = input_img.cuda() #(b, 3, 96, 128)

            cur_batch = aefe_input.shape[0]

            ##########################################ENCODER##########################################
            combined_hm_preds = model_StackedHourglassForKP(aefe_input)[:, num_nstack-1, :, :, :]
            #save heat map
            #if (epoch % 10 == 0):
            #heatmap_save_filename = ("SaveHeatMapImg/heatmap_%s_epoch_%s.jpg" % (cur_filename, epoch))
            #seaborn.heatmap(combined_hm_preds[0,0,:,:].detach().cpu().clone().numpy())
            #save_image(combined_hm_preds, heatmap_save_filename)
            #plt.savefig(heatmap_save_filename)

            #combined_hm_preds = Batchnorm2d(combined_hm_preds)
            start2 = time.time()
            fn_DetectionConfidenceMap2keypoint = DetectionConfidenceMap2keypoint(shgs_out=combined_hm_preds, shgs_out_dim=stacked_hourglass_oupdim_kp)
            DetectionMap, keypoints, zeta = fn_DetectionConfidenceMap2keypoint(combined_hm_preds, cur_batch, num_of_kp)
            #print("encoding: ", time.time()-start2)

            start3 = time.time()
            fn_softmask = create_softmask()
            softmask = fn_softmask(DetectionMap, zeta)  # (b,k,96,128)
            #print("softmask_creation: ", time.time() - start3)

            start_loss = time.time()
            fn_loss_concentration = loss_concentration().cuda()
            fn_loss_separation = loss_separation().cuda()
            cur_conc_loss = fn_loss_concentration(softmask)
            cur_sep_loss = fn_loss_separation(keypoints)
            #print("loss computation time:", time.time() - start_loss)

            start4 = time.time()
            get_descriptors = combined_hm_preds
            fn_relu = torch.nn.ReLU().cuda()
            #leakyrelu4descriptors = torch.nn.LeakyReLU(0.01).cuda()
            #get_descriptors = fn_relu(get_descriptors) #(b,k,96,128)

            Wk_raw = get_descriptors * DetectionMap
            Wk_rsz = Wk_raw.view(Wk_raw.shape[0], Wk_raw.shape[1], Wk_raw.shape[2] * Wk_raw.shape[3])
            Wk_ = model_feature_descriptor(Wk_rsz) #(b, k, 96*128) -> (b,k,128)
            mul_all_torch = (softmask*fn_relu(get_descriptors)).sum(dim=[2, 3]).unsqueeze(2)
            my_descriptor = (mul_all_torch * fn_relu(Wk_)) #(b, k, 128)
            #print("descriptor time", time.time() - start4)

            ##########################################DECODER##########################################
            start5 = time.time()
            #########
            #fn_reconDetectionMap = ReconDetectionConfidenceMap()
            #reconDetectionMap = fn_reconDetectionMap(keypoints, DetectionMap)
            ##########
            reconDetectionMap = model_detection_map(keypoints, DetectionMap)
            reconDetectionMap = fn_relu(reconDetectionMap)
            #print("recon Detection Map time:", time.time() - start5)

            start6 = time.time()
            til_Wk = model_dec_feature_descriptor(my_descriptor) #(b, 16, 128)
            dec_Wk = til_Wk.view(Wk_raw.shape[0], Wk_raw.shape[1], Wk_raw.shape[2], Wk_raw.shape[3])
            dec_Wk = fn_relu(dec_Wk)
            reconFeatureMap = dec_Wk * reconDetectionMap
            #print("recon Feature Map time: ", time.time() - start6)

            start7 = time.time()
            concat_recon = torch.cat((reconDetectionMap, reconFeatureMap), 1) #(b, 2n, 96, 128) channel-wise concatenation
            #concat_recon = recon_Batchnorm2d(concat_recon)

            reconImg = model_StackedHourglassImgRecon(concat_recon) #(b, 8, 3, 192, 256)
            reconImg = reconImg[:, num_nstack-1, :, :, :] #(b,3,192,256)
            #print("reconstruction time: ", time.time() - start7)

            """
            my_sigmoid = nn.Sigmoid()
            reconImg = my_sigmoid(reconImg)
            reconImg = 0.5 * torch.log(1 + 2 * torch.exp(reconImg))
            """
            # ================Backward================
            optimizer_StackedHourglass_kp.zero_grad()
            optimizer_Wk_.zero_grad()
            optimizer_decfeatureDescriptor.zero_grad()
            optimizer_ImgRecon.zero_grad()
            optimizer_reconDetection.zero_grad()

            start8 = time.time()
            cur_recon_loss = F.mse_loss(reconImg, aefe_input.detach())

            loss = 1000 * cur_conc_loss.cuda() + 1e-4 * cur_sep_loss.cuda() + 1 * cur_recon_loss.cuda()

            #print("loss computation time: ", time.time() - start8)

            print("\n", "Raw Loss=>", "concentration loss: ", cur_conc_loss.item(), ",", "separation loss: ", cur_sep_loss.item(), ",", "cur_recon_loss: ", cur_recon_loss.item())
            print("Computing Loss=>", "concentration loss: ", 1000 * cur_conc_loss.item(), ",", "separation loss: ", 1e-4 * cur_sep_loss.item(), ",", "cur_recon_loss: ", 1 * cur_recon_loss.item())
            loss.backward()

            optimizer_StackedHourglass_kp.step()
            optimizer_Wk_.step()
            optimizer_decfeatureDescriptor.step()
            optimizer_ImgRecon.step()
            optimizer_reconDetection.step()

            running_loss = running_loss + loss.item()
            running_recon_loss = running_recon_loss + cur_recon_loss.item()
            running_sep_loss = running_sep_loss + 1e-4 * cur_sep_loss.item()
            running_conc_loss = running_conc_loss + 1000 * cur_conc_loss.item()

            print("Running Loss=>", "All loss", running_loss, "concentration loss: ", running_conc_loss, ",", "separation loss: ", running_sep_loss, ",", "cur_recon_loss: ", running_recon_loss)

            my_iteration = my_iteration + 1

            vis.line(Y=[running_loss], X=np.array([my_iteration]), win=plot_all_iter, update='append')
            vis.line(Y=[running_recon_loss], X=np.array([my_iteration]), win=plot_recon_iter, update='append')
            vis.line(Y=[running_sep_loss], X=np.array([my_iteration]), win=plot_sep_iter, update='append')
            vis.line(Y=[running_conc_loss], X=np.array([my_iteration]), win=plot_conc_iter, update='append')

            #print("batch training with backprop", time.time() - start)

            start13 = time.time()
            if (epoch % 5 == 0):
                fn_save_kpimg = saveKPimg()
                fn_save_kpimg(kp_img, keypoints, epoch, cur_filename)
                img_save_filename = ("SaveReconstructedImg/recon_%s_epoch_%s.jpg" % (cur_filename, epoch))
                save_image(reconImg, img_save_filename)
                if(epoch != 0):
                    torch.save({
                                'model_StackedHourglassForKP': model_StackedHourglassForKP.state_dict(),
                                'model_feature_descriptor': model_feature_descriptor.state_dict(),
                                'model_detection_map': model_detection_map.state_dict(),
                                'model_dec_feature_descriptor': model_dec_feature_descriptor.state_dict(),
                                'model_StackedHourglassImgRecon': model_StackedHourglassImgRecon.state_dict(),

                                'optimizer_StackedHourglass_kp': optimizer_StackedHourglass_kp.state_dict(),
                                'optimizer_Wk_': optimizer_StackedHourglass_kp.state_dict(),
                                'optimizer_decfeatureDescriptor': optimizer_decfeatureDescriptor.state_dict(),
                                'optimizer_ImgRecon': optimizer_ImgRecon.state_dict(),
                                'optimizer_reconDetection': optimizer_reconDetection.state_dict(),
                                }, "./SaveModelCKPT/train_model.pth")

        vis.line(Y=[running_loss], X=np.array([epoch]), win=plot_all, update='append')
        vis.line(Y=[running_recon_loss], X=np.array([epoch]), win=plot_recon, update='append')
        vis.line(Y=[running_sep_loss], X=np.array([epoch]), win=plot_sep, update='append')
        vis.line(Y=[running_conc_loss], X=np.array([epoch]), win=plot_conc, update='append')

        print('epoch [{}/{}], loss:{:.4f} '.format(epoch + 1, num_epochs, running_loss))
        
##########################################################################################################################
if __name__ == '__main__':
    torch.cuda.empty_cache()
    if not os.path.exists("SaveKPImg"):
        os.makedirs("SaveKPImg")
    if not os.path.exists("SaveReconstructedImg"):
        os.makedirs("SaveReconstructedImg")
    if not os.path.exists("SaveHeatMapImg"):
        os.makedirs("SaveHeatMapImg")
    if not os.path.exists("SaveModelCKPT"):
        os.makedirs("SaveModelCKPT")

    train()

# torch.save(model.state_dict(), './StackedHourGlass.pth')
##########################################################################################################################