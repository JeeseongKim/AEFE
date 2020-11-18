import torch
from torch import nn
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from torch.autograd import Variable
import time
import numpy as np
from model.layers import Linear, dec_Linear
from StackedHourglass import StackedHourglassForKP, StackedHourglassImgRecon
from DetectionConfidenceMap import DetectionConfidenceMap2keypoint, ReconDetectionConfidenceMap, create_softmask
from loss import loss_concentration, loss_separation
from utils import my_dataset

from torch.multiprocessing import Process
from multiprocessing import Pool

import warnings
warnings.filterwarnings("ignore")

from torch.utils.data import TensorDataset, DataLoader

import visdom
vis = visdom.Visdom()
#plot = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]))
plot = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]))
draw_epoch = []
draw_loss = []

torch.multiprocessing.set_start_method('spawn', force=True)
#########################################parameter#########################################
h0 = 480
w0 = 640

input_height = 192
input_width = 256

num_of_kp = 100
feature_dimension = 128

num_epochs = 100
batch_size = 8

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
    optimizer_ImgRecon = torch.optim.Adam(model_StackedHourglassImgRecon.parameters(), lr=1e-3, weight_decay=2e-4)

    dataset = my_dataset()
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    save_loss = []
    my_iteration = 0

    for epoch in range(num_epochs):
        print("===epoch=== ", epoch)
        running_loss = 0
        for i, data in enumerate(train_loader):
            start = time.time()
            torch.cuda.empty_cache()
            input_img, cur_filename = data

            input_img = Variable(input_img).cuda()
            aefe_input = input_img[:, 0, :, :, :] #(b, 3, 192, 256)
            cur_batch = aefe_input.shape[0]

            combined_hm_preds = model_StackedHourglassForKP(aefe_input)
            combined_hm_preds = Variable(combined_hm_preds.type(dtype), requires_grad=True)
            combined_hm_preds = combined_hm_preds[:, num_nstack-1, :, :, :] # => last layer #(1,n,96,128)

            fn_DetectionConfidenceMap2keypoint = DetectionConfidenceMap2keypoint(shgs_out=combined_hm_preds, shgs_out_dim=stacked_hourglass_oupdim_kp)
            #fn_DetectionConfidenceMap2keypoint = nn.DataParallel(fn_DetectionConfidenceMap2keypoint)
            fn_DetectionConfidenceMap2keypoint
            DetectionMap, keypoints, zeta = fn_DetectionConfidenceMap2keypoint(combined_hm_preds.cuda(), cur_batch, num_of_kp)
            #print("kp size", keypoints.shape)
            #print("kp time :", time.time() - start)
            #print("---DetectionMap and Keypoints are generated---")

            ############################## My Loss Function ############################
            #loss_fn_start = time.time()
            fn_loss_concentration = loss_concentration().cuda()
            #fn_loss_concentration = DataParallelCriterion(fn_loss_concentration)
            cur_conc_loss = fn_loss_concentration(DetectionMap, zeta)

            fn_loss_separation = loss_separation().cuda()
            #fn_loss_separation = DataParallelCriterion(fn_loss_separation)
            cur_sep_loss = fn_loss_separation(keypoints)

            fn_loss_recon = torch.nn.MSELoss().cuda()
            #fn_loss_recon = DataParallelCriterion(fn_loss_recon)
            #print("loss function setup time: ", time.time() - loss_fn_start)
            #############################################################################

            ## descriptors
            #descriptor_time = time.time()
            get_descriptors = combined_hm_preds

            #leakyrelu4descriptors = torch.nn.LeakyReLU(0.1).cuda()
            relu4descriptors = torch.nn.ReLU()
            get_descriptors = relu4descriptors(get_descriptors)
            #print("get_descriptors", get_descriptors.shape) #(b,k,96,128)

            #tensor_1 = get_descriptors
            #tensor_2 = DetectionMap
            Wk_raw = get_descriptors.cuda() * DetectionMap
            Wk_rsz = Wk_raw.view(Wk_raw.shape[0], Wk_raw.shape[1], Wk_raw.shape[2] * Wk_raw.shape[3]).cuda(1)

            Wk_ = model_feature_descriptor(Wk_rsz)
            Wk_ = Variable(Wk_.type(dtype), requires_grad=True)
            #print("descriptor weight: ", time.time() - descriptor_time)

            ######-------decoder std -------######
            std_x = 5
            std_y = 5
            correlation = 0.3

            softmask_std_x = std_x
            softmask_std_y = std_y
            reconDetectionMap_std_x = std_x
            reconDetectionMap_std_y = std_y
            #######################################

            #softmask_start = time.time()
            fn_softmask = create_softmask()
            softmask = fn_softmask(softmask_std_x, softmask_std_y, keypoints, DetectionMap, zeta) #(b,k,96,128)
            #softmask = softmask.cuda()
            #print("softmask creation time :", time.time() - softmask_start)

            descriptor_gen_time = time.time()
            mul_all_torch = torch.zeros(softmask.shape[0], softmask.shape[1], 1)
            sm_sz_0, sm_sz_1, sm_sz_2, sm_sz_3 = softmask.shape
            for b in range(sm_sz_0):
                for k in range(sm_sz_1):
                    #mul = 0
                    #mul = (softmask[b, k, :, :] * get_descriptors[b, k, :, :]).sum()
                    """
                    for i in range(sm_sz_2):
                        for j in range(sm_sz_3):
                            mul_1 = softmask[b, k, i, j]
                            mul_2 = get_descriptors[b, k, i, j]
                            mul += mul_1 * mul_2
                    """
                    mul = (softmask[b, k, :, :].cuda() * get_descriptors[b, k, :, :].cuda()).sum()
                    #print(mul)
                    mul_all_torch[b, k, :] = mul

            my_descriptor = (mul_all_torch * Wk_) #(b, 16, 128)
            #print("my_descriptor_gen time: ", time.time() - descriptor_gen_time)

            recon_start = time.time()
            fn_reconDetectionMap = ReconDetectionConfidenceMap()
            fn_reconDetectionMap = fn_reconDetectionMap
            reconDetectionMap = fn_reconDetectionMap(cur_batch, keypoints, reconDetectionMap_std_x, reconDetectionMap_std_y, correlation, DetectionMap)
            #print("recon time: ", time.time() - recon_start)
            #print("---DetectionMaps are generated---")

            dec_descriptor_start = time.time()
            til_Wk = model_dec_feature_descriptor(my_descriptor) #(b, 16, 128)
            til_Wk = Variable(til_Wk.type(dtype), requires_grad=True)
            #print("***", til_Wk.shape) #(b, n, h*w)
            dec_Wk = til_Wk.view(Wk_raw.shape[0], Wk_raw.shape[1], Wk_raw.shape[2], Wk_raw.shape[3])
            #print("***", dec_Wk.shape) #(b,n,96,128)

            #leakyrelu4dec_Wk = torch.nn.LeakyReLU(0.1).cuda(1)
            relu4dec_Wk = torch.nn.ReLU()
            dec_Wk = relu4dec_Wk(dec_Wk)
            # print("***", dec_Wk.shape) #(b,n,96,128)
            #print("recon descriptor time: ", time.time() - dec_descriptor_start)

            reconFeatureMap = dec_Wk * reconDetectionMap
            concat_recon = torch.cat((reconDetectionMap, reconFeatureMap), 1) #(b, 2n, 96, 128)
            #concat_recon = concat_recon.view(concat_recon.shape[0], concat_recon.shape[1], concat_recon.shape[2], concat_recon.shape[3]).cuda(1)
            #print("concat_recon", concat_recon.shape)  #(b, 2n, 96, 128)

            recon_img_start = time.time()
            reconImg = model_StackedHourglassImgRecon(concat_recon).cuda(1)
            reconImg = Variable(reconImg.type(dtype), requires_grad=True).cuda(1) #(b, 8, 3, 192, 256)
            #print("reconImg111", reconImg.shape)
            reconImg = reconImg[:, num_nstack-1, :, :, :] #(b,3,192,256)
            #print("recon img time: ", time.time() - recon_img_start)

            # ================Backward================
            #print("---Optimizer---")
            optimizer_StackedHourglass_kp.zero_grad()
            optimizer_Wk_.zero_grad()
            optimizer_decfeatureDescriptor.zero_grad()
            optimizer_ImgRecon.zero_grad()

            cur_recon_loss = fn_loss_recon(reconImg.cuda(), aefe_input.cuda())

            loss = 1 * cur_conc_loss.cuda() + 0.8 * cur_sep_loss.cuda() + 6000*cur_recon_loss.cuda()
            print("Loss=>", "concentration loss: ", cur_conc_loss.item(), ",", "separation loss: ", cur_sep_loss.item(), "," , "cur_recon_loss: ", cur_recon_loss.item())


            """
            #save img tmp
            print("saving my currently reconstructed image")
            save_my_reconImg_tmp = reconImg.detach().cpu().numpy()
            save_image(reconImg, "test.jpg")
            print("----saving is done----")
            """
            loss.backward()

            optimizer_StackedHourglass_kp.step()
            #optimizer_StackedHourglass_desc.step()
            optimizer_Wk_.step()
            optimizer_decfeatureDescriptor.step()
            optimizer_ImgRecon.step()

            running_loss = running_loss + loss.item()
            my_iteration = my_iteration + 1

            draw_epoch.append(my_iteration)
            draw_loss.append(loss)
            #print("draw epoch: ", draw_epoch)
            #print("draw loss: ", draw_loss)

            #vis.line(Y=[loss], X=np.array([my_iteration]), win=plot, update='append')
            #vis.line(Y=draw_loss, X=draw_epoch, win=plot, update='append')

        print('epoch [{}/{}], loss:{:.4f} '.format(epoch + 1, num_epochs, running_loss))
        vis.line(Y=[running_loss], X=np.array([epoch]), win=plot, update='append')
        #vis.line(Y=[running_loss], X=np.array([epoch]), win=plot, update='append')

        save_loss.append(running_loss)

        if (epoch % 10 == 0):
            img_save_filename = ("SaveReconstructedImg/recon_%s_epoch_%s.jpg" % (cur_filename, epoch))
            save_my_reconImg = reconImg.detach().cpu().numpy()
            save_image(reconImg, img_save_filename)
            ##print("---saving image--- ", img_save_filename)

    plt.plot(save_loss)

##########################################################################################################################
if __name__ == '__main__':
    torch.cuda.empty_cache()
    train()

# torch.save(model.state_dict(), './StackedHourGlass.pth')
##########################################################################################################################