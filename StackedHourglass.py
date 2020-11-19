import torch
from torch import nn
from model.layers import Conv, Residual, Hourglass, Merge, Pool, upsample_recon
from model.Heatmap import HeatmapLoss

class StackedHourglassForKP(nn.Module):
    def __init__(self, nstack, inp_dim, oup_dim, bn=False, increase=0, **kwargs):
        super(StackedHourglassForKP, self).__init__()

        self.nstack = nstack

        self.pre = nn.Sequential(
            Conv(3, 64, 3, 1, bn=True, relu=True),
            Residual(64, 128),
            Pool(2, 2),
            Residual(128, 128),
            Residual(128, inp_dim)
        )

        self.hgs = nn.ModuleList([
            nn.Sequential(
                Hourglass(4, inp_dim, bn, increase),
            ) for i in range(nstack)])

        self.features = nn.ModuleList([
            nn.Sequential(
                Residual(inp_dim, inp_dim),
                Conv(inp_dim, inp_dim, 1, bn=True, relu=True)
            ) for i in range(nstack)])

        self.outs = nn.ModuleList([
            nn.Sequential(
                Conv(inp_dim, oup_dim, 1, relu=False, bn=False),
            )for i in range(nstack)
         ])

        self.merge_features = nn.ModuleList([
            Merge(inp_dim, inp_dim) for i in range(nstack - 1)
        ])

        self.merge_preds = nn.ModuleList([
            Merge(oup_dim, inp_dim) for i in range(nstack - 1)
        ])

        self.nstack = nstack

        self.heatmapLoss = HeatmapLoss()

    def forward(self, imgs):
        ## our posenet
        # print(imgs.shape)
        # x = imgs.permute(0, 3, 1, 2) #x of size 1,3, inpdim, inpdim
        x = imgs #(b,3,192,256)
        x = self.pre(x) #(b,3,192,256)
        combined_hm_preds = []
        append = combined_hm_preds.append
        for i in range(self.nstack):
            hg = self.hgs[i](x)
            #print("hg", hg.shape) #(b, 256, 96, 128)
            feature = self.features[i](hg)
            preds = self.outs[i](feature) #--> heatmap prediction
            #combined_hm_preds.append(preds)
            append(preds)
            #print("combined_hm_preds", combined_hm_preds.shape)
            if (i < self.nstack - 1):
                x = x + self.merge_preds[i](preds) + self.merge_features[i](feature)
        #print("***", torch.stack(combined_hm_preds, 1).shape) #(0.5*b, n_stack, 100, 96, 128)
        return torch.stack(combined_hm_preds, 1)

class StackedHourglassForDesc(nn.Module):
    def __init__(self, nstack, inp_dim, oup_dim, bn=False, increase=0, **kwargs):
        super(StackedHourglassForDesc, self).__init__()
        self.nstack = nstack
        self.pre = nn.Sequential(
            Conv(3, 64, 3, 1, bn=True, relu=True),
            Residual(64, 128),
            Pool(2, 2),
            Residual(128, 128),
            Residual(128, inp_dim)
        )

        self.hgs = nn.ModuleList([
            nn.Sequential(
                Hourglass(4, inp_dim, bn, increase),
            ) for i in range(nstack)])

        self.features = nn.ModuleList([
            nn.Sequential(
                Residual(inp_dim, inp_dim),
                Conv(inp_dim, inp_dim, 1, bn=True, relu=True)
            ) for i in range(nstack)])

        self.outs = nn.ModuleList([
            nn.Sequential(
                Conv(inp_dim, oup_dim, 1, relu=False, bn=False),
                # Conv(oup_dim, 3, 1, relu=False, bn=False)
            ) for i in range(nstack)
        ])

        self.merge_features = nn.ModuleList([
            Merge(inp_dim, inp_dim) for i in range(nstack - 1)
        ])

        self.merge_preds = nn.ModuleList([
            Merge(oup_dim, inp_dim) for i in range(nstack - 1)
        ])

        self.nstack = nstack

        self.heatmapLoss = HeatmapLoss()

    def forward(self, imgs):
        x = imgs
        x = self.pre(x)
        combined_hm_preds = []
        for i in range(self.nstack):
            # print("x", x.shape)
            hg = self.hgs[i](x)
            # print("hg", hg.shape)
            feature = self.features[i](hg)
            # print("feature", feature.shape)
            preds = self.outs[i](feature)  # --> heatmap prediction
            #print("preds", preds.shape) #--> (1,16,64,64)
            # print("111")
            combined_hm_preds.append(preds)
            # print("222")
            #print("combined_hm_preds", combined_hm_preds.shape)
            if i < self.nstack - 1:
                x = x + self.merge_preds[i](preds) + self.merge_features[i](feature)

        return torch.stack(combined_hm_preds, 1)

class StackedHourglassImgRecon(nn.Module):
    def __init__(self,num_of_kp, nstack, inp_dim, oup_dim, bn=False, increase=0, **kwargs):
        super(StackedHourglassImgRecon, self).__init__()
        self.nstack = nstack
        self.pre = nn.Sequential(
            Conv(2 * num_of_kp, 2 * num_of_kp, 3, 1, bn=True, relu=True), #(2n, hidden_1, 3,1)
            Conv(2 * num_of_kp, 128, 3, 1, bn=True, relu=True),  # (2n, hidden_1, 3,1)
            Conv(128, 128, 3, 1, bn=True, relu=True),
            #upsample_recon(2, mode='bilinear', align_corners=True),
            Residual(128, 128),
            #upsample_recon(2, mode='bilinear', align_corners=True),
            #Residual(128, 128),
            Residual(128, inp_dim),
            Residual(inp_dim, inp_dim)
        )

        self.hgs = nn.ModuleList([
            nn.Sequential(
                Hourglass(4, inp_dim, bn, increase),
            ) for i in range(nstack)])

        self.features = nn.ModuleList([
            nn.Sequential(
                Residual(inp_dim, inp_dim),
                Conv(inp_dim, inp_dim, 1, bn=True, relu=True)
            ) for i in range(nstack)])

        self.outs = nn.ModuleList([
            nn.Sequential(
                Conv(inp_dim, oup_dim, 1, relu=False, bn=False),
                # Conv(oup_dim, 3, 1, relu=False, bn=False)
            ) for i in range(nstack)
        ])

        self.merge_features = nn.ModuleList([
            Merge(inp_dim, inp_dim) for i in range(nstack - 1)
        ])

        self.merge_preds = nn.ModuleList([
            Merge(oup_dim, inp_dim) for i in range(nstack - 1)
        ])

        self.nstack = nstack

        self.heatmapLoss = HeatmapLoss()

    def forward(self, concat_recon):
        x = concat_recon
        #print("x", x.shape) #(b, 2n, 96, 128)
        x = self.pre(x) #(b, 256, 96, 128)
        #print("pre(x)", x.shape)
        model_upsample = upsample_recon(2, mode='bilinear', align_corners=True)
        x = model_upsample(x)
        #print("upsample_x",  x.shape) #(1,256,192,256)
        combined_hm_preds = []
        append = combined_hm_preds.append
        for i in range(self.nstack):
            hg = self.hgs[i](x)
            #print("hg", hg.shape) #(1,256,192,256)
            feature = self.features[i](hg)
            #print("feature", feature.shape) #(1,256,192,256)
            preds = self.outs[i](feature)  # --> heatmap prediction
            #print("111")
            #print("preds", preds.shape) #--> (1,16,64,64)
            #combined_hm_preds.append(preds)
            append(preds)
            #print("222")
            if i < self.nstack - 1:
                x = x + self.merge_preds[i](preds) + self.merge_features[i](feature)
            # print("new X", x.shape) #=> (1,256,64,64)
            # print("new X", x)
        #print("combined_hm_preds",combined_hm_preds.shape)
        return torch.stack(combined_hm_preds, 1)
