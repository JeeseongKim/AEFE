import torch
from torch import nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('nn.Conv2d') != -1:
        torch.nn.init.kaiming_normal_(m.weight)
    elif classname.find('nn.Linear') != -1:
        torch.nn.init.xavier_normal_(m.weight)


Pool = nn.MaxPool2d

def batchnorm(x):
    return nn.BatchNorm2d(x.size()[1])(x)

class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=True)
        self.relu = None
        self.bn = None
        # print("inp_dim = ", inp_dim)
        # print("out_dim = " , out_dim)
        # print("kernel_size = ",  kernel_size)
        # print("stride = " , stride)

        if relu:
            self.relu = nn.ReLU()
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        self.conv.apply(weights_init)
        #print("convconvconv")
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(Residual, self).__init__()
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(inp_dim)
        self.conv1 = Conv(inp_dim, int(out_dim / 2), 1, relu=False)  # kernel= 1, stride=1
        self.bn2 = nn.BatchNorm2d(int(out_dim / 2))
        self.conv2 = Conv(int(out_dim / 2), int(out_dim / 2), 3, relu=False)
        self.bn3 = nn.BatchNorm2d(int(out_dim / 2))
        self.conv3 = Conv(int(out_dim / 2), out_dim, 1, relu=False)
        self.skip_layer = Conv(inp_dim, out_dim, 1, relu=False)
        if inp_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True

    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.bn1(x)
        out = self.relu(out)
        self.conv1.apply(weights_init)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        self.conv2.apply(weights_init)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        self.conv3.apply(weights_init)
        out = self.conv3(out)
        out += residual
        return out

class Hourglass(nn.Module):
    def __init__(self, n, f, bn=None, increase=0):
        super(Hourglass, self).__init__()
        nf = f + increase
        self.up1 = Residual(f, f)
        # Lower branch
        self.pool1 = Pool(2, 2)
        self.low1 = Residual(f, nf)
        self.n = n
        # Recursive hourglass
        if self.n > 1:
            self.low2 = Hourglass(n - 1, nf, bn=bn)
        else:
            self.low2 = Residual(nf, nf)
        self.low3 = Residual(nf, f)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        up1 = self.up1(x)
        pool1 = self.pool1(x)
        low1 = self.low1(pool1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2 = self.up2(low3)

        return up1 + up2

class UnFlatten(nn.Module):
    def forward(self, input):
        return input.view(-1, 256, 4, 4)

class Merge(nn.Module):
    def __init__(self, x_dim, y_dim):
        super(Merge, self).__init__()
        self.conv = Conv(x_dim, y_dim, 1, relu=False, bn=False)

    def forward(self, x):
        self.conv.apply(weights_init)
        return self.conv(x)

class Linear(nn.Module):
    def __init__(self, img_width, img_height, feature_dimension):
        super(Linear, self).__init__()
        self.linear = torch.nn.Linear(img_width*img_height, feature_dimension)

    def forward(self, input):
        self.linear.apply(weights_init)
        return self.linear(input)

class dec_Linear(nn.Module):
    def __init__(self, feature_dimension, img_width, img_height):
        super(dec_Linear, self).__init__()
        self.linear_dec = torch.nn.Linear(feature_dimension, img_width * img_height)

    def forward(self, input):
        self.linear_dec.apply(weights_init)
        return self.linear_dec(input)

class img_rsz_Linear(nn.Module):
    def __init__(self, from_input_1, from_input_2, to_output_1, to_output_2):
        super(img_rsz_Linear, self).__init__()

        self.linear_img_rsz_final = torch.nn.Linear(from_input_1*from_input_2, to_output_1*to_output_2)

    def forward(self, input):
        self.linear_img_rsz_final.apply(weights_init)
        return self.linear_img_rsz_final(input)


class upsample_recon(nn.Module):
    def __init__(self, scale_factor, mode, align_corners):
        super(upsample_recon, self).__init__()
        self.model_upsample_recon = torch.nn.Upsample(scale_factor=scale_factor, mode=mode, align_corners=align_corners)

    def forward(self, input):
        return self.model_upsample_recon(input)