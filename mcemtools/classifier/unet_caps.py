import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

def dynamic_routing(u_hat, squash, routing_iterations=3):
    b_ij = torch.zeros(*u_hat.size(), device = u_hat.device)
    for iterations in range(routing_iterations):
        c_ij = softmax(b_ij, dim=2)
        s_j = (c_ij*u_hat).sum(dim=2, keepdim=True)
        v_j = squash(s_j)
        if iterations < routing_iterations-1:
            a_ij = (u_hat * v_j).sum(dim=-1, keepdim=True)
            b_ij = b_ij + a_ij
    return v_j

class ClassifierCaps(nn.Module):
    def __init__(self, num_classes, Prim_out_channels=32,
                 Classifier_in_channels=8, Classifier_out_channels=16, 
                 routing_iterations=3, PrimaryCaps_n_pix = 6*6):
        super(ClassifierCaps, self).__init__()
        self.routing_iterations = routing_iterations
        self.W = nn.Parameter(torch.randn(
            num_classes, Prim_out_channels*PrimaryCaps_n_pix, 
            Classifier_in_channels, Classifier_out_channels))
        
    def forward(self, x):
        x = self.squash(x)
        x = x[None, :, :, None, :]
        W = self.W[:, None, :, :, :]
        x_hat = torch.matmul(x, W)
        
        v_j = dynamic_routing(
            x_hat, self.squash, routing_iterations=self.routing_iterations)
        
        v_j = F.leaky_relu(v_j)**2
        v_j = v_j.squeeze().transpose(0, 1)
        v_j_sum = v_j.sum(dim=(1, 2), keepdim=True)
        v_j = (v_j / v_j_sum).float()
        
        return v_j

    def squash(self, x):
        squared_norm = (x**2).sum(dim=-1, keepdim=True)
        scale = squared_norm/(1+squared_norm)
        out = scale * x/torch.sqrt(squared_norm)
        return out

class unet_caps(nn.Module):
    def __init__(self, num_classes, img_ch=16, output_ch=16, n_kernels = 64,
                 Prim_out_channels=32, Classifier_in_channels=8,
                 Classifier_out_channels=16, routing_iterations=3, 
                 PrimaryCaps_n_pix = 6*6):
        super(unet_caps,self).__init__()
        
        self.classifiercaps = ClassifierCaps(
            num_classes, Prim_out_channels=32, Classifier_in_channels=8,
            Classifier_out_channels=16, routing_iterations=3, 
            PrimaryCaps_n_pix = 6*6)
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=n_kernels)
        self.Conv2 = conv_block(ch_in=n_kernels,ch_out=2*n_kernels)
        self.Conv3 = conv_block(ch_in=2*n_kernels,ch_out=4*n_kernels)
        self.Conv4 = conv_block(ch_in=4*n_kernels,ch_out=8*n_kernels)
        self.Conv5 = conv_block(ch_in=8*n_kernels,ch_out=16*n_kernels)

        self.Up5 = up_conv(ch_in=16*n_kernels,ch_out=8*n_kernels)
        self.Up_conv5 = conv_block(ch_in=16*n_kernels, ch_out=8*n_kernels)

        self.Up4 = up_conv(ch_in=8*n_kernels,ch_out=4*n_kernels)
        self.Up_conv4 = conv_block(ch_in=8*n_kernels, ch_out=4*n_kernels)
        
        self.Up3 = up_conv(ch_in=4*n_kernels,ch_out=2*n_kernels)
        self.Up_conv3 = conv_block(ch_in=4*n_kernels, ch_out=2*n_kernels)
        
        self.Up2 = up_conv(ch_in=2*n_kernels,ch_out=n_kernels)
        self.Up_conv2 = conv_block(ch_in=2*n_kernels, ch_out=n_kernels)

        self.Conv_1x1 = nn.Conv2d(n_kernels,output_ch,kernel_size=1,stride=1,padding=0)

    def reset(self):
        for layer in self.children():
           if hasattr(layer, 'reset_parameters'):
               layer.reset_parameters()
    
    def forward(self, x):
        
        x1 = self.Conv1(x)
        
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        
        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)
        
        cls_out = self.classifiercaps(x3.view(x3.shape[0], x3.shape[1], -1))
        cls_out = cls_out.sum(dim=2)
        x5 = x5 * cls_out

        # d5 = self.Up5(x5)
        # d5 = torch.cat((x4,d5),dim=1)
        #
        # d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        d1 = d1 ** 2
        
        return d1, cls_out

class truth_network:
    def __init__(self):
        self.dgen = ...
    def eval(self):
        return
    def train(self):
        return
    def __call__(self, x, inds):
        _, label = self.dgen(inds)
        return torch.from_numpy(label).float().cuda()

if __name__ == '__main__':
    ...#U_Net