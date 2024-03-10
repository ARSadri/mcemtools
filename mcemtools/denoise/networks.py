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

class MLP(nn.Module):
    def __init__(self,n_neurons_list = [20000, 2000, 200, 20, 1]):
        super(MLP,self).__init__()
        self.fc1 = nn.Linear(2, n_neurons_list[0])
        self.ReLU1 = nn.ReLU()
        self.fc2 = nn.Linear(n_neurons_list[0], n_neurons_list[1])
        self.ReLU2 = nn.ReLU()
        self.fc3 = nn.Linear(n_neurons_list[1], n_neurons_list[2])
        self.ReLU3 = nn.ReLU()
        self.fc4 = nn.Linear(n_neurons_list[2], n_neurons_list[3])
        self.ReLU4 = nn.ReLU()
        self.fc5 = nn.Linear(n_neurons_list[3], n_neurons_list[4])
    
    def forward(self, x0):
        x1_fc = self.fc1(x0)
        x1_out = self.ReLU1(x1_fc)
        
        x2_fc = self.fc2(x1_out)
        x2_out = self.ReLU2(x2_fc)
        
        x3_fc = self.fc3(x2_out)
        x3_out = self.ReLU3(x3_fc)
        
        x4_fc = self.fc4(x3_out)
        x4_out = self.ReLU4(x4_fc)
        
        x5_fc = self.fc5(x4_out)
        
        return x5_fc

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

class Recurrent_block(nn.Module):
    def __init__(self,ch_out,t=2):
        super(Recurrent_block,self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        for i in range(self.t):

            if i==0:
                x1 = self.conv(x)
            
            x1 = self.conv(x+x1)
        return x1
        
class RRCNN_block(nn.Module):
    def __init__(self,ch_in,ch_out,t=2):
        super(RRCNN_block,self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out,t=t),
            Recurrent_block(ch_out,t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x+x1

class single_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(single_conv,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class U_Net(nn.Module):
    def __init__(self, img_ch=1, output_ch=1, n_kernels = 64, mask = None):
        super(U_Net,self).__init__()
        
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

        self.mask = mask
        self.mu_eaxct = False
        self.mu = None
        self.PACBED = None
        
    def forward(self,x, inds):
        
        x1 = self.Conv1(x)
        
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)
        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
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
        
        for dim in range(d1.shape[0]):
            if(self.PACBED is not None):
                d1[dim] *= self.PACBED
            if(self.mu_eaxct):
                d1[dim] /= d1[dim].sum()
                d1[dim] *= self.mu_eaxct[inds[dim]]
            if(self.mu is not None):
                d1[dim] *= self.mu[inds[dim]]

        if(self.mask is not None):
            d1[:, :, self.mask==0] = 0 
        
        return d1
        

class U_Net_fieldImage(nn.Module):
    def __init__(self, img_ch=1, output_ch=1, n_kernels = 64, 
                 Ne_input = 1, Ne_output = 1):
        super(U_Net_fieldImage,self).__init__()
        
        assert Ne_output == 1, 'You cannot set the Ne_output at initialization'\
            + ' because you have to first infer all data points then decide ' \
            + ' what this factor should be.'
        
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
        self.Ne_input = Ne_input
        self.Ne_output = 1
        
    def forward(self,x):
        x /= self.Ne_input
        
        x1 = self.Conv1(x)
        
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)
        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        
        d1 *= self.Ne_output
        return d1

class R2U_Net(nn.Module):
    def __init__(self,img_ch=3,output_ch=1,t=2):
        super(R2U_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_ch,ch_out=64,t=t)

        self.RRCNN2 = RRCNN_block(ch_in=64,ch_out=128,t=t)
        
        self.RRCNN3 = RRCNN_block(ch_in=128,ch_out=256,t=t)
        
        self.RRCNN4 = RRCNN_block(ch_in=256,ch_out=512,t=t)
        
        self.RRCNN5 = RRCNN_block(ch_in=512,ch_out=1024,t=t)

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512,t=t)
        
        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256,t=t)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128,t=t)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64,t=t)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.Up_RRCNN5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)

        return d1

class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.relu = nn.ReLU(inplace=True)

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi

class AttU_Net(nn.Module):
    def __init__(self,img_ch=1, output_ch=1, n_kernels = 64, mask = None, mu = None):
        super(AttU_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=n_kernels)
        self.Conv2 = conv_block(ch_in=n_kernels,ch_out=n_kernels*2)
        self.Conv3 = conv_block(ch_in=n_kernels*2,ch_out=n_kernels*4)
        self.Conv4 = conv_block(ch_in=n_kernels*4,ch_out=n_kernels*8)
        self.Conv5 = conv_block(ch_in=n_kernels*8,ch_out=n_kernels*16)

        self.Up5 = up_conv(ch_in=n_kernels*16,ch_out=n_kernels*8)
        self.Att5 = Attention_block(F_g=n_kernels*8,F_l=n_kernels*8,F_int=n_kernels*4)
        self.Up_conv5 = conv_block(ch_in=n_kernels*16, ch_out=n_kernels*8)

        self.Up4 = up_conv(ch_in=n_kernels*8,ch_out=n_kernels*4)
        self.Att4 = Attention_block(F_g=n_kernels*4,F_l=n_kernels*4,F_int=n_kernels*2)
        self.Up_conv4 = conv_block(ch_in=n_kernels*8, ch_out=n_kernels*4)
        
        self.Up3 = up_conv(ch_in=n_kernels*4,ch_out=n_kernels*2)
        self.Att3 = Attention_block(F_g=n_kernels*2,F_l=n_kernels*2,F_int=n_kernels)
        self.Up_conv3 = conv_block(ch_in=n_kernels*4, ch_out=n_kernels*2)
        
        self.Up2 = up_conv(ch_in=n_kernels*2,ch_out=n_kernels)
        self.Att2 = Attention_block(F_g=n_kernels,F_l=n_kernels,F_int=int(n_kernels/2))
        self.Up_conv2 = conv_block(ch_in=n_kernels*2, ch_out=n_kernels)

        self.Conv_1x1 = nn.Conv2d(n_kernels,output_ch,kernel_size=1,stride=1,padding=0)
       
        self.mask = mask
        self.mu = mu
        self.PACBED = None

    def forward(self,x, inds):
        # encoding path
        x1 = self.Conv1(x)
        
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        
        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)

        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)
        
        d1 = self.Conv_1x1(d2)

        if(self.mask is not None):
            d1[:, :, self.mask==0] = 0 
        d1 = d1 ** 2
        # d1 = torch.relu(d1)
        for dim in range(d1.shape[0]):
            if(self.PACBED is not None):
                d1[dim] *= self.PACBED
            if(self.mu is not None):
                d1[dim] *= self.mu[inds[dim]]          
        
        return d1

class R2AttU_Net(nn.Module):
    def __init__(self,img_ch=3,output_ch=1,t=2):
        super(R2AttU_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_ch,ch_out=64,t=t)

        self.RRCNN2 = RRCNN_block(ch_in=64,ch_out=128,t=t)
        
        self.RRCNN3 = RRCNN_block(ch_in=128,ch_out=256,t=t)
        
        self.RRCNN4 = RRCNN_block(ch_in=256,ch_out=512,t=t)
        
        self.RRCNN5 = RRCNN_block(ch_in=512,ch_out=1024,t=t)
        

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Att5 = Attention_block(F_g=512,F_l=512,F_int=256)
        self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512,t=t)
        
        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Att4 = Attention_block(F_g=256,F_l=256,F_int=128)
        self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256,t=t)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Att3 = Attention_block(F_g=128,F_l=128,F_int=64)
        self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128,t=t)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Att2 = Attention_block(F_g=64,F_l=64,F_int=32)
        self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64,t=t)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.Up_RRCNN5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)

        return d1

class ResNetBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride = 1, kernel_size = 3, padding = 1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        #self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        #self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.LeakyReLU()
        self.relu2 = nn.LeakyReLU()

        '''
        self.identity_downsample = 
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 
                                    kernel_size=1, stride=stride, padding=0),
                          nn.BatchNorm2d(out_channels))
        '''
        self.identity_downsample = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=1, 
            stride=stride, 
            padding=0)
        
    def forward(self, x):
        identity = self.identity_downsample(x)
        
        x = self.conv1(x)
        #x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        #x = self.bn2(x)
        
        x += identity

        x = self.relu2(x)
        return(x)

class ResNet18(nn.Module): #[2,2,2,2]
    
    def __init__(self, 
                 layers = np.array([1, 1, 1]),
                 image_channels = 1, 
                 num_classes = 1,
                 mid_channels = 32):
        super(ResNet18, self).__init__()
        self.mid_channels = mid_channels
        
        self.conv1 = nn.Conv2d(
            image_channels, self.mid_channels, kernel_size=3, stride=1)
        #self.bn = nn.BatchNorm2d(self.mid_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        
        #ResNet18 layers
        self.layer1 = self._make_layer(
            layers[0], layer_out_channels = 1*self.mid_channels, stride=1) 
        self.layer2 = self._make_layer(
            layers[1], layer_out_channels = 2*self.mid_channels, stride=1)
        self.layer3 = self._make_layer(
            layers[2], layer_out_channels = 2*self.mid_channels, stride=1)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(self.mid_channels, num_classes)
        
    def _make_layer(self, num_residual_blocks, layer_out_channels, stride):
        layers = []
        layers.append(ResNetBlock(self.mid_channels, 
                                  layer_out_channels, stride=stride))
        self.mid_channels = layer_out_channels
        for i in range(num_residual_blocks - 1 ):
            layers.append(ResNetBlock(self.mid_channels, 
                                      layer_out_channels, stride=stride))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        #x = self.bn(x) ##############################################################
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x) 
        x = self.layer2(x) 
        x = self.layer3(x) 
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return(x)

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query):
        # Get number of training examples
        N = query.shape[0]

        value_len, key_len, query_len = \
            values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        query = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)  # (N, value_len, heads, head_dim)
        keys = self.keys(keys)  # (N, key_len, heads, head_dim)
        queries = self.queries(query)  # (N, query_len, heads, heads_dim)

        # Einsum does matrix mult. for query*keys for each training example
        # with every other training example, don't be confused by einsum
        # it's just how I like doing matrix multiplication & bmm

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # queries shape: (N, query_len, heads, heads_dim),
        # keys shape: (N, key_len, heads, heads_dim)
        # energy: (N, heads, query_len, key_len)

        # Normalize energy values similarly to seq2seq + attention
        # so that they sum to 1. Also divide by scaling factor for
        # better stability
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        # attention shape: (N, heads, query_len, key_len)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)
        # out after matrix multiply: (N, query_len, heads, head_dim), then
        # we reshape and flatten the last two dimensions.

        out = self.fc_out(out)
        # Linear layer doesn't modify the shape, final shape will be
        # (N, query_len, embed_size)

        return out

class surfuce_fitter(nn.Module):
    def __init__(self, 
                 img_ch=1, output_ch=1, num_classes = 1,
                 n_ch=[8, 16, 32, 64, 128, 256],
                 embed_size = None, heads = None):
        super(surfuce_fitter, self).__init__()
        self.embed_size = embed_size

        if(self.embed_size is not None):
            self.attention = SelfAttention(embed_size = embed_size, heads = heads)
        
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.Maxpool5 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.Maxpool6 = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = ResNetBlock(in_channels=img_ch ,out_channels=n_ch[0])

        self.Conv2 = ResNetBlock(in_channels=n_ch[0],out_channels=n_ch[1])
        self.Conv3 = ResNetBlock(in_channels=n_ch[1],out_channels=n_ch[2])
        self.Conv4 = ResNetBlock(in_channels=n_ch[2],out_channels=n_ch[3])
        self.Conv5 = ResNetBlock(in_channels=n_ch[3],out_channels=n_ch[4])
        self.Conv6 = ResNetBlock(in_channels=n_ch[4],out_channels=n_ch[5])

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(n_ch[5], num_classes)

    def forward(self,x):
        # encoding path
        
        x1 = self.Conv1(x)

        x2 = self.Maxpool2(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool3(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool4(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool5(x4)
        x5 = self.Conv5(x5)

        x6 = self.Maxpool6(x5)
        x6 = self.Conv6(x6)

        if(self.embed_size is not None):
            xAtt = x6.reshape(x6.shape[0], x6.shape[1], x6.shape[2]*x6.shape[3])
            xAtt = self.attention(xAtt, xAtt, xAtt) + xAtt
            x6 = xAtt.reshape(\
                x6.shape[0], x6.shape[1], x6.shape[2], x6.shape[3])

        x7 = self.avgpool(x6)
        x7 = x7.reshape(x7.shape[0], -1)
        
        y = self.fc(x7)
        return(y)

def test_ResNet18():
    rnet18 = ResNet18(layers = np.array([2, 2, 2]), 
                      image_channels = 1, 
                      num_classes=10)
    x = torch.randn(2, 1, 64, 64)
    y = rnet18(x).to('cuda')
    print(y.shape)

class denoise4DSTEM_resNet(nn.Module):
    def __init__(self, 
                 img_ch=1, output_ch=1,
                 n_ch_list=[8, 16, 16, 32, 32, 64, 64],
                 embed_size = None, heads = None,
                 mask = None):
        super(denoise4DSTEM_resNet, self).__init__()
        self.embed_size = embed_size

        if(self.embed_size is not None):
            self.attention = SelfAttention(embed_size = embed_size, heads = heads)
        
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2,stride=1)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2,stride=1)
        self.Maxpool5 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.Maxpool6 = nn.MaxPool2d(kernel_size=2,stride=1)
        self.Maxpool7 = nn.MaxPool2d(kernel_size=3,stride=2)

        self.Conv1 = ResNetBlock(in_channels=img_ch ,out_channels=n_ch_list[0])

        self.Conv2 = ResNetBlock(in_channels=n_ch_list[0],out_channels=n_ch_list[1])
        self.Conv3 = ResNetBlock(in_channels=n_ch_list[1],out_channels=n_ch_list[2])
        self.Conv4 = ResNetBlock(in_channels=n_ch_list[2],out_channels=n_ch_list[3])
        self.Conv5 = ResNetBlock(in_channels=n_ch_list[3],out_channels=n_ch_list[4])
        self.Conv6 = ResNetBlock(in_channels=n_ch_list[4],out_channels=n_ch_list[5])
        self.Conv7 = ResNetBlock(in_channels=n_ch_list[5],out_channels=n_ch_list[6])
        
        self.mask = mask
        
    def forward(self,x):
        # encoding path
        
        x1 = self.Conv1(x)
        # print(f'Conv1:{x1.shape}')
        
        x2 = self.Maxpool2(x1)
        # print(f'Maxpool2:{x2.shape}')        
        x2 = self.Conv2(x2)
        # print(f'Conv2:{x2.shape}')        
        
        x3 = self.Maxpool3(x2)
        # print(f'Maxpool3:{x3.shape}')        
        x3 = self.Conv3(x3)
        # print(f'Conv3:{x3.shape}')        

        x4 = self.Maxpool4(x3)
        # print(f'Maxpool4:{x4.shape}')        
        x4 = self.Conv4(x4)
        # print(f'Conv4:{x4.shape}')        

        x5 = self.Maxpool5(x4)
        # print(f'Maxpool5:{x5.shape}')        
        x5 = self.Conv5(x5)
        # print(f'Conv5:{x5.shape}')        

        x6 = self.Maxpool6(x5)
        # print(f'Maxpool6:{x6.shape}')        
        x6 = self.Conv6(x6)
        # print(f'Maxpool6:{x6.shape}')        

        x7 = self.Maxpool7(x6)
        # print(f'Maxpool7:{x7.shape}')        
        x7 = self.Conv7(x7)
        # print(f'Maxpool7:{x7.shape}')        

        if(self.embed_size is not None):
            xAtt = x7.reshape(x7.shape[0], x7.shape[1], x7.shape[2]*x7.shape[3])
            xAtt = self.attention(xAtt, xAtt, xAtt) + xAtt
            x7 = xAtt.reshape(\
                x7.shape[0], x7.shape[1], x7.shape[2], x7.shape[3])

        x8 = 0*x7[:, :1, :, :]
        x8[:, 0] = x7.mean(1)
        if(self.mask is not None):
            x8[:, :, self.mask[9:-9, 9:-9]==0] = 0
        return(x8)

def test_denoise4DSTEM_resNet():
    denoiser_model = denoise4DSTEM_resNet().to('cuda')
    x = torch.randn(2, 1, 7*128, 7*128).to('cuda')
    y = denoiser_model(x)
    print(f'{x.shape} --> NN --> {y.shape} ')
    
if __name__ == '__main__':
    test_denoise4DSTEM_resNet()