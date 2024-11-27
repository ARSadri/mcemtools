"""license
an adaptation of the Notebook released under the Apache 2.0 open source license: 
https://www.kaggle.com/code/abhigupta4981/capsule-net-for-mnist-using-pytorch/notebook
"""

import torch
import torch.nn.functional as F
import torch.nn as nn

class ConvLayer(nn.Module):
    def __init__(self, in_channels=1, out_channels=256,
                       kernel_size=9, stride=1, padding=0):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 
                              kernel_size=kernel_size, 
                              stride=stride, padding=padding)
    def forward(self, x):
        x = F.relu(self.conv(x))
        return x

class PrimaryCaps(nn.Module):
    def __init__(self, num_capsules=8, in_channels=256, 
                 out_channels=32, PrimaryCaps_n_pix = 6*6,
                 kernel_size=9, stride=2, padding=0):
        super(PrimaryCaps, self).__init__()
        self.out_channels = out_channels
        self.PrimaryCaps_n_pix = PrimaryCaps_n_pix
        self.capsules = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, 
                      kernel_size=kernel_size, stride=stride, padding=padding)
            for _ in range(num_capsules)
        ])
    def forward(self, x):
        batch_size = len(x)
        # I don't know what *6*6 is
        u = []
        for capsule in self.capsules:
            capsx = capsule(x)
            try:
                capsx = capsx.view(
                    batch_size, self.out_channels*self.PrimaryCaps_n_pix, 1)
            except Exception as e:
                print(
                    'capsnet: ',
                    f'capsx shape is {capsx.shape}, batch_size is {batch_size}, '
                    f'out_channels shape is {self.out_channels}, '
                    f'Argument named PrimaryCaps_n_pix is {self.PrimaryCaps_n_pix}',
                    'with the given image size you should have passed: ',
                    f'{capsx.shape[2]}*{capsx.shape[3]}', '\n', e, '\n', '-'*67)
                
                exit()
            u.append(capsx)
        
        u = torch.cat(u, dim=-1)
        u_squashed = self.squash(u)
        return u_squashed
    
    def squash(self, x):
        squared_norm = (x**2).sum(dim=-1, keepdim=True)
        scale = squared_norm/(1+squared_norm)
        output = scale * x/torch.sqrt(squared_norm)
        return output
    
def softmax(x, dim=1):
    transposed_inp = x.transpose(dim, len(x.size())-1)
    softmaxed = F.softmax(transposed_inp.contiguous().view(-1, transposed_inp.size(-1)), dim=-1)
    return softmaxed.view(*transposed_inp.size()).transpose(dim, len(x.size())-1)

class SelfAttention(nn.Module):
    def __init__(self, in_channels, n_heads):
        super(SelfAttention, self).__init__()
        self.n_heads = n_heads
        # self.linear = nn.Linear(in_channels, in_channels * n_heads)

        self.linear = nn.Linear(in_channels, in_channels * n_heads)
        self.sig = nn.Sigmoid()

    def forward(self, x_hat):
        x_hat_shape = x_hat.shape
        x_prine = x_hat.swapaxes(2, 3).swapaxes(3, 4)
        x_prine = x_prine.reshape(-1, x_prine.shape[-1])
        cov = torch.matmul(x_prine.T, x_prine) / x_hat_shape[2]
        cov = self.linear(cov)
        attention_output = torch.matmul(x_prine, cov) / x_hat_shape[2]
        attention_output = attention_output.reshape(
            attention_output.shape[0], -1, self.n_heads)
        attention_output = attention_output.mean(dim=-1)

        attention_output = self.sig(attention_output)

        attention_output = attention_output.view(
            x_hat_shape[0], x_hat_shape[1], x_hat_shape[4], 1, -1)
        attention_output = attention_output.swapaxes(4, 3).swapaxes(3, 2).swapaxes(4, 3)
        return attention_output

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
    def __init__(self, num_classes=10, Prim_out_channels=32,
                 Classifier_in_channels=8, Classifier_out_channels=16, 
                 routing_iterations=3, PrimaryCaps_n_pix = 6*6, n_heads = 8):
        super(ClassifierCaps, self).__init__()
        self.routing_iterations = routing_iterations
        self.W = nn.Parameter(torch.randn(
            num_classes, Prim_out_channels*PrimaryCaps_n_pix, 
            Classifier_in_channels, Classifier_out_channels))
        self.attention = SelfAttention(Prim_out_channels*PrimaryCaps_n_pix, n_heads)
        
    def forward(self, x):
        x = x[None, :, :, None, :]
        W = self.W[:, None, :, :, :]
        x_hat = torch.matmul(x, W)
        
        if 1: # 1 with or 0 without dynamic routing
            v_j = dynamic_routing(
                x_hat, self.squash, routing_iterations=self.routing_iterations)
        else:
            v_j = self.attention(x_hat).mean(2, keepdim=True)
        
        return v_j

    def squash(self, x):
        squared_norm = (x**2).sum(dim=-1, keepdim=True)
        scale = squared_norm/(1+squared_norm)
        out = scale * x/torch.sqrt(squared_norm)
        return out

class Decoder(nn.Module):
    def __init__(self, Decoder_input_vector_length=16, 
                       num_classes=10, Decoder_hidden_dim=512,
                       image_n_pix=28*28, output_channels=3):
        super(Decoder, self).__init__()
        self.num_classes = num_classes
        self.output_channels = output_channels  # Number of output channels
        self.image_n_pix = image_n_pix
        input_dim = Decoder_input_vector_length * num_classes
        self.lin_layers = nn.Sequential(
            nn.Linear(input_dim, Decoder_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(Decoder_hidden_dim, Decoder_hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(Decoder_hidden_dim * 2, image_n_pix * output_channels),  # Adjusted for multiple channels
            nn.Sigmoid()  # Output is between 0 and 1
        )
    
    def forward(self, x):
        classes = (x**2).sum(dim=-1)**0.5
        classes = F.softmax(classes, dim=-1)
        _, max_length_indices = classes.max(dim=1)
        
        sparse_matrix = torch.eye(self.num_classes).cuda()
        y = sparse_matrix.index_select(dim=0, index=max_length_indices.data)
        x = x * y[:, :, None]  # Element-wise multiplication
        
        flattened_x = x.reshape(x.size(0), -1)
        reconstructed = self.lin_layers(flattened_x)
        
        # Reshaping the output to have the required number of channels
        reconstructed = reconstructed.reshape(-1, self.output_channels, self.image_n_pix)  # Example for 28x28 image
        
        return reconstructed, y

class CapsuleNetwork(nn.Module):
    def __init__(self, 
                 num_classes = 52, 
                 in_channels=8,
                 image_n_pix = 24*24,
                 PrimaryCaps_n_pix = 4*4,
                 Enc_out_channels=256, 
                 Prim_out_channels = 32,
                 Prim_n_capsules = 8,
                 Classifier_out_channels = 16,
                 Decoder_hidden_dim = 512):
        super(CapsuleNetwork, self).__init__()
        self.image_n_pix = image_n_pix
        self.conv_layer = ConvLayer(in_channels = in_channels,
                                    out_channels = Enc_out_channels)
        self.primary_capsule = PrimaryCaps(
            in_channels = Enc_out_channels,
            out_channels = Prim_out_channels,
            num_capsules = Prim_n_capsules,
            PrimaryCaps_n_pix = PrimaryCaps_n_pix)
        
        self.Classifier_capsule = ClassifierCaps(
            num_classes = num_classes,
            Prim_out_channels = Prim_out_channels,
            PrimaryCaps_n_pix = PrimaryCaps_n_pix,
            Classifier_in_channels=Prim_n_capsules, 
            Classifier_out_channels=Classifier_out_channels,
            )
        self.decoder = Decoder(
            num_classes = num_classes,
            Decoder_input_vector_length = Classifier_out_channels,
            Decoder_hidden_dim = Decoder_hidden_dim,
            image_n_pix = image_n_pix,
            output_channels = in_channels)
        
    def forward(self, x):
        assert x.shape[2]*x.shape[3] == self.image_n_pix, \
            f'capsnet: You said images have {self.image_n_pix} pixels, ' \
            + f' but they actually have {x.shape[2]*x.shape[3]}.'
        features = self.conv_layer(x)
        primary_caps_out = self.primary_capsule(features)
        caps_out = self.Classifier_capsule(
            primary_caps_out).squeeze().transpose(0, 1)
        reconstructed, y = self.decoder(caps_out)
        return caps_out, reconstructed, y

class CapsuleLoss(nn.Module):
    def __init__(self, data_gen, recon_class_weight = 100):
        super(CapsuleLoss, self).__init__()
        self.reconstruction_loss = nn.MSELoss(size_average=False)
        self.data_gen = data_gen
        self.recon_class_weight = recon_class_weight
        self.n_calls = 0
    
    def forward(self, preds, labels, inds):#x, labels, images, reconstructions):
        self.n_calls += 1
        x, reconstructions, y = preds
        batch_size = len(x)
        v_c = torch.sqrt((x**2).sum(dim=2, keepdim=True))
        left = F.relu(0.9-v_c).view(batch_size, -1)
        right = F.relu(v_c-0.1).view(batch_size, -1)
        margin_loss_all = labels * left + 0.5 * (1.-labels) * right
        margin_loss_all = self.recon_class_weight * margin_loss_all
        margin_loss = margin_loss_all.mean()

        images = self.data_gen(inds)[0]
        try:
            images = torch.from_numpy(images).float().cuda()
        except: pass
        # images = images.sum(1)
        images = images.view(reconstructions.shape[0], reconstructions.shape[1], -1)
        
        reconstruction_loss_all = []
        for reconstructions_, images_ in zip(reconstructions, images):
            reconstruction_loss_all.append(self.reconstruction_loss(reconstructions_, images_))
        reconstruction_loss_all = torch.tensor(reconstruction_loss_all)

        reconstruction_loss = self.reconstruction_loss(reconstructions, images) / batch_size

        loss =  margin_loss + reconstruction_loss
        return (loss,
                margin_loss_all.detach(), reconstruction_loss_all.detach())

from mcemtools.denoise.LossFunc import mseLoss

if __name__ == '__main__':
    capsule_net = CapsuleNetwork(
        num_classes = 40, 
        in_channels=8,
        image_n_pix = 48*48,
        PrimaryCaps_n_pix = 16*16,
        Enc_out_channels=256, 
        Prim_out_channels = 32,
        Prim_n_capsules = 8,
        Classifier_out_channels = 16,
        Decoder_hidden_dim = 512).cuda().float()
    loss = mseLoss().cuda()
    capsule_net.train()
    TRAIN_ON_GPU = torch.cuda.is_available()
    if TRAIN_ON_GPU: print('training on gpu')
    print(capsule_net)
    print('-'*30)
    import time
    data = torch.rand(512, 8, 8, 48, 48).float().cuda()
    label = torch.rand(512, 8, 40).float().cuda()
    print(data.shape)
    time_time = time.time()
    for data_, label_ in zip(data, label): 
        caps_out, reconstructed, y = capsule_net(data_)
        # loss(y, label_)
    print(1000*(time.time() - time_time)/data.shape[0]/data.shape[1])
    
    
    