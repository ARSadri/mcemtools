import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FocalLossV1(nn.Module):
    def __init__(self,
                 alpha=0.25,
                 gamma=2,
                 reduction='mean',):
        super(FocalLossV1, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.crit = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, label):
        '''
        logits and label have same shape, and label data type is long
        args:
            logits: tensor of shape (N, ...)
            label: tensor of shape(N, ...)
        '''

        # compute loss
        logits = logits.float() # use fp32 if logits is fp16
        with torch.no_grad():
            alpha = torch.empty_like(logits).fill_(1 - self.alpha)
            alpha[label == 1] = self.alpha

        probs = torch.sigmoid(logits)
        pt = torch.where(label == 1, probs, 1 - probs)
        ce_loss = self.crit(logits, label.float())
        loss = (alpha * torch.pow(1 - pt, self.gamma) * ce_loss)
        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss

#PyTorch
# class DiceLoss(nn.Module):
#     def __init__(self, weight=None, size_average=True):
#         super(DiceLoss, self).__init__()
# 
#     def forward(self, inputs, targets, smooth=1):
#         
#         #comment out if your model contains a sigmoid or equivalent activation layer
#         #inputs = F.sigmoid(inputs)       
#         
#         #flatten label and prediction tensors
#         inputs = inputs.view(-1)
#         targets = targets.view(-1)
#         
#         intersection = (inputs * targets).sum()                            
#         dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
#         
#         return 1 - dice
    
    
class dice_loss(nn.Module):
    def __init__(self):
        super(dice_loss, self).__init__()
        
        
    def forward(self, pred, target):    
        # if(len(pred.shape)==3):
        #     pred = pred.unsqueeze(0)
        #     target = target.unsqueeze(0)   
        smooth = 1 
        # try:
            # intersection = (pred * target).sum(dim=2).sum(dim=2)
        intersection = (pred * target).sum()
        # except:
            # pass
        # loss = (1 - ((2 * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
        loss = (1 - ((2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)))
    
        return loss.mean()    
    

class diffraLossSq_perImage(nn.Module):
    def __init__(self):
        super(diffraLossSq_perImage, self).__init__()

    def forward(self, inputs, targets):
                
        #flatten label and prediction tensors
        #return(((((targets - inputs)**2).mean(0))**0.5).mean())
        return((((targets - inputs)**2).mean(1).mean(1).mean(1))**0.5)

class STEM4DLoss2(nn.Module):
    def __init__(self, alpha, beta = None, noneg_coeff = 100):
        super(STEM4DLoss2, self).__init__()
        self.alpha = alpha
        self.beta = beta
        if(self.beta is None):
            self.beta = self.alpha/10
        self.noneg_coeff = noneg_coeff

    def forward(self, x, targets):
        L_x = 1/1 * (      F.relu(x - 0 * self.alpha, inplace = True)    \
                     - 2 * F.relu(x - 1 * self.alpha, inplace = True)    \
                         + F.relu(x - 2 * self.alpha, inplace = True))   \
            + 1/2 * (      F.relu(x - 3 * self.alpha, inplace = True)    \
                     - 2 * F.relu(x - 4 * self.alpha, inplace = True)    \
                         + F.relu(x - 5 * self.alpha, inplace = True))   \
            + 1/4 * (      F.relu(x - 6 * self.alpha, inplace = True)    \
                     - 2 * F.relu(x - 7 * self.alpha, inplace = True)    \
                         + F.relu(x - 8 * self.alpha, inplace = True))
        
        Poissonian_loss = self.beta \
            * (L_x + self.noneg_coeff * F.relu(-x, inplace = True))
        res = (targets - x*16)**2
        res += Poissonian_loss**2
        mse_loss_ = (res.mean())**0.5
        
        return(mse_loss_)

class STEM4DLoss(nn.Module):
    def __init__(self, LAG_MUL = 0, print_norms = False):
        super(STEM4DLoss, self).__init__()
        self.LAG_MUL = LAG_MUL
        self.print_norms = print_norms
        
    def forward(self, y_out, x_in):
        err = y_out - x_in
        res = (err**2).mean(3).mean(2).mean(1)
        # reg = (err.mean(3).mean(2).mean(1))**2

        res_norm = (res.mean(0))**0.5
        # reg_norm = (reg.mean(0))**0.5
        
        # if(self.print_norms):
        #     print(f's:{res_norm:.6f}, g:{reg_norm:.6f}')
        
        return(res_norm)# + self.LAG_MUL*reg_norm)


class STEM4D_PoissonLoss_sym(nn.Module):
    def __init__(self, mask_backprop):
        super(STEM4D_PoissonLoss_sym, self).__init__()
        self.mask_backprop = mask_backprop
        self.mask_backprop_sum = mask_backprop.sum()
        
    def forward(self, y_out, x_in):
        x_in = x_in[:, :, self.mask_backprop==1]
        y_out = y_out[:, :, self.mask_backprop==1]
        
        err1 = x_in[y_out >  x_in] * torch.log(                          y_out[y_out >  x_in] + 1e-8) -   y_out[y_out >  x_in]
        err2 = x_in[y_out <= x_in] * torch.log(2 * x_in[y_out <= x_in] - y_out[y_out <= x_in] + 1e-8) - (-y_out[y_out <= x_in])

        # err1 = x_in[y_out <= x_in] * torch.log(                          y_out[y_out <= x_in] + 1e-8) -   y_out[y_out <= x_in]
        # err2 = x_in[y_out >  x_in] * torch.log(2 * x_in[y_out >  x_in] - y_out[y_out >  x_in] + 1e-8) - (-y_out[y_out >  x_in])
        
        res = - (err1.sum() + err2.sum()) / self.mask_backprop_sum
        return(res)

class STEM4D_PoissonLoss(nn.Module):
    def __init__(self, mask_backprop):
        super(STEM4D_PoissonLoss, self).__init__()
        self.mask_backprop = mask_backprop
        self.mask_backprop_sum = mask_backprop.sum()
        
    def forward(self, y_out, x_in):
        with torch.no_grad():
            n_images = x_in.shape[0]
        err = x_in * torch.log(y_out + 1e-8) - y_out
        err[:, :, self.mask_backprop == 0] = 0
        res = - err.sum() / self.mask_backprop_sum / n_images
        
        return(res)
    
class STEM4D_PoissonLoss_pow2(nn.Module):
    def __init__(self, mask_backprop):
        super(STEM4D_PoissonLoss_pow2, self).__init__()
        self.mask_backprop = mask_backprop
        self.mask_backprop_sum = mask_backprop.sum()
        
    def forward(self, y_out, x_in):
        err = 2 * x_in**2 * torch.log(y_out + 1e-8) - y_out**2
        err[:, :, self.mask_backprop == 0] = 0
        res = - err.sum() / self.mask_backprop_sum
        
        return(res)

class STEM4D_PoissonLoss_FnormLoss(nn.Module):
    def __init__(self, mask_backprop, LAGMUL = 1, 
                 x_in_max = 1000, device = 'cuda',
                 output_stabilizer = 1e-6,
                 noisy_PACBED = None,
                 noisy_mSTEM = None,
                 PAC_loss_factor = 0.01,
                 mSTEM_loss_factor = 0.01,
                 CoNs = None):
        
        super(STEM4D_PoissonLoss_FnormLoss, self).__init__()
        self.LAGMUL = LAGMUL
        self.mask_backprop = mask_backprop
        self.mask_backprop_sum = mask_backprop.sum()
        
        self.output_stabilizer = output_stabilizer
        
        self.noisy_PACBED = noisy_PACBED
        self.accumulated_PACBED = 0*noisy_PACBED
        self.accumulated_n_images = 0
        self.PAC_loss_factor = PAC_loss_factor

        self.noisy_mSTEM = noisy_mSTEM
        self.accumulated_mSTEM = 0*noisy_mSTEM
        self.accumulated_inds = 0*self.accumulated_mSTEM
        self.mSTEM_loss_factor = mSTEM_loss_factor
        
        self.accumulated_PACBED.requires_grad = False
        self.accumulated_mSTEM.requires_grad = False
        self.accumulated_inds.requires_grad = False
        
        self.CoMs = CoNs

    def forward(self, y_out, x_in, inds):
        n_images = x_in.shape[0]
        log_factorial_x_in = torch.from_numpy(np.cumsum(np.log((x_in + 
            self.output_stabilizer).cpu().numpy())).reshape(x_in.shape)).cuda()

        log_y_out = torch.log(y_out.float() + self.output_stabilizer)
        err_p = x_in * log_y_out - y_out - log_factorial_x_in
        err_p[:, :, self.mask_backprop == 0] = 0
        res_p = - err_p.sum() / self.mask_backprop_sum / n_images
        
        err_f = (x_in - y_out)**2 
        err_f[:, :, self.mask_backprop == 0] = 0
        res_f = (err_f.sum() / self.mask_backprop_sum / n_images)**0.5
        
        res = self.LAGMUL * res_f + (1 - self.LAGMUL)*res_p
        
        if self.CoMs is None:
            res_CoMs = 0
        else:
            res_CoMs = 0
            
        if self.noisy_PACBED is None:
            res_PAC = 0
        else:
            new_element_pac = y_out[:, 0].sum(0)
            with torch.no_grad():
                accumulated_PACBED = self.accumulated_PACBED + 1
                accumulated_n_images = self.accumulated_n_images + 1
                self.accumulated_n_images += n_images
                self.accumulated_PACBED += new_element_pac
                # err_PAC = (y_out[:, 0].sum(0) / n_images - self.noisy_PACBED)**2
            
            err_PAC = ( (accumulated_PACBED - 1 + new_element_pac
                        )/ (accumulated_n_images - 1 + n_images) - self.noisy_PACBED)**2
            err_PAC[self.mask_backprop == 0] = 0
            res_PAC = (err_PAC.sum() / self.mask_backprop_sum )**0.5

        if self.noisy_mSTEM is None:
            res_mSTEM = 0
        else:
            new_element_stem = y_out.sum((1,2,3)) / self.mask_backprop_sum
            
            with torch.no_grad():
                accumulated_mSTEM = self.accumulated_mSTEM + 1
                self.accumulated_mSTEM[inds] += new_element_stem
                self.accumulated_inds[inds] += 1
            
            accumulated_mSTEM[inds] += new_element_stem -1
            accumulated_mSTEM[self.accumulated_inds > 0] /= \
                self.accumulated_inds[self.accumulated_inds>0]
            
            # err_PAC = (y_out[:, 0].sum(0) / n_images - self.noisy_PACBED)**2
            err_mSTEM = (accumulated_mSTEM - self.noisy_mSTEM)**2
            res_mSTEM = err_mSTEM.mean()**0.5
        
        # res = (1 - self.PAC_loss_factor)*res + self.PAC_loss_factor*res_PAC
        res = (1 - self.mSTEM_loss_factor - self.PAC_loss_factor)*res + \
               self.PAC_loss_factor*res_PAC + self.mSTEM_loss_factor*res_mSTEM        
        
        return(res)
    
class STEM4D_PoissonLoss_MeanScaleLoss(nn.Module):
    def __init__(self, LAGMUL = 0.5):
        super(STEM4D_PoissonLoss_FnormLoss, self).__init__()
        self.LAGMUL = LAGMUL
        
    def forward(self, y_out, x_in):
        
        labels = x_in[:x_in.shape[0]//2]
        mus = x_in[x_in.shape[0]//2:]
        
        err_p = x_in * torch.log(y_out + 1e-8) - y_out
        res_p = - err_p.mean()
        
        err_f = (x_in - y_out)**2 
        res_f = (err_f.mean())**0.5
        
        res = (1 - self.LAGMUL)*res_p + self.LAGMUL * res_f
        
        return(res)

class STEM4DLoss_with_mask_relu(nn.Module):
    def __init__(self, alpha = 1, mask = None):
        super(STEM4DLoss, self).__init__()
        self.alpha = alpha
        self.mask = mask
        
    def forward(self, x, y):
        res = y - x
        if(self.mask is None):
            loss = (res**2).mean()
        else:
            if(len(res.shape) == 4):
                loss = (res[:, :, self.mask>0]**2).mean()
            elif(len(res.shape) == 3):
                loss = (res[:, self.mask>0]**2).mean()
            elif(len(res.shape) == 2):
                loss = (res[self.mask>0]**2).mean()
        return(loss**0.5)

    def forward_pow2(self, x, y):
        res = y - x + self.alpha * F.relu(-x, inplace = True)
        if(self.mask is None):
            loss = (res**2).mean()
        else:
            if(len(res.shape) == 4):
                loss = (res[:, :, self.mask>0]**2).mean()
            elif(len(res.shape) == 3):
                loss = (res[:, self.mask>0]**2).mean()
            elif(len(res.shape) == 2):
                loss = (res[self.mask>0]**2).mean()
        return(loss**0.5)

class mseLoss(nn.Module):
    def __init__(self):
        super(mseLoss, self).__init__()

    def forward(self, inputs, targets, inds = None):
        return (((targets - inputs)**2).mean())**0.5

class CoMLoss(nn.Module):
    def __init__(self):
        super(CoMLoss, self).__init__()

    def forward(self, inputs, targets, inds = None):
        rec_error = (((targets - inputs)**2).mean())**0.5
        CoM_error = (targets.mean())**2 * (inputs.mean())**2
        return 0.5*rec_error + 0.5*CoM_error

class justLoss(nn.Module):
    def __init__(self):
        super(justLoss, self).__init__()

    def forward(self, inputs, targets):
        return((targets - inputs).mean())

def calc_loss(pred, target, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)

    pred = F.sigmoid(pred)
    dice = dice_loss(pred, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    #metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    #metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    #metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss