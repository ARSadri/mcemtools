import time
import torch
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from lognflow.plt_utils import plt_colorbar
from lognflow import lognflow, printprogress

import mcemtools

from .DATOS         import DATOS, nn_from_torch
from .networks      import U_Net as network4D
from .networks      import U_Net_fieldImage as network2D
# from .networks_3 import DcUnet as network4D
from .LossFunc      import mseLoss, STEM4D_PoissonLoss_FnormLoss 

# torch.autograd.set_detect_anomaly(True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def infer_STEM(
    torch_handler, data_gen_STEM, logger, infer_size = 1):
    
    data2D_shape = (data_gen_STEM.n_r, data_gen_STEM.n_c)
    _, predStat_list, output_vals = \
        torch_handler.infer(
            indices = np.arange(data_gen_STEM.n_pts, dtype='int'),
            return_predictions = True, infer_size = infer_size,
            predictions_statfunc_list = [data_gen_STEM.dist2Truth,
                                         data_gen_STEM.dist2label])
    
    fitting_errors = predStat_list[1]
    output_vals_vis = data_gen_STEM.reconstruct(output_vals[:, 0])
    output_vals_vis = output_vals_vis.reshape(data2D_shape)
    return fitting_errors, output_vals_vis
    
def train_STEM(
    n_kSweeps, torch_handler, data_gen_STEM,
    logger, DATOS_sampler, trainable_inds = None, infer_size = 1,
    mask_rate = 0.9):
    
    if(trainable_inds is None):
        trainable_inds = np.arange(data_gen_STEM.n_pts , dtype='int')
        assert trainable_inds.shape[0] == DATOS_sampler.N ,\
            'tarinable indices length must be the same as DATOS num of points'
    perv_loss = np.nan
    time_time = time.time()
    pBar = printprogress(n_kSweeps*DATOS_sampler.n_pos, 
                         print_function = None)
    for kcnt in range(n_kSweeps):
        
        # data_gen_STEM.randomize()
        
        with torch.no_grad():
            # torch_handler.torchModel.Ne_output = 1
            fitting_errors, output_values = infer_STEM(
                torch_handler, data_gen_STEM, logger, infer_size)
            logger.log_imshow(
                'STEM_denoiser/training/output_values', output_values)
            # torch_handler.torchModel.Ne_output = \
            #     data_gen_STEM.inimg_mean/output_values.mean()        
            logger.log_single(r'STEM_denoiser/model/Ne_input', 
                   float(torch_handler.torchModel.Ne_input), time_tag = False)
            logger.log_single(r'STEM_denoiser/model/Ne_output', 
                   float(torch_handler.torchModel.Ne_output), time_tag = False)
        DATOS_sampler.sort(fitting_errors[trainable_inds], order = 'descend')
        status = True
        ind_cnt = 0
        data_gen_STEM.mask_rate = mask_rate
        while(status):
            DATOS_inds, status = DATOS_sampler()
            if status:
                ETA = int(pBar(DATOS_inds.shape[0]))
                ind_cnt += DATOS_inds.shape[0]
                inds = trainable_inds[DATOS_inds].copy()
                loss = torch_handler.update(inds)
                logger.log_var('STEM_denoiser/training/training_loss', loss)
                if (time.time() - time_time > 30) :
                    time_vals, loss_vals = logger.get_var(
                        'STEM_denoiser/training/training_loss')
                    loss_change = loss_vals.mean() - perv_loss
                    perv_loss = loss_vals.mean()
                    logger(f'FI -> Sweep: {kcnt}/{n_kSweeps}'
                           f', ETA: {ETA}'
                           f', counter: {ind_cnt}/{DATOS_sampler.n_pos}'
                           f', loss: {loss_vals.mean():.6f}'
                           f', change: {loss_change:.6f}')
                    time_time = time.time()
        try:
            logger.log_plot('STEM_denoiser/training/losses', 
                            loss_vals, time_vals, time_tag = False)
        except:
            pass
    fpath = logger.log_single('STEM_denoiser/model/model', 
                      torch_handler.torchModel, suffix = 'torch',
                      time_tag = False)
    return fpath

def infer_I4D(
    torch_handler_I4D, data_gen_I4D, logger, infer_size_I4D = 1,
    use_trainable_inds = False):

    if use_trainable_inds:
        indices = data_gen_I4D.trainable_inds.copy()
    else:
        indices = np.arange(data_gen_I4D.n_pts, dtype='int')

    _, predStat_list, output_vals = \
        torch_handler_I4D.infer(
            indices = indices,
            return_predictions = True, infer_size = infer_size_I4D, show_progress = True,
            predictions_statfunc_list = [data_gen_I4D.dist2Truth,
                                         data_gen_I4D.dist2label])
    fitting_errors = predStat_list[1]
    dist2Truth_vals = predStat_list[0]
    
    if not use_trainable_inds:
        indices = None
    
    fitting_errors_vis = data_gen_I4D.reconstruct2D(fitting_errors, indices)
    logger.log_imshow('I4D_denoiser/train/fitting_errors', fitting_errors_vis)

    dist2Truth_vals_vis = data_gen_I4D.reconstruct2D(dist2Truth_vals, indices)
    logger.log_imshow('I4D_denoiser/train/dist2Truth_vals', dist2Truth_vals_vis)

    pvalue = output_vals[:25, 0
                         ].copy().swapaxes(0,1).swapaxes(1,2)
    logger.log_imshow_by_subplots(
        'I4D_denoiser/train/data4D_denoised', pvalue)
    
    _denoisedI4D_STEM = data_gen_I4D.reconstruct2D(
        output_vals[:, 0].sum(2).sum(1), indices)
    logger.log_imshow('I4D_denoiser/after_train_STEM/denoisedI4D_STEM', 
        _denoisedI4D_STEM)

    return fitting_errors, output_vals

def init_I4D(
        noisy_STEM,
        noisy_PACBED,
        logger,
        data_gen_I4D,
        torch_handler_I4D,
        infer_size_I4D):
    
    if(noisy_PACBED is not None):
        torch_handler_I4D.torchModel.PACBED = None
        _, pred_I4D = infer_I4D(torch_handler_I4D, data_gen_I4D, 
                                logger, infer_size_I4D)
    
        net_PACBED = pred_I4D.sum(0).squeeze()
        PACBED_eq = 0*net_PACBED.copy()
        PACBED_eq[net_PACBED>0] = \
            noisy_PACBED[net_PACBED>0]/net_PACBED[net_PACBED>0] 
        torch_handler_I4D.torchModel.PACBED = \
            torch.from_numpy(PACBED_eq).float().to(device)
    
    if(noisy_STEM is not None):
        torch_handler_I4D.torchModel.mu = None
        _, pred_I4D = infer_I4D(
            torch_handler_I4D, data_gen_I4D, logger, infer_size_I4D)
        net_mu = pred_I4D.sum((1, 2, 3)).squeeze()
        
        net_mu = data_gen_I4D.reconstruct2D(net_mu).ravel()
        mu = np.ones(net_mu.shape)
        mu[net_mu != 0] = (noisy_STEM.ravel())[net_mu != 0]/net_mu[net_mu != 0]
        
        # torch_handler_I4D.torchModel.mu_eaxct = noisy_STEM.ravel()
        torch_handler_I4D.torchModel.mu = \
            torch.from_numpy(mu).float().to(device)

def train_I4D(
        kcnt,
        logger,
        data_gen_I4D,
        DATOS_sampler_I4D,
        torch_handler_I4D,
        trainable_inds = None,
        perv_loss = None,
        infer_size_I4D = 1,
        ETA_base = 0):

    if(trainable_inds is None):
        trainable_inds = np.arange(data_gen_I4D.n_pts , dtype='int')
        assert trainable_inds.shape[0] == DATOS_sampler_I4D.N ,\
            'tarinable indices length must be the same as DATOS num of points'
    order = 'ascend'#'ascend'
    if((order == 'ascend') | (order == 'descend')):    
        fitting_errors, _ = infer_I4D(
            torch_handler_I4D, data_gen_I4D, logger, infer_size_I4D,
            use_trainable_inds = True)
        if perv_loss is None:
            perv_loss = fitting_errors.mean()
        DATOS_sampler_I4D.sort(fitting_errors, order = order)
    else:
        DATOS_sampler_I4D.sort(order = order)

    status = True
    time_time = time.time()
    DATOS_sampler_I4D_n_pos = DATOS_sampler_I4D.n_pos
    pBar = printprogress(DATOS_sampler_I4D.n_pos, print_function = None)
    while(status):
        DATOS_inds, status = DATOS_sampler_I4D()
        DATOS_prog = DATOS_sampler_I4D.next_in_line
        n_classes = DATOS_sampler_I4D.n_classes
        if status:
            ETA = int(pBar(DATOS_inds.shape[0] / n_classes)) + ETA_base
            inds = trainable_inds[DATOS_inds].copy()
            loss = torch_handler_I4D.update(inds)
            if np.isinf(loss):
                logger('inf' + '.'*50 + f' {inds}')
                raise ValueError
            if np.isnan(loss):
                logger('nan' + '.'*50 + f' {inds}')
                raise ValueError
            
            logger.log_var('I4D_denoiser/train/training_loss', loss)
            if (time.time() - time_time > 10) :
                time_vals, loss_vals = logger.get_var(
                    'I4D_denoiser/train/training_loss')
                loss_change = loss_vals.mean() - perv_loss
                perv_loss = loss_vals.mean()
                logger(f'I4D -> Sweep: {kcnt}, ETA: {int(ETA)}'
                       f', counter: {DATOS_prog}/{DATOS_sampler_I4D_n_pos}'
                       f', loss: {loss_vals.mean():.6f}'
                       f', change: {loss_change:.6f}')
                time_time = time.time()
    try:
        logger.log_plot('I4D_denoiser/train/losses', 
                        loss_vals, time_vals, time_tag = False)
        logger.log_single('I4D_denoiser/train/losses', 
                        loss_vals, time_tag = False)
    except:
        logger(f'There is no loss, status is {status}')
    fpath = logger.log_single('I4D_model/model.torch', 
        torch_handler_I4D.torchModel, time_tag = False)
    if torch_handler_I4D.torchModel.mask is not None:
        logger.log_single('I4D_model/mask', 
        torch_handler_I4D.torchModel.mask.detach().cpu().numpy())
    if torch_handler_I4D.torchModel.mu_eaxct:
        logger.log_single('I4D_model/mu_eaxct', 
        torch_handler_I4D.torchModel.mu_eaxct)
    if torch_handler_I4D.torchModel.mu is not None:
        logger.log_single('I4D_model/mu', 
        torch_handler_I4D.torchModel.mu.detach().cpu().numpy())
    if torch_handler_I4D.torchModel.PACBED is not None:
        logger.log_single('I4D_model/PACBED', 
        torch_handler_I4D.torchModel.PACBED.detach().cpu().numpy())
    return fpath, perv_loss

def cluster4_unet(
    logs_root, 
    exp_name,
    ref_dir,
    include_training,
    pretrained_fpaths_tuple,
    FLAG_denoise_STEM,                          
    log_exist_ok,
    hyps_STEM,
    hyps_I4D,
    trainable_area_STEM2D,
    trainable_area_I4D,
    PACBED_mask,
    denoise_STEM_mask,
    repeat_by_scattering,
    n_canvas_patterns,
    denoise_STEM_for_I4D,
    log_denoised_every_sweep,
    criterion_I4D_LAGMUL,
    use_classes_by_scattering,
    use_repeat_by_scattering,
    use_denoised_STEM,
    ):

    FLAG_train_STEM, FLAG_train_I4D = include_training
    include_training = any(include_training)
    pretrained_fpath_STEM, pretrained_fpath_I4D = pretrained_fpaths_tuple
    
    if(include_training):
        if(not log_exist_ok):
            logged = lognflow(logs_root)
            exp_list_names = logged.get_flist(f'{exp_name}*')
            if len(exp_list_names)>0:
                return
        logger = lognflow(logs_root, log_dir_prefix = exp_name)
    else:
        if(not log_exist_ok):
            logger = lognflow(logs_root)
        else:
            logger = lognflow(log_dir = pretrained_fpath_I4D.parent.parent)    

    logged_ref     = lognflow(ref_dir)
    data4D_noisy   = logged_ref.get_single('noisy.npy')
    data4D_nonoise = logged_ref.get_single('nonoise.npy')
    if use_denoised_STEM:
        denoised_STEM  = logged_ref.get_single('denoised_STEM.npy')
        assert denoised_STEM is not None, 'set use_denoised_STEM = False'
    
    data4D_shape = data4D_noisy.shape
    n_x, n_y, n_r, n_c = data4D_shape
    
    data4D_noisy[..., PACBED_mask==0] = 0
    data4D_nonoise[..., PACBED_mask==0] = 0

    mch = mcemtools.data4D_to_frame(data4D_nonoise[
            hyps_I4D['n_prob']//2:n_canvas_patterns - hyps_I4D['n_prob']//2,
            hyps_I4D['n_prob']//2:n_canvas_patterns - hyps_I4D['n_prob']//2])
    im = plt.imshow(mch); plt_colorbar(im)
    print(logger.log_plt('I4D_denoiser/canvas_nonoise/nonoise', dpi = 4000))  
    
    mch = mcemtools.data4D_to_frame(data4D_noisy[
            hyps_I4D['n_prob']//2:n_canvas_patterns - hyps_I4D['n_prob']//2,
            hyps_I4D['n_prob']//2:n_canvas_patterns - hyps_I4D['n_prob']//2])
    im = plt.imshow(mch); plt_colorbar(im)
    print(logger.log_plt('I4D_denoiser/canvas_noisy/noisy', dpi = 4000))        

    logger(f'hyps_STEM:')
    logger(hyps_STEM)
    logger(f'hyps_I4D:')
    logger(hyps_I4D)
    logger(f'Orginal data4D shape: {data4D_shape}')
    noisy_STEM, noisy_PACBED = mcemtools.sum_4D(data4D_noisy)
    logger(f'noisy_STEM.shape: {noisy_STEM.shape}')
    logger.log_single('noisy/STEM', noisy_STEM)
    logger.log_imshow('noisy/STEM', noisy_STEM)
    logger(f'noisy_PACBED.shape: {noisy_PACBED.shape}')
    logger.log_single('noisy/noisy_PACBED', noisy_PACBED)
    logger.log_imshow('noisy/PACBED', noisy_PACBED)
    Ne_estimated = noisy_STEM.mean()
    logger(f'estimated Ne per probe position: {Ne_estimated}')
    nonoise_STEM, nonoise_PACBED = mcemtools.sum_4D(data4D_nonoise)
    logger.log_single('nonoise/STEM', nonoise_STEM)
    logger.log_imshow('nonoise/STEM', nonoise_STEM)
    logger.log_single('nonoise/nonoise_PACBED', nonoise_PACBED)
    logger.log_imshow('nonoise/PACBED', nonoise_PACBED)
    
    #### denoising of STEM image ##########################################
    if(FLAG_denoise_STEM):
        if not use_denoised_STEM:
            denoise_STEM_mask_4D = None
            if denoise_STEM_mask is not None:
                denoise_STEM_mask = mcemtools.mask2D_to_4D(
                    denoise_STEM_mask, data4D_shape)
                
            noisy_STEM_BF, noisy_PACBED_BF = mcemtools.sum_4D(
                data4D_noisy, denoise_STEM_mask)
            
            logger.log_imshow('STEM_denoiser/noisy_STEM_BF', noisy_STEM_BF, 
                              time_tag = False)
            logger.log_imshow('STEM_denoiser/noisy_PACBED_BF', noisy_PACBED_BF, 
                              time_tag = False)
            
            data_gen_STEM = mcemtools.data_maker_2D(
                noisy_STEM_BF.copy(), noisy_STEM_BF.copy(), 
                win_shape = (hyps_STEM['win_length'], hyps_STEM['win_length']),
                skip = (hyps_STEM['skip_length'], hyps_STEM['skip_length']),
                mask_rate = hyps_STEM['mask_rate'])
    
            trainable_mask_viewed = data_gen_STEM.imbywin.image2views(
                trainable_area_STEM2D)
            trainable_mask_viewed_sums = trainable_mask_viewed.mean((1,2))
            trainable_inds = np.where(
                trainable_mask_viewed_sums == trainable_mask_viewed_sums.max())[0]
            trainable_mask_viewed *= 0
            trainable_mask_viewed[trainable_inds] += 1
            trainable_mask = data_gen_STEM.imbywin.views2image(
                trainable_mask_viewed)
            
            logger.log_imshow('STEM_denoiser/trainable_mask', trainable_mask, 
                              cmap = 'prism', time_tag = False)
            logger.log_imshow('STEM_denoiser/trainable_mask', trainable_mask > 0, 
                              cmap = 'gray', time_tag = False)
            logger.log_single('STEM_denoiser/trainable_mask', trainable_mask, 
                              time_tag = False)
            #### models definition #################################################
            torchModel_STEM = network2D(
                n_kernels = hyps_STEM['n_kernels'], 
                Ne_input = Ne_estimated)
            if(pretrained_fpath_STEM.is_file()):
                logger(f'Using: {pretrained_fpath_STEM}')
                torchModel_STEM.load_state_dict(
                    torch.load(pretrained_fpath_STEM), strict=False)
                torchModel_STEM.Ne_input = float(
                    np.load(pretrained_fpath_STEM.parent / 'Ne_input.npy'))
                torchModel_STEM.Ne_output = float(
                    np.load(pretrained_fpath_STEM.parent / 'Ne_output.npy'))
            torchModel_STEM = torchModel_STEM.float().to(device)
            
            criterion_STEM = mseLoss()
            torch_handler_STEM = \
                nn_from_torch(
                    data_generator = data_gen_STEM,
                    torchModel = torchModel_STEM,
                    lossFunc = criterion_STEM,
                    device = device,
                    logger = logger,
                    learning_rate = hyps_STEM['learning_rate'],
                    momentum = hyps_STEM['momentum'],
                    pass_indices_to_model = False) 
            
            _, denoised_STEM = infer_STEM(torch_handler_STEM,  data_gen_STEM, 
                                          logger,hyps_STEM['infer_size_STEM'])
            torchModel_STEM.Ne_output = Ne_estimated/denoised_STEM.mean()
            if(FLAG_train_STEM):
                DATOS_sampler_STEM = DATOS(trainable_inds.shape[0], 
                                         n_segments = hyps_STEM['n_segments'],
                                         mbatch_size = hyps_STEM['mbatch_size'],
                                         n_epochs = hyps_STEM['n_epochs'])
                model_path = train_STEM(hyps_STEM['n_kSweeps'], 
                                      torch_handler_STEM, 
                                      data_gen_STEM,
                                      logger, 
                                      DATOS_sampler_STEM,
                                      trainable_inds,
                                      hyps_STEM['infer_size_STEM'],
                                      hyps_STEM['mask_rate'])
                logger(f'Training STEM image denoising is finished.')
                logger(f'Model is at: {model_path.absolute()}')
    
            _, denoised_STEM = infer_STEM(torch_handler_STEM, 
                                      data_gen_STEM, logger, 
                                      hyps_STEM['infer_size_STEM'])
            logger.log_single('STEM_denoiser/denoised_STEM.npy', denoised_STEM)
            logger.log_imshow('STEM_denoiser/denoised_STEM.jpg', denoised_STEM)
            denoised_STEM_mean = denoised_STEM.mean()
            if ( (denoised_STEM_mean < Ne_estimated / 2) |
                 (denoised_STEM_mean > Ne_estimated * 2) ):
                logger('The STEM denoiser was unsuccessful'
                       f'The estimated Ne was {Ne_estimated}, but the STEM denoiser '
                       f'came up with {denoised_STEM_mean}')
                raise ValueError
        else:
            logger('denoised_STEM is provided already!')
        data4D_noisy_new = data4D_noisy.copy()
        if(denoise_STEM_for_I4D):
            ### applying denoised field image to the noisy 4D data
            data4D_noisy_new = data4D_noisy_new.reshape(n_x * n_y, n_r, n_c)
            data4D_noisy_new = data4D_noisy_new.reshape(n_x * n_y, n_r * n_c)
            data4D_noisy_new_sum = data4D_noisy_new.sum(1)
            data4D_noisy_new_tile = np.tile(
                np.array([data4D_noisy_new_sum]).swapaxes(0,1), (1, n_r * n_c))
            data4D_noisy_new[data4D_noisy_new_tile>0] /= \
                data4D_noisy_new_tile[data4D_noisy_new_tile>0]
            denoised_STEM = denoised_STEM.reshape(n_x * n_y)
            denoised_STEM_tile = np.tile(
                np.array([denoised_STEM]).swapaxes(0,1), (1, n_r * n_c))
            data4D_noisy_new *= denoised_STEM_tile
            data4D_noisy_new = data4D_noisy_new.reshape(n_x * n_y, n_r, n_c)
            data4D_noisy_new = data4D_noisy_new.reshape(n_x, n_y, n_r, n_c)
            denoised_STEM, _ = mcemtools.sum_4D(data4D_noisy_new)
    else:
        data4D_noisy_new = data4D_noisy
        denoised_STEM, _ = mcemtools.sum_4D(data4D_noisy_new)
    ########################################################################
    #### Declaring I4D model, loss and data maker ######################
    
    torchModel_I4D = network4D(
        hyps_I4D['n_prob']**2 - 1,
        n_kernels = hyps_I4D['n_kernels'],
        mask = torch.from_numpy(PACBED_mask).float().cuda()
        ).float().to(device)
    
    if(pretrained_fpath_I4D.is_file()):
        logger(f'Using: {pretrained_fpath_I4D}')
        torchModel_I4D.load_state_dict(
            torch.load(pretrained_fpath_I4D), strict=False)
        #here should be loaded more configurations for the netowrk
    if(0):
        _, PACBED  = mcemtools.sum_4D(data4D_noisy_new)
        loss_weight = PACBED.max() - noisy_PACBED
        loss_weight = loss_weight / loss_weight.max()
        loss_weight = loss_weight * 0.5 + (1 - 0.5)
        logger.log_imshow('I4D_denoiser/loss_weight', loss_weight)
        logger.log_single('I4D_denoiser/loss_weight', loss_weight)
        loss_weight = torch.from_numpy(loss_weight).float().to(device)
    else:
        loss_weight = None

    data_gen_I4D = mcemtools.feature_maker_4D(
        data4D_noisy_new, data4D_nonoise, len_side = hyps_I4D['n_prob'],
        trainable_area_I4D = trainable_area_I4D)
    
    pvalue = data_gen_I4D.reconstruct4D(data_gen_I4D.GNDTruth)
    pvalue = pvalue[hyps_I4D['n_prob']//2:5+hyps_I4D['n_prob']//2, 
                    hyps_I4D['n_prob']//2:5+hyps_I4D['n_prob']//2
                    ].reshape(25, n_r, n_c)
    logger.log_imshow_by_subplots(
        'I4D_denoiser/sample_100/data4D_nonoise', pvalue)
    pvalue = data_gen_I4D.reconstruct4D(data_gen_I4D.Y_label)
    pvalue = pvalue[hyps_I4D['n_prob']//2:5+hyps_I4D['n_prob']//2,
                    hyps_I4D['n_prob']//2:5+hyps_I4D['n_prob']//2
                    ].reshape(25, n_r, n_c)
    logger.log_imshow_by_subplots(
        'I4D_denoiser/sample_100/data4D_noisy', pvalue)
    classes = None
    if(repeat_by_scattering is None):
        trainable_inds = np.where(
            trainable_area_I4D[data_gen_I4D.xx, data_gen_I4D.yy] == 1)[0]
    else:
        logger(f'repeat_by_scattering: {repeat_by_scattering}')
        if(0):
            kmeans = KMeans(n_clusters=len(repeat_by_scattering)).fit(
                denoised_STEM.ravel().reshape(-1, 1))
            lbl = kmeans.labels_.astype('int')
            sort_inds = np.argsort(kmeans.cluster_centers_.ravel())
            new_lbls = np.empty_like(sort_inds)
            new_lbls[sort_inds] = np.arange(sort_inds.shape[0])
            lbl = mcemtools.revalue_elements(
                lbl, new_lbls).reshape(*denoised_STEM.shape)
            
            lbl_0 = lbl == 0
            lbl_0 = mcemtools.remove_islands_by_size(
                lbl_0, min_n_pix = 3, max_n_pix = np.inf)
            lbl_0 = scipy.ndimage.binary_dilation(lbl_0)
            lbl_0 = scipy.ndimage.binary_fill_holes(lbl_0)
            lbl_0 = scipy.ndimage.binary_erosion(lbl_0)
            lbl[(lbl == 0) & (lbl_0 == 0)] = 1
            lbl[lbl_0 == 1] = 0
            lbl += 1
        else:
            lbl = np.zeros(denoised_STEM.shape)
            n_classes = len(repeat_by_scattering)
            percentages_list = np.linspace(0, 100, n_classes + 1).astype('int')
            for lbl_cnt in range(n_classes):
                rng_st = np.percentile(denoised_STEM.ravel(), 
                                       percentages_list[lbl_cnt])
                rng_end = np.percentile(denoised_STEM.ravel(), 
                                       percentages_list[lbl_cnt + 1])
                lbl[(rng_st <= denoised_STEM) & 
                    (denoised_STEM < rng_end)] = lbl_cnt + 1
                lbl[lbl<1] = 1
                lbl[lbl > n_classes] = n_classes
                lbl = lbl.astype('int')
        new_lbls, lbl_counts = np.unique(lbl, return_counts = True)
            
        trainable_area_I4D[trainable_area_I4D > 0] = lbl[trainable_area_I4D > 0]
        trainable_inds = np.array([], dtype='int')
        for lblcnt, _lbl in enumerate(new_lbls):
            _inds = np.where(trainable_area_I4D[
                data_gen_I4D.xx, data_gen_I4D.yy] == _lbl)[0]
            n_repeat = 1
            if(use_repeat_by_scattering):
                n_repeat = repeat_by_scattering[lblcnt]
            if n_repeat > 1:
                _inds = np.tile(_inds, n_repeat)
            trainable_inds = np.concatenate((trainable_inds, _inds), axis = 0)

        if(use_classes_by_scattering):
            classes = lbl.copy().ravel()[trainable_inds]
            
    data_gen_I4D.trainable_inds = trainable_inds.copy()
            
    trainable_mask = 0*trainable_area_I4D.copy()
    for ind in trainable_inds:
        trainable_mask[data_gen_I4D.xx[ind], data_gen_I4D.yy[ind]] += 1
    logger.log_imshow('I4D_denoiser/trainable_mask_I4D', trainable_mask)
    logger.log_single('I4D_denoiser/trainable_mask_I4D', trainable_mask)
    logger.log_single('I4D_denoiser/classes', classes)

    logger(f'There are {data4D_noisy_new[trainable_mask>0].sum()} electrons'
           ' in this dataset.')

    noisy_PACBED_in = data_gen_I4D.noisy_PACBED.copy() / data_gen_I4D.n_pts
    noisy_STEM_in = data_gen_I4D.noisy_mu.copy()
    
    noisy_PACBED_loss = noisy_PACBED_in / data_gen_I4D.n_pts
    noisy_STEM_loss = noisy_STEM_in / PACBED_mask.sum()
    criterion_I4D = STEM4D_PoissonLoss_FnormLoss(
        mask_backprop = torch.from_numpy(PACBED_mask).float().to(device),
        x_in_max = 1000, device = 'cuda',
        output_stabilizer = 1e-6,
        noisy_PACBED = torch.from_numpy(noisy_PACBED_loss).float().to(device),
        noisy_mSTEM = torch.from_numpy(noisy_STEM_loss.ravel()).float().to(device),
        PAC_loss_factor = hyps_I4D['PAC_loss_factor'],
        mSTEM_loss_factor = hyps_I4D['mSTEM_loss_factor'],
        )

    torch_handler_I4D = nn_from_torch(
        data_generator = data_gen_I4D,
        torchModel = torchModel_I4D,
        lossFunc = criterion_I4D,
        device = device,
        logger = logger,
        learning_rate = hyps_I4D['learning_rate'],
        momentum = hyps_I4D['momentum'],
        pass_indices_to_model = True,
        fix_during_infer = True) 
    
    perv_loss = np.nan
    Elapsed_time = 0
    for kcnt in range(hyps_I4D['n_ksweeps']):
        ETA_base = Elapsed_time * (hyps_I4D['n_ksweeps'] - kcnt)
        # init_I4D(None, noisy_PACBED_in,
        #         logger, data_gen_I4D, torch_handler_I4D, hyps_I4D['infer_size_I4D'])

        if kcnt == 0:
            n_epochs = 1
        else:
            n_epochs = hyps_I4D['n_epochs']

        if kcnt < hyps_I4D['n_ksweeps'] / 2:
            init_I4D(noisy_STEM_in, None,
                     logger, data_gen_I4D, torch_handler_I4D, 
                     hyps_I4D['infer_size_I4D'])

        if(not FLAG_train_I4D):
            kcnt = hyps_I4D['n_ksweeps']
        
        criterion_I4D.LAGMUL = criterion_I4D_LAGMUL(kcnt, hyps_I4D['n_ksweeps'])
        logger(f'criterion_I4D.LAGMUL: {criterion_I4D.LAGMUL}')
        if(FLAG_train_I4D):
            DATOS_sampler_I4D = DATOS(
                trainable_inds.shape[0], 
                classes = classes,
                n_segments = hyps_I4D['n_segments'],
                mbatch_size = hyps_I4D['mbatch_size'],
                n_epochs = n_epochs)
            Elapsed_time = logger.time_stamp
            model_I4D_fpath, perv_loss = train_I4D(kcnt,
                                                   logger,
                                                   data_gen_I4D,
                                                   DATOS_sampler_I4D,
                                                   torch_handler_I4D,
                                                   trainable_inds,
                                                   perv_loss,
                                                   hyps_I4D['infer_size_I4D'],
                                                   ETA_base)
            Elapsed_time = logger.time_stamp - Elapsed_time
            logger(f'Model I4D so far: {model_I4D_fpath}')
        
        ferrs, data4D_denoised = infer_I4D(torch_handler_I4D,
                                           data_gen_I4D, logger,
                                           hyps_I4D['infer_size_I4D'])
        logger.log_imshow('I4D_denoiser/ferrs/ferrs', ferrs.reshape(
            n_x - 2 * (hyps_I4D['n_prob']//2),
            n_y - 2 * (hyps_I4D['n_prob']//2) ))
        data4D_denoised = data_gen_I4D.reconstruct4D(data4D_denoised)
            
        logger(f'data4D_denoised shape: {data4D_denoised.shape}')
        pvalue = data4D_denoised[
            hyps_I4D['n_prob']//2:hyps_I4D['n_prob']//2 + 5, 
            hyps_I4D['n_prob']//2:hyps_I4D['n_prob']//2 + 5
                ].reshape(25, n_r, n_c)
        logger.log_imshow_by_subplots(
            'I4D_denoiser/samples/data4D_denoised', 
            pvalue)
        
        ####################################################################
        data4D_noisy_new[hyps_I4D['n_prob']//2 : -(hyps_I4D['n_prob']//2),
                         hyps_I4D['n_prob']//2 : -(hyps_I4D['n_prob']//2)] \
            = data4D_denoised.copy().astype('float32')
        if(log_denoised_every_sweep):
            logger.log_single('I4D_denoiser/I4D_denoised_inter/denoised', 
                              data4D_noisy_new)
        
        # ratio = 1 / hyps_I4D['n_ksweeps']
        # data4D_noisy_new[hyps_I4D['n_prob']//2 : -(hyps_I4D['n_prob']//2),
        #                  hyps_I4D['n_prob']//2 : -(hyps_I4D['n_prob']//2)] \
        #     *=(1 - ratio)
        # data4D_noisy_new[hyps_I4D['n_prob']//2 : -(hyps_I4D['n_prob']//2),
        #                  hyps_I4D['n_prob']//2 : -(hyps_I4D['n_prob']//2)] \
        #     += ratio * np.random.poisson(data4D_denoised).astype('float32')
        ####################################################################
        
        noisy_STEM, noisy_PACBED  = mcemtools.sum_4D(data4D_noisy_new)
        Ne_estimated = noisy_STEM.mean()
        logger(f'estimated Ne per probe position: {Ne_estimated}')
        logger.log_imshow(
            'I4D_denoiser/I4D_denoised_STEM/final_STEM_noisy', 
            noisy_STEM[hyps_I4D['n_prob']//2:-hyps_I4D['n_prob']//2,
                    hyps_I4D['n_prob']//2:-hyps_I4D['n_prob']//2])
        logger.log_imshow(
            'I4D_denoiser/I4D_denoised_PACBED/final_noisy_PACBED', 
            noisy_PACBED[hyps_I4D['n_prob']//2:-hyps_I4D['n_prob']//2,
                    hyps_I4D['n_prob']//2:-hyps_I4D['n_prob']//2])
        
        mch = mcemtools.data4D_to_frame(data4D_noisy_new[
            hyps_I4D['n_prob']//2:n_canvas_patterns - hyps_I4D['n_prob']//2,
            hyps_I4D['n_prob']//2:n_canvas_patterns - hyps_I4D['n_prob']//2])
        im = plt.imshow(mch); plt_colorbar(im)
        print(logger.log_plt('I4D_denoiser/canvas_denoised/denoised', 
                             dpi = 4000))
        
        if not FLAG_train_I4D:
            break
        
    logger.log_single('I4D_denoiser/I4D_denoised/denoised', data4D_noisy_new)
    
    logger(r'--\|/'*16)
    logger_dir = logger.log_dir
    
    del logger
    
    return logger_dir
