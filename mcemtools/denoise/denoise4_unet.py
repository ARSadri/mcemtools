import time
import torch
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
from lognflow.plt_utils import plt_colorbar, plt_imshow
from lognflow import lognflow, printprogress

import mcemtools

from .DATOS         import DATOS, nn_from_torch
from .networks      import U_Net as network4D, truth_network
from .networks      import U_Net_fieldImage as network2D
# from .networks_3 import DcUnet as network4D
from .LossFunc      import mseLoss, CoMLoss, STEM4D_PoissonLoss_FnormLoss 

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
    mask_rate = 0.9, logger_prefix = ''):
    
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
            logger.imshow(
                f'{logger_prefix}_denoiser/training/output_values', output_values)
            # torch_handler.torchModel.Ne_output = \
            #     data_gen_STEM.inimg_mean/output_values.mean()        
            logger.save(f'{logger_prefix}_denoiser/model/Ne_input', 
                   float(torch_handler.torchModel.Ne_input), time_tag = False)
            logger.save(f'{logger_prefix}_denoiser/model/Ne_output', 
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
                logger.log_var(f'{logger_prefix}_denoiser/training/training_loss', loss)
                if (time.time() - time_time > 30) :
                    time_vals, loss_vals = logger.get_var(
                        f'{logger_prefix}_denoiser/training/training_loss')
                    loss_change = loss_vals.mean() - perv_loss
                    perv_loss = loss_vals.mean()
                    logger(f'FI -> Sweep: {kcnt}/{n_kSweeps}'
                           f', ETA: {ETA}'
                           f', counter: {ind_cnt}/{DATOS_sampler.n_pos}'
                           f', loss: {loss_vals.mean():.6f}'
                           f', change: {loss_change:.6f}')
                    time_time = time.time()
        try:
            logger.log_plot(f'{logger_prefix}_denoiser/training/losses', 
                            loss_vals, time_vals, time_tag = False)
        except:
            pass
    fpath = logger.save(f'{logger_prefix}_denoiser/model/model', 
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

    _, predStat_list, output_vals = torch_handler_I4D.infer(
        indices = indices,
        return_predictions = True, 
        infer_size = infer_size_I4D, 
        show_progress = True,
        predictions_statfunc_list = [data_gen_I4D.dist2Truth,
                                     data_gen_I4D.dist2label])
    fitting_errors = predStat_list[1]
    dist2Truth_vals = predStat_list[0]
    
    if not use_trainable_inds:
        indices = None
    
    fitting_errors_vis = data_gen_I4D.reconstruct2D(fitting_errors, indices)
    logger.imshow('I4D_denoiser/train/fitting_errors', fitting_errors_vis)

    dist2Truth_vals_vis = data_gen_I4D.reconstruct2D(dist2Truth_vals, indices)
    logger.imshow('I4D_denoiser/train/dist2Truth_vals', dist2Truth_vals_vis)

    if torch_handler_I4D.torchModel.mu is not None:
        logger.imshow('I4D_denoiser/train/model/mu', 
            torch_handler_I4D.torchModel.mu.reshape(data_gen_I4D.imbywin.grid_shape))

    _denoisedI4D_STEM = data_gen_I4D.reconstruct2D(
        output_vals[:, 0].sum(2).sum(1), indices)
    logger.imshow('I4D_denoiser/after_train_STEM/denoisedI4D_STEM', 
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
        ETA_base = 0,
        max_n_ValueError = 5):

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
    ValueError_cnt = 0
    pBar = printprogress(DATOS_sampler_I4D.n_pos, print_function = None)
    DATOS_inds_log = []
    while(status):
        DATOS_inds, status = DATOS_sampler_I4D()
        if status:

            # torch_handler_I4D.torchModel.train()
            # data, labels = data_gen_I4D([0, 1])
            # data = torch.from_numpy(data).float().cuda()
            # preds = torch_handler_I4D.torchModel(data[0], [0, 1])
            # print(preds.sum((1,2, 3)))
            # torch_handler_I4D.torchModel.eval()
            # preds = torch_handler_I4D.torchModel(data[0], [0, 1])
            # print(preds.sum((1,2, 3)))

            DATOS_prog = DATOS_sampler_I4D.next_in_line
            n_classes = DATOS_sampler_I4D.n_classes
            DATOS_inds_log.append(DATOS_inds)
            ETA = int(pBar(DATOS_inds.shape[0] / n_classes)) + ETA_base
            inds = trainable_inds[DATOS_inds].copy()
            loss = torch_handler_I4D.update(inds)
            if np.isinf(loss):
                # loss = torch_handler_I4D.update(inds)
                ValueError_cnt += 1
                ValueError_reset = False
                logger(f'inf no. {ValueError_cnt}' + '.'*30 + f' {inds}')
                if ValueError_cnt > max_n_ValueError:
                    raise ValueError
                else:
                    continue
            else:
                ValueError_reset = True
            if np.isnan(loss):
                ValueError_cnt += 1
                ValueError_reset = False
                logger(f'nan no. {ValueError_cnt}' + '.'*30 + f' {inds}')
                if ValueError_cnt > max_n_ValueError:
                    raise ValueError
                else:
                    continue
            else:
                ValueError_reset = True
            if ValueError_reset:
                ValueError_cnt = 0
            
            logger.log_var('I4D_denoiser/train/training_loss', loss)
            if (time.time() - time_time > 20) :
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
        logger.save('I4D_denoiser/train/losses', 
                        loss_vals, time_tag = False)
    except:
        logger(f'There is no loss, status is {status}')
    fpath = logger.save('I4D_model/model.torch', 
        torch_handler_I4D.torchModel, time_tag = False)
    if torch_handler_I4D.torchModel.mask is not None:
        logger.save('I4D_model/mask', 
        torch_handler_I4D.torchModel.mask.detach().cpu().numpy())
    if torch_handler_I4D.torchModel.mu_eaxct is not None:
        logger.save('I4D_model/mu_eaxct', 
        torch_handler_I4D.torchModel.mu_eaxct.detach().cpu().numpy())
    if torch_handler_I4D.torchModel.mu is not None:
        logger.save('I4D_model/mu', 
        torch_handler_I4D.torchModel.mu.detach().cpu().numpy())
    if torch_handler_I4D.torchModel.PACBED is not None:
        logger.save('I4D_model/PACBED', 
        torch_handler_I4D.torchModel.PACBED.detach().cpu().numpy())
    return fpath, perv_loss

def denoise4_unet(
    logs_root, 
    exp_name,
    ref_dir,
    include_training,
    pretrained_fpaths_tuple,
    FLAG_denoise_STEM,                          
    log_exist_ok,
    hyps_STEM,
    hyps_I4D,
    hyps_CoM,
    trainable_area_STEM2D,
    trainable_area,
    PACBED_mask,
    denoise_STEM_mask,
    repeat_by_scattering,
    n_canvas_patterns,
    denoise_STEM_for_I4D,
    log_denoised_every_sweep,
    criterion_I4D_LAGMUL,
    use_classes_by_scattering,
    use_repeat_by_scattering,
    use_pre_denoised_STEM,
    STEM_denoiser_model_type,
    rank_info,
    FLAG_train_CoM,      
    use_pre_denoised_CoM,
    FLAG_denoise_CoM,    
    denoise_CoM_for_I4D,
    CoM_denoiser_model_type,
    ):

    FLAG_train_STEM, include_training = include_training
    include_training = any(include_training)
    pretrained_STEM_fpath, pretrained_model_fpath = pretrained_fpaths_tuple
    
    if(include_training):
        if(not log_exist_ok):
            logged = lognflow(log_dir = logs_root)
            exp_list_names = logged.get_flist(
                f'denoised*/I4D_denoiser/I4D_denoised/denoised_*.npy')
            if len(exp_list_names)>0:
                return
        logger = lognflow(logs_root, log_dir_prefix = 'denoised4D_UNet')
    else:
        if(not log_exist_ok):
            logger = lognflow(logs_root)
        else:
            logger = lognflow(log_dir = pretrained_model_fpath.parent.parent)    

    logger_ref            = lognflow(log_dir = ref_dir, time_tag = False)
    data4D_noisy          = logger_ref.load('noisy.npy')
    data4D_nonoise        = logger_ref.load('nonoise.npy')
    if use_pre_denoised_STEM:
        denoised_STEM  = logger_ref.load('denoised_STEM*.npy')
        if denoised_STEM is None:
            use_pre_denoised_STEM = False
    
    data4D_shape = data4D_noisy.shape
    n_x, n_y, n_r, n_c = data4D_shape
    
    data4D_noisy[..., PACBED_mask==0] = 0
    data4D_nonoise[..., PACBED_mask==0] = 0

    logger(f'Orginal data4D shape: {data4D_shape}')
    noisy_STEM, noisy_PACBED = mcemtools.sum_4D(data4D_noisy)
    nonoise_STEM, nonoise_PACBED = mcemtools.sum_4D(data4D_nonoise)
    Ne_estimated = noisy_STEM.mean()
    logger(f'estimated Ne per probe position: {Ne_estimated}')

    n_probes = hyps_I4D['n_prob']
    edgew = int(n_probes//2)
    
    #### denoising of STEM image ##########################################
    if(FLAG_denoise_STEM):
        logger(f'hyps_STEM:')
        logger(hyps_STEM)
        
        if not use_pre_denoised_STEM:
            if STEM_denoiser_model_type == 'UNET':
                
                denoise_STEM_mask_4D = None
                if denoise_STEM_mask is not None:
                    denoise_STEM_mask = mcemtools.mask2D_to_4D(
                        denoise_STEM_mask, data4D_shape)
                    
                noisy_STEM_BF, noisy_PACBED_BF = mcemtools.sum_4D(
                    data4D_noisy, denoise_STEM_mask)
                
                logger.imshow('STEM_denoiser/noisy_STEM_BF', noisy_STEM_BF, 
                                  time_tag = False)
                logger.imshow('STEM_denoiser/noisy_PACBED_BF', noisy_PACBED_BF, 
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
                
                logger.imshow('STEM_denoiser/trainable_mask', trainable_mask, 
                                  cmap = 'prism', time_tag = False)
                logger.imshow('STEM_denoiser/trainable_mask', trainable_mask > 0, 
                                  cmap = 'gray', time_tag = False)
                logger.save('STEM_denoiser/trainable_mask', trainable_mask, 
                                  time_tag = False)
                #### models definition ###############################################
                torchModel_STEM = network2D(
                    n_kernels = hyps_STEM['n_kernels'], 
                    Ne_input = Ne_estimated)
                if(pretrained_STEM_fpath.is_file()):
                    logger(f'Using: {pretrained_STEM_fpath}')
                    torchModel_STEM.load_state_dict(
                        torch.load(pretrained_STEM_fpath), strict=False)
                    torchModel_STEM.Ne_input = float(
                        np.load(pretrained_STEM_fpath.parent / 'Ne_input.npy'))
                    torchModel_STEM.Ne_output = float(
                        np.load(pretrained_STEM_fpath.parent / 'Ne_output.npy'))
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
                        momentum = hyps_STEM['learning_momentum'],
                        pass_indices_to_model = False) 
                
                _, denoised_STEM = infer_STEM(torch_handler_STEM,  data_gen_STEM, 
                                              logger,hyps_STEM['infer_size_STEM'])
                torchModel_STEM.Ne_output = Ne_estimated/denoised_STEM.mean()
                if(FLAG_train_STEM):
                    DATOS_sampler_STEM = DATOS(
                        trainable_inds.shape[0], 
                        n_segments = hyps_STEM['n_segments'],
                        mbatch_size = hyps_STEM['mbatch_size'],
                        n_epochs = hyps_STEM['n_epochs'])
                    model_path = train_STEM(
                        hyps_STEM['n_kSweeps'], 
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
                logger.save('STEM_denoiser/denoised_STEM.npy', 
                                  denoised_STEM)
                logger.imshow('STEM_denoiser/denoised_STEM.jpg',
                                  denoised_STEM)
                denoised_STEM_mean = denoised_STEM.mean()
                if ( (denoised_STEM_mean < Ne_estimated / 2) |
                     (denoised_STEM_mean > Ne_estimated * 2) ):
                    logger('The STEM denoiser was unsuccessful'
                           f'The estimated Ne was {Ne_estimated},'
                           f' but the STEM denoiser '
                           f'came up with {denoised_STEM_mean}')
                    raise ValueError
            elif STEM_denoiser_model_type == 'TSVD':
                denoised_STEM = mcemtools.tensor_svd.tensor_svd_denoise(
                    data4D_noisy, rank = (
                        rank_info.n_x, rank_info.n_y, rank_info.n_pix))
                denoised_STEM = denoised_STEM.reshape(*data4D_shape)
                denoised_STEM, _ = mcemtools.sum_4D(denoised_STEM)
            else:
                denoised_STEM, _ = mcemtools.sum_4D(data4D_noisy)
        else:
            logger('denoised_STEM is provided already!')
        data4D_noisy = data4D_noisy.copy()
        if(denoise_STEM_for_I4D):
            ### applying denoised stem image to the noisy 4D data
            data4D_noisy = data4D_noisy.reshape(n_x * n_y, n_r, n_c)
            data4D_noisy = data4D_noisy.reshape(n_x * n_y, n_r * n_c)
            data4D_noisy_sum = data4D_noisy.sum(1)
            data4D_noisy_tile = np.tile(
                np.array([data4D_noisy_sum]).swapaxes(0,1), (1, n_r * n_c))
            data4D_noisy[data4D_noisy_tile>0] /= \
                data4D_noisy_tile[data4D_noisy_tile>0]
            denoised_STEM = denoised_STEM.reshape(n_x * n_y)
            denoised_STEM_tile = np.tile(
                np.array([denoised_STEM]).swapaxes(0,1), (1, n_r * n_c))
            data4D_noisy *= denoised_STEM_tile
            data4D_noisy = data4D_noisy.reshape(n_x * n_y, n_r, n_c)
            data4D_noisy = data4D_noisy.reshape(n_x, n_y, n_r, n_c)
            denoised_STEM, denoised_PACBED = mcemtools.sum_4D(data4D_noisy)
    else:
        data4D_noisy = data4D_noisy.copy()
        denoised_STEM, denoised_PACBED = mcemtools.sum_4D(data4D_noisy)
    ########################################################################
    
    if(FLAG_denoise_CoM):
        logger(f'hyps_CoM:')
        logger(hyps_CoM)
        if not use_pre_denoised_CoM:
            if CoM_denoiser_model_type == 'UNET':
                
                denoise_CoM_mask_4D = None
                # if denoise_CoM_mask is not None:
                #     denoise_CoM_mask = mcemtools.mask2D_to_4D(
                #         denoise_CoM_mask, data4D_shape)
                    
                noisy_CoM_x, noisy_CoM_y = mcemtools.centre_of_mass_4D(
                    data4D_noisy)
                
                if hyps_CoM['denoise_amp_angle']:
                    coms = noisy_CoM_x + 1j * noisy_CoM_y
                    noisy_CoM_x, noisy_CoM_y = np.abs(coms), np.angle(coms)
                
                logger.imshow(
                    'noisy_CoMs', noisy_CoM_x + 1j * noisy_CoM_y, 
                    time_tag = False)
                logger.imshow(
                    'noisy_CoMs_c', noisy_CoM_x + 1j * noisy_CoM_y, 
                    time_tag = False, cmap = 'complex')

                denoised_CoMs = []
                for noisy_CoM, dir_name in zip(
                    [noisy_CoM_x, noisy_CoM_y], ['CoM_x', 'CoM_y']):
                
                    logger.imshow(f'{dir_name}_denoiser/noisy_{dir_name}', noisy_CoM, 
                                      time_tag = False)
                    data_gen_CoM = mcemtools.data_maker_2D(
                        noisy_CoM.copy(), noisy_CoM.copy(), 
                        win_shape = (hyps_CoM['win_length'], hyps_CoM['win_length']),
                        skip = (hyps_CoM['skip_length'], hyps_CoM['skip_length']),
                        mask_rate = hyps_CoM['mask_rate'])
                    
                    trainable_area_CoM2D = np.ones((n_x, n_y))
                    
                    trainable_mask_viewed = data_gen_CoM.imbywin.image2views(
                        trainable_area_CoM2D)
                    trainable_mask_viewed_sums = trainable_mask_viewed.mean((1,2))
                    trainable_inds = np.where(
                        trainable_mask_viewed_sums == trainable_mask_viewed_sums.max())[0]
                    trainable_mask_viewed *= 0
                    trainable_mask_viewed[trainable_inds] += 1
                    trainable_mask = data_gen_CoM.imbywin.views2image(
                        trainable_mask_viewed)
                    
                    logger.imshow(
                        f'{dir_name}_denoiser/trainable_mask', trainable_mask, 
                        cmap = 'prism', time_tag = False)
                    logger.imshow(
                        f'{dir_name}_denoiser/trainable_mask', trainable_mask > 0, 
                                      cmap = 'gray', time_tag = False)
                    logger.save(
                        f'{dir_name}_denoiser/trainable_mask', trainable_mask, 
                                      time_tag = False)
                    #### models definition #################################################
                    torchModel_CoM = network2D(
                        n_kernels = hyps_CoM['n_kernels'], 
                        Ne_input = Ne_estimated)
                    torchModel_CoM = torchModel_CoM.float().to(device)
                    
                    if hyps_CoM['denoise_amp_angle']:
                        if dir_name == 'CoM_x':
                            criterion_CoM = mseLoss()
                        else:
                            criterion_CoM = CoMLoss()#mseLoss()
                    torch_handler_CoM = \
                        nn_from_torch(
                            data_generator = data_gen_CoM,
                            torchModel = torchModel_CoM,
                            lossFunc = criterion_CoM,
                            device = device,
                            logger = logger,
                            learning_rate = hyps_CoM['learning_rate'],
                            momentum = hyps_CoM['learning_momentum'],
                            pass_indices_to_model = False) 
                    
                    _, denoised_CoM = infer_STEM(torch_handler_CoM,  data_gen_CoM, 
                                                  logger,hyps_CoM['infer_size_CoM'])
                    if(FLAG_train_CoM):
                        DATOS_sampler_CoM = DATOS(
                            trainable_inds.shape[0], 
                            n_segments = hyps_CoM['n_segments'],
                            mbatch_size = hyps_CoM['mbatch_size'],
                            n_epochs = hyps_CoM['n_epochs'])
                        model_path = train_STEM(
                            hyps_CoM['n_kSweeps'], 
                            torch_handler_CoM, 
                            data_gen_CoM,
                            logger, 
                            DATOS_sampler_CoM,
                            trainable_inds,
                            hyps_CoM['infer_size_CoM'],
                            hyps_CoM['mask_rate'],
                            logger_prefix = dir_name)
                        logger(f'Training CoM_{dir_name} image denoising is finished.')
                        logger(f'Model is at: {model_path.absolute()}')
            
                    _, denoised_CoM = infer_STEM(torch_handler_CoM, 
                                              data_gen_CoM, logger, 
                                              hyps_CoM['infer_size_CoM'])
                    logger.save(f'{dir_name}_denoiser/denoised_CoM.npy', 
                                      denoised_CoM)
                    logger.imshow(f'{dir_name}_denoiser/denoised_CoM.jpg',
                                      denoised_CoM)
                    denoised_CoMs.append(denoised_CoM)
                denoised_CoMs = np.array(denoised_CoMs)
                if hyps_CoM['denoise_amp_angle']:
                    denoised_CoM_x = denoised_CoMs[0] * np.cos(denoised_CoMs[1])
                    denoised_CoM_y = denoised_CoMs[0] * np.sin(denoised_CoMs[1])
                    denoised_CoMs = np.array([denoised_CoM_x, denoised_CoM_y])
            elif CoM_denoiser_model_type == 'TSVD':
                denoised_CoM = mcemtools.tensor_svd.tensor_svd_denoise(
                    data4D_noisy, rank = (
                        rank_info.n_x, rank_info.n_y, rank_info.n_pix))
                denoised_CoM = denoised_CoM.reshape(*data4D_shape)
                denoised_CoM_x, denoised_CoM_y = mcemtools.centre_of_mass_4D(denoised_CoM)
                denoised_CoMs = np.array([denoised_CoM_x, denoised_CoM_y])
            else:
                denoised_CoM_x, denoised_CoM_y = mcemtools.centre_of_mass_4D(data4D_noisy)
                denoised_CoMs = np.array([denoised_CoM_x, denoised_CoM_y])
            
            logger.save('denoised_CoMs', denoised_CoMs, time_tag = False)
            logger.imshow(
                'denoised_CoMs', denoised_CoMs[0] + 1j * denoised_CoMs[1], 
                time_tag = False)
            logger.imshow(
                'denoised_CoMs_c', denoised_CoMs[0] + 1j * denoised_CoMs[1], 
                time_tag = False, cmap = 'complex')
        else:
            denoised_CoMs = logger_ref.load('denoised_CoMs.npy')
            logger('denoised_CoMs is provided in ref  already!')
    else:
        denoised_CoM_x, denoised_CoM_y = mcemtools.centre_of_mass_4D(data4D_noisy)
        denoised_CoMs = np.array([denoised_CoM_x, denoised_CoM_y])
    
    #### Declaring I4D model, loss and data maker ######################
    logger(f'hyps_I4D:')
    logger(hyps_I4D)
    logger.save('I4D_denoiser/n_kernels', 
                      np.array([hyps_I4D['n_kernels']]), time_tag = False)
    logger.imshow('I4D_denoised_STEM', denoised_STEM)
    logger.imshow('I4D_denoised_PACBED', denoised_PACBED)
    
    if hyps_I4D['test_mode']:
        torchModel_I4D = truth_network()
    else:
        torchModel_I4D = network4D(
            n_probes**2 - 1,
            n_kernels = hyps_I4D['n_kernels'],
            mask = torch.from_numpy(PACBED_mask).float().cuda()).cuda()
    
    if hyps_I4D['test_mode']:
        logger('NOTE: NO TRAINING WILL BE CARRIED ON AS THIS IS TEST MODE')
        include_training = False
        
    if(pretrained_model_fpath.is_file()):
        logger(f'Using: {pretrained_model_fpath}')
        torchModel_I4D.load_state_dict(
            torch.load(pretrained_model_fpath), strict=False)
    
    loss_weight = None

    logger('Making a training dataset using noisy')
    data_gen_I4D = mcemtools.data_maker_4D(
        data4D_noisy.astype('float32'), data4D_nonoise, len_side = n_probes,
        trainable_area = trainable_area)
    
    recon = data_gen_I4D.reconstruct4D(data_gen_I4D.GNDTruth)
    frame = mcemtools.data4D_to_frame(recon[
        edgew:n_canvas_patterns+edgew, edgew:n_canvas_patterns+edgew])
    logger.imshow('I4D_denoiser/sample/nonoise', frame)

    recon = data_gen_I4D.reconstruct4D(data_gen_I4D.Y_label)
    frame = mcemtools.data4D_to_frame(recon[
        edgew:n_canvas_patterns+edgew, edgew:n_canvas_patterns+edgew])
    logger.imshow('I4D_denoiser/sample/noisy', frame)

    classes = None
    if(repeat_by_scattering is None):
        trainable_inds = np.where(
            trainable_area[data_gen_I4D.xx, data_gen_I4D.yy] == 1)[0]
    else:
        logger(f'repeat_by_scattering: {repeat_by_scattering}')
        lbl = np.zeros(denoised_STEM.shape)
        n_classes = len(repeat_by_scattering)
        percentages_list = np.linspace(0, 100, n_classes + 1).astype('int')
        for lbl_cnt in range(n_classes):
            rng_st = np.percentile(
                denoised_STEM.ravel(), percentages_list[lbl_cnt])
            rng_end = np.percentile(
                denoised_STEM.ravel(), percentages_list[lbl_cnt + 1])
            lbl[(rng_st <= denoised_STEM) & 
                (denoised_STEM < rng_end)] = lbl_cnt + 1
            lbl[lbl<1] = 1
            lbl[lbl > n_classes] = n_classes
            lbl = lbl.astype('int')
        new_lbls, lbl_counts = np.unique(lbl, return_counts = True)
            
        trainable_area[trainable_area > 0] = lbl[trainable_area > 0]
        trainable_inds = np.array([], dtype='int')
        for lblcnt, _lbl in enumerate(new_lbls):
            _inds = np.where(trainable_area[
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
            
    trainable_mask = 0*trainable_area.copy()
    for ind in trainable_inds:
        trainable_mask[data_gen_I4D.xx[ind], data_gen_I4D.yy[ind]] += 1
    logger.imshow('I4D_denoiser/trainable_mask_I4D', trainable_mask, time_tag = False)
    logger.save('I4D_denoiser/trainable_mask_I4D', trainable_mask, time_tag = False)
    logger.save('I4D_denoiser/classes', classes, time_tag = False)

    logger(f'There are {data4D_noisy[trainable_mask>0].sum()} electrons'
           ' in this dataset.')

    noisy_PACBED_in = data_gen_I4D.noisy_PACBED.copy() / data_gen_I4D.n_pts
    noisy_STEM_in = data_gen_I4D.noisy_mu.copy()
    
    if hyps_I4D['use_mu_eaxct']:
        mu_eaxct = noisy_STEM_in[edgew:-edgew, edgew:-edgew].ravel()
        torchModel_I4D.mu_eaxct = torch.from_numpy(mu_eaxct.copy()).float().cuda()
    
    noisy_PACBED_loss = noisy_PACBED_in / data_gen_I4D.n_pts
    noisy_STEM_loss = noisy_STEM_in / PACBED_mask.sum()
    criterion_I4D = STEM4D_PoissonLoss_FnormLoss(
        mask_backprop = torch.from_numpy(PACBED_mask).float().to(device),
        x_in_max = 1000, device = 'cuda',
        output_stabilizer = 1e-6,
        noisy_PACBED = torch.from_numpy(noisy_PACBED_loss).float().to(device),
        noisy_mSTEM = torch.from_numpy(
            noisy_STEM_loss.ravel()).float().to(device),
        PAC_loss_factor = hyps_I4D['PAC_loss_factor'],
        mSTEM_loss_factor = hyps_I4D['mSTEM_loss_factor'],
        )
    if hyps_I4D['test_mode']:
        torchModel_I4D.dgen = data_gen_I4D
    torch_handler_I4D = nn_from_torch(
        data_generator = data_gen_I4D,
        torchModel = torchModel_I4D,
        lossFunc = criterion_I4D,
        device = device,
        logger = logger,
        learning_rate = 1e-6,
        momentum = 1e-7,
        pass_indices_to_model = True,
        fix_during_infer = True,
        test_mode = hyps_I4D['test_mode'])    
    
    perv_loss = np.nan
    Elapsed_time = 0

    n_refine_steps = hyps_I4D['n_refine_steps']

    refine_step_list = np.arange(1, n_refine_steps)
    for refine_step in refine_step_list:
        rejection_ratio = hyps_I4D['rejection_ratio_list'][refine_step]
        if hyps_I4D['reset_on_refine'] & (not hyps_I4D['test_mode']):
            torch_handler_I4D.reset()

        if (refine_step > 1) & (not hyps_I4D['test_mode']):
            torch_handler_I4D.update_learning_rate(hyps_I4D['learning_rate'])
            torch_handler_I4D.update_momentum(hyps_I4D['learning_momentum'])

        n_ksweeps = hyps_I4D['n_ksweeps']
        if refine_step == n_refine_steps - 1:
            n_ksweeps = hyps_I4D['n_ksweeps_last']

        for kcnt in range(n_ksweeps):
            ETA_base = Elapsed_time * ((n_ksweeps - kcnt) + 
                                    n_ksweeps * (len(refine_step_list) - refine_step))

            if refine_step == 1:
                if kcnt == 0:
                    n_epochs = 1
                elif (not hyps_I4D['test_mode']):
                    torch_handler_I4D.update_learning_rate(hyps_I4D['learning_rate'])
                    torch_handler_I4D.update_momentum(hyps_I4D['learning_momentum'])
                    n_epochs = hyps_I4D['n_epochs']

                if kcnt == 0:
                    init_I4D(noisy_STEM_in, None,
                             logger, data_gen_I4D, torch_handler_I4D, 
                             hyps_I4D['infer_size_I4D'])

                if(not include_training):
                    kcnt = n_ksweeps
                
                criterion_I4D.LAGMUL = criterion_I4D_LAGMUL(kcnt, n_ksweeps)
            logger(f'criterion_I4D.LAGMUL: {criterion_I4D.LAGMUL}')
            if(include_training):
                DATOS_sampler_I4D = DATOS(
                    trainable_inds.shape[0], classes = classes,
                    n_segments  = hyps_I4D['n_segments'],
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
            
            ferrs, data4D_denoised = infer_I4D(
                torch_handler_I4D, data_gen_I4D, logger, 
                hyps_I4D['infer_size_I4D'])
            
            print(f'data4D_denoised: {data4D_denoised.shape}')
            
            if rejection_ratio:
                for patt in data4D_denoised:
                    patt[patt < np.percentile(patt.ravel(), rejection_ratio)] = 0
            
            logger.imshow('I4D_denoiser/fitting_errors/ferrs',
                          data_gen_I4D.reconstruct2D(ferrs))

            data4D_denoised = data_gen_I4D.reconstruct4D(data4D_denoised)
            frame = mcemtools.data4D_to_frame(data4D_denoised[
                edgew:n_canvas_patterns+edgew, edgew:n_canvas_patterns+edgew])
            logger.imshow('I4D_denoiser/sample_denoised/denoised', frame)

            com_x, com_y = mcemtools.centre_of_mass_4D(
                data4D_denoised[edgew:-edgew, edgew:-edgew])
            logger.imshow('I4D_denoiser/com_complex/com',
                           com_x + 1j * com_y, cmap = 'complex')
            logger.imshow('I4D_denoiser/com_xy/com',
                           com_x + 1j * com_y, cmap = 'real_imag')
            
            data4D_denoised = data_gen_I4D.reconstruct4D(data4D_denoised)
            logger(f'data4D_denoised: {data4D_denoised.shape}')
            frame = mcemtools.data4D_to_frame(
                data4D_denoised[edgew:n_show+edgew, edgew:n_show+edgew])
            logger.imshow('I4D_denoiser/sample_denoised/denoised', frame)

            com_x, com_y = mcemtools.centre_of_mass_4D(
                data4D_denoised[edgew:-edgew, edgew:-edgew])
            logger.imshow('I4D_denoiser/com_complex/com',
                           com_x + 1j * com_y, cmap = 'complex')
            logger.imshow('I4D_denoiser/com_xy/com',
                           com_x + 1j * com_y, cmap = 'real_imag')
            
            noisy_STEM, noisy_PACBED  = mcemtools.sum_4D(
                data4D_denoised[edgew:-edgew, edgew:-edgew])
            Ne_estimated = noisy_STEM.mean()
            logger(f'estimated Ne per probe position: {Ne_estimated}')
            logger.imshow(
                'I4D_denoiser/I4D_denoised_STEM/final_STEM_noisy', noisy_STEM)
            logger.imshow(
                'I4D_denoiser/I4D_denoised_PACBED/final_noisy_PACBED', noisy_PACBED)
            
            if not include_training:
                break
        
        if len(refine_step_list) > 1:
            logger('Making a diffused input dataset')
            beta = refine_step/(len(refine_step_list) + 1)
            logger(f'The combination weight is: {beta}')
            
            if(log_denoised_every_sweep):
                logger.save('I4D_denoiser/I4D_denoised_inter/denoised', data4D_denoised)
            
            data4D_noisy_diffused = data4D_noisy.copy()
            data4D_noisy_diffused[edgew : -(edgew), edgew : -(edgew)] *= beta
            data4D_noisy_diffused[edgew : -(edgew), edgew : -(edgew)] += \
                data4D_denoised[edgew:-edgew, edgew:-edgew].copy() * (1 - beta)
            data4D_noisy_diffused = data4D_noisy_diffused.astype('float32')
            data_gen_I4D.update(data4D_noisy_diffused, 
                                update_label = hyps_I4D['refine_by_labels'])
            
            if hyps_I4D['refine_by_labels']:
                logger('labels are diffused too.')
                recon = data_gen_I4D.reconstruct4D(data_gen_I4D.Y_label)
                frame = mcemtools.data4D_to_frame(
                    recon[edgew:n_show+edgew, edgew:n_show+edgew])
                logger.imshow('I4D_denoiser/sample/data4D_diffused', frame)
            
            logger('training dataset is modified with diffused input')
            del data4D_noisy_diffused
    
            hyps_I4D['learning_rate'] *= hyps_I4D['learning_rate_decay']
            hyps_I4D['learning_momentum'] *= hyps_I4D['learning_momentum_decay']
            logger(hyps_I4D)
    
    logger.save('I4D_denoiser/I4D_denoised/denoised', data4D_noisy)
    
    frame_denoised = mcemtools.data4D_to_frame(data4D_noisy)
    frame_nonoise = mcemtools.data4D_to_frame(data4D_nonoise)
    frame_noisy = mcemtools.data4D_to_frame(data4D_noisy)
    logger.save('I4D_denoiser/canvases/denoised', frame_denoised)
    logger.save('I4D_denoiser/canvases/nonoise', frame_nonoise)
    logger.save('I4D_denoiser/canvases/noisy', frame_noisy)
    
    logger_dir = logger.log_dir
    logger(f'Check in: {logger_dir}')
    logger('__`T*'*16)
    
    if hyps_I4D['test_mode']:
        logger('NOTE: NO TRAINING WAS CARRIED ON AS THIS IS TEST MODE')
        
    del logger
    
    return logger_dir

def criterion_I4D_LAGMUL_recom(kcnt, n_ksweeps): 
    k_ratio = kcnt/n_ksweeps
    #LAGMUL: 1 is for Gaussian and 0 is for Poisson
    if(  (0   < k_ratio) | (k_ratio <= 2/3)):
        LAGMUL = 1
    elif((2/3 < k_ratio) | (k_ratio <=   1)):
        LAGMUL = 0
    return LAGMUL

def denoise4D_unet(
    logs_root, 
    hyps_I4D,
    criterion_I4D_LAGMUL = criterion_I4D_LAGMUL_recom,
    include_training = True,
    pretrained_model_fpath = None,
    n_show = 16,
    log_denoised_every_sweep = True,
    ):

    exp_name = logs_root.stem

    assert (logs_root / 'ref/noisy.npy').is_file(), \
        'make sure you have a ref directory in logs_root that includes noisy.npy'
    
    logger = lognflow(logs_root, log_dir_prefix = 'denoised4D_UNet')
    logger.log_code()
    logger(f'hyps_I4D:{hyps_I4D}')
    logger.save('I4D_denoiser/hyps_I4D', hyps_I4D, time_tag = False)

    data4D_noisy = np.load(logs_root / 'ref/noisy.npy').astype('float32')
    if not (logs_root / 'ref/nonoise.npy').is_file():
        print('ref/nonoise.npy not found!')
        data4D_nonoise = data4D_noisy
    else:
        data4D_nonoise = np.load(logs_root / 'ref/nonoise.npy').astype('float32')
    
    data4D_shape = data4D_noisy.shape
    n_x, n_y, n_r, n_c = data4D_shape
    logger(f'Orginal data4D shape: {data4D_shape}')
    
    if 'PACBED_mask' in hyps_I4D:
        PACBED_mask = hyps_I4D['PACBED_mask']
    else:
        PACBED_mask = None
        
    if PACBED_mask is None:
        PACBED_mask = np.ones((n_r, n_c))
    if 'trainable_area' in hyps_I4D:
        trainable_area = hyps_I4D['trainable_area']
    else:
        trainable_area = None
        
        
    if trainable_area is None:
        trainable_area = np.ones((n_x, n_y))
    
    if PACBED_mask.mean() > 0:
        logger(f'PACBED_mask with average {PACBED_mask.mean()}')
        data4D_noisy[..., PACBED_mask==0] = 0
        data4D_nonoise[..., PACBED_mask==0] = 0

    noisy_STEM, noisy_PACBED = mcemtools.sum_4D(data4D_noisy)
    Ne_estimated = noisy_STEM.mean()
    logger(f'estimated Ne per probe position: {Ne_estimated}')

    n_probes = hyps_I4D['n_prob']
    assert n_probes //2 != n_probes / 2, f'n_probes, {n_probes}, should be odd'
    edgew = int(n_probes//2)
    
    #### defining I4D model, loss and data maker ######################
    
    if hyps_I4D['test_mode']:
        torchModel_I4D = truth_network()
    else:
        torchModel_I4D = network4D(
            n_probes**2 - 1,
            n_kernels = hyps_I4D['n_kernels'],
            mask = torch.from_numpy(PACBED_mask).float().cuda()).cuda()
    
    if hyps_I4D['test_mode']:
        logger('NOTE: NO TRAINING WILL BE CARRIED ON AS THIS IS TEST MODE')
        include_training = False
        
    if pretrained_model_fpath is not None:
        if(pretrained_model_fpath.is_file()):
            logger(f'Using: {pretrained_model_fpath}')
            torchModel_I4D.load_state_dict(
                torch.load(pretrained_model_fpath), strict=False)
    
    loss_weight = None

    logger('Making a training dataset using noisy')
    data_gen_I4D = mcemtools.data_maker_4D(
        data4D_noisy, data4D_nonoise, len_side = n_probes,
        trainable_area_I4D = trainable_area)
    
    recon = data_gen_I4D.reconstruct4D(data_gen_I4D.GNDTruth)
    frame = mcemtools.data4D_to_frame(recon[edgew:n_show+edgew, edgew:n_show+edgew])
    logger.imshow('I4D_denoiser/sample/nonoise', frame)

    recon = data_gen_I4D.reconstruct4D(data_gen_I4D.Y_label)
    frame = mcemtools.data4D_to_frame(recon[edgew:n_show+edgew, edgew:n_show+edgew])
    logger.imshow('I4D_denoiser/sample/noisy', frame)

    classes = None
    repeat_by_scattering = hyps_I4D['repeat_by_scattering']
    if(repeat_by_scattering is None):
        trainable_inds = np.where(
            trainable_area[data_gen_I4D.xx, data_gen_I4D.yy] == 1)[0]
    else:
        logger(f'repeat_by_scattering: {repeat_by_scattering}')
        lbl = np.zeros(noisy_STEM.shape)
        n_classes = len(repeat_by_scattering)
        percentages_list = np.linspace(0, 100, n_classes + 1).astype('int')
        for lbl_cnt in range(n_classes):
            rng_st = np.percentile(
                noisy_STEM.ravel(), percentages_list[lbl_cnt])
            rng_end = np.percentile(
                noisy_STEM.ravel(), percentages_list[lbl_cnt + 1])
            lbl[(rng_st <= noisy_STEM) & 
                (noisy_STEM < rng_end)] = lbl_cnt + 1
            lbl[lbl<1] = 1
            lbl[lbl > n_classes] = n_classes
            lbl = lbl.astype('int')
        new_lbls, lbl_counts = np.unique(lbl, return_counts = True)
            
        trainable_area[trainable_area > 0] = lbl[trainable_area > 0]
        trainable_inds = np.array([], dtype='int')
        for lblcnt, _lbl in enumerate(new_lbls):
            _inds = np.where(trainable_area[
                data_gen_I4D.xx, data_gen_I4D.yy] == _lbl)[0]
            n_repeat = 1
            if repeat_by_scattering is not None:
                n_repeat = repeat_by_scattering[lblcnt]
            if n_repeat > 1:
                _inds = np.tile(_inds, n_repeat)
            trainable_inds = np.concatenate((trainable_inds, _inds), axis = 0)

        if(repeat_by_scattering is not None):
            classes = lbl.copy().ravel()[trainable_inds]
            
    data_gen_I4D.trainable_inds = trainable_inds.copy()
            
    trainable_mask = 0*trainable_area.copy()
    for ind in trainable_inds:
        trainable_mask[data_gen_I4D.xx[ind], data_gen_I4D.yy[ind]] += 1
    logger.imshow('I4D_denoiser/trainable_mask_I4D', trainable_mask, time_tag = False)
    logger.save('I4D_denoiser/trainable_mask_I4D', trainable_mask, time_tag = False)
    logger.save('I4D_denoiser/classes', classes, time_tag = False)

    logger(f'There are {data4D_noisy[trainable_mask>0].sum()} electrons'
           ' in this dataset.')

    noisy_PACBED_in = data_gen_I4D.noisy_PACBED.copy() / data_gen_I4D.n_pts
    noisy_STEM_in = data_gen_I4D.noisy_mu.copy()
    
    if hyps_I4D['use_mu_eaxct']:
        mu_eaxct = noisy_STEM_in[edgew:-edgew, edgew:-edgew].ravel()
        torchModel_I4D.mu_eaxct = torch.from_numpy(mu_eaxct.copy()).float().cuda()
    
    noisy_PACBED_loss = noisy_PACBED_in / data_gen_I4D.n_pts
    noisy_STEM_loss = noisy_STEM_in / PACBED_mask.sum()
    criterion_I4D = STEM4D_PoissonLoss_FnormLoss(
        mask_backprop = torch.from_numpy(PACBED_mask).float().to(device),
        x_in_max = 1000,
        device = 'cuda',
        output_stabilizer = 1e-6,
        noisy_PACBED = torch.from_numpy(noisy_PACBED_loss).float().to(device),
        noisy_mSTEM = torch.from_numpy(
            noisy_STEM_loss.ravel()).float().to(device),
        PAC_loss_factor = hyps_I4D['PAC_loss_factor'],
        mSTEM_loss_factor = hyps_I4D['mSTEM_loss_factor'],
        )
    if hyps_I4D['test_mode']:
        torchModel_I4D.dgen = data_gen_I4D
    
    
    learning_rate_avoid_nan = 1e-6
    momentum_avoid_nan = 1e-7 
    torch_handler_I4D = nn_from_torch(
        data_generator = data_gen_I4D,
        torchModel = torchModel_I4D,
        lossFunc = criterion_I4D,
        device = device,
        logger = logger,
        learning_rate = learning_rate_avoid_nan,
        momentum = momentum_avoid_nan,
        pass_indices_to_model = True,
        fix_during_infer = True,
        test_mode = hyps_I4D['test_mode'])    
    
    perv_loss = np.nan
    Elapsed_time = 0
    n_epochs = 1
    
    n_refine_steps = hyps_I4D['n_refine_steps']

    if n_refine_steps > 1:
        refine_step_list = np.arange(1, n_refine_steps)
    else:
        refine_step_list = [1]
    for refine_step in refine_step_list:
        if hyps_I4D['rejection_ratio_list'] is None:
            rejection_ratio = 0
        else:
            assert len(refine_step_list) <= len(hyps_I4D['rejection_ratio_list']), \
                'if not None, the length of rejection_ratio_list should be the same as n_refine_steps.'
            rejection_ratio = hyps_I4D['rejection_ratio_list'][refine_step]
        
        if hyps_I4D['reset_on_refine'] & (not hyps_I4D['test_mode']):
            torch_handler_I4D.reset()
            torch_handler_I4D.update_learning_rate(learning_rate_avoid_nan)
            torch_handler_I4D.update_momentum(momentum_avoid_nan)

        n_ksweeps = hyps_I4D['n_ksweeps']
        if refine_step == n_refine_steps - 1:
            n_ksweeps = hyps_I4D['n_ksweeps_last']

        for kcnt in range(n_ksweeps):
            ETA_base = Elapsed_time * (
                (n_ksweeps - kcnt) + n_ksweeps * (len(refine_step_list) - refine_step))

            if kcnt == 0:
                init_I4D(noisy_STEM_in, None,
                         logger, data_gen_I4D, torch_handler_I4D, 
                         hyps_I4D['infer_size_I4D'])
            else:
                n_epochs = hyps_I4D['n_epochs']
                if (not hyps_I4D['test_mode']):
                    torch_handler_I4D.update_learning_rate(hyps_I4D['learning_rate'])
                    torch_handler_I4D.update_momentum(hyps_I4D['learning_momentum'])

            if(not include_training):
                kcnt = n_ksweeps    #avoid error
                
            criterion_I4D.LAGMUL = criterion_I4D_LAGMUL(kcnt, n_ksweeps)
            logger(f'criterion_I4D.LAGMUL: {criterion_I4D.LAGMUL}')
        
            if(include_training):
                DATOS_sampler_I4D = DATOS(
                    trainable_inds.shape[0],
                    classes = classes,
                    n_segments  = hyps_I4D['n_segments'],
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
            
            ferrs, data4D_denoised = infer_I4D(
                torch_handler_I4D, data_gen_I4D, logger, 
                hyps_I4D['infer_size_I4D'])
            
            print(f'data4D_denoised: {data4D_denoised.shape}')
            
            if rejection_ratio:
                for patt in data4D_denoised:
                    patt[patt < np.percentile(patt.ravel(), rejection_ratio)] = 0
            
            logger.imshow('I4D_denoiser/fitting_errors/ferrs',
                          data_gen_I4D.reconstruct2D(ferrs))

            data4D_denoised = data_gen_I4D.reconstruct4D(data4D_denoised)
            logger(f'data4D_denoised: {data4D_denoised.shape}')
            frame = mcemtools.data4D_to_frame(
                data4D_denoised[edgew:n_show+edgew, edgew:n_show+edgew])
            logger.imshow('I4D_denoiser/sample_denoised/denoised', frame)

            com_x, com_y = mcemtools.centre_of_mass_4D(
                data4D_denoised[edgew:-edgew, edgew:-edgew])
            logger.imshow('I4D_denoiser/com_complex/com',
                           com_x + 1j * com_y, cmap = 'complex')
            logger.imshow('I4D_denoiser/com_xy/com',
                           com_x + 1j * com_y, cmap = 'real_imag')
            
            noisy_STEM, noisy_PACBED  = mcemtools.sum_4D(
                data4D_denoised[edgew:-edgew, edgew:-edgew])
            Ne_estimated = noisy_STEM.mean()
            logger(f'estimated Ne per probe position: {Ne_estimated}')
            logger.imshow(
                'I4D_denoiser/I4D_denoised_STEM/final_STEM_noisy', noisy_STEM)
            logger.imshow(
                'I4D_denoiser/I4D_denoised_PACBED/final_noisy_PACBED', noisy_PACBED)
            
            if not include_training:
                break
        
        if len(refine_step_list) > 1:
            logger('Making a diffused input dataset')
            beta = refine_step/(len(refine_step_list) + 1)
            logger(f'The combination weight is: {beta}')
            
            if(log_denoised_every_sweep):
                logger.save('I4D_denoiser/I4D_denoised_inter/denoised', data4D_denoised)
            
            data4D_noisy_diffused = data4D_noisy.copy()
            data4D_noisy_diffused[edgew : -(edgew), edgew : -(edgew)] *= beta
            data4D_noisy_diffused[edgew : -(edgew), edgew : -(edgew)] += \
                data4D_denoised[edgew:-edgew, edgew:-edgew].copy() * (1 - beta)
            data4D_noisy_diffused = data4D_noisy_diffused.astype('float32')
            data_gen_I4D.update(data4D_noisy_diffused, 
                                update_label = hyps_I4D['refine_by_labels'])
            
            if hyps_I4D['refine_by_labels']:
                logger('labels are diffused too.')
                recon = data_gen_I4D.reconstruct4D(data_gen_I4D.Y_label)
                frame = mcemtools.data4D_to_frame(
                    recon[edgew:n_show+edgew, edgew:n_show+edgew])
                logger.imshow('I4D_denoiser/sample/data4D_diffused', frame)
            
            logger('training dataset is modified with diffused input')
            del data4D_noisy_diffused
    
            hyps_I4D['learning_rate'] *= hyps_I4D['learning_rate_decay']
            hyps_I4D['learning_momentum'] *= hyps_I4D['learning_momentum_decay']
            logger(hyps_I4D)
            
    logger.save('I4D_denoiser/I4D_denoised/denoised', data4D_noisy)
    
    frame_denoised = mcemtools.data4D_to_frame(data4D_noisy[
        edgew:n_show - edgew, edgew:n_show - edgew])
    
    logger.imshow('I4D_denoiser/canvases/denoised', frame_denoised)
    
    logger_dir = logger.log_dir
    logger(f'Find the results in: {logger_dir}')
    logger('~~~~~~~~ denoising finished ~~~~~~~~~')
    
    if hyps_I4D['test_mode']:
        logger('NOTE: NO TRAINING WAS CARRIED ON AS THIS IS TEST MODE')
        
    del logger
    
    return logger_dir
