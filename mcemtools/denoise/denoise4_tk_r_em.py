import time
import numpy as np
import matplotlib.pyplot as plt
from lognflow.plt_utils import plt_colorbar
from lognflow import lognflow

import mcemtools

def tk_r_em(
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
    use_pre_denoised_STEM,
    STEM_denoiser_model_type,
    rank_info,
    FLAG_train_CoM,      
    use_pre_denoised_CoM,
    FLAG_denoise_CoM,    
    denoise_CoM_for_I4D,
    CoM_denoiser_model_type,):

    FLAG_train_STEM, FLAG_train_I4D = include_training
    include_training = any(include_training)
    pretrained_fpath_STEM, pretrained_fpath_I4D = pretrained_fpaths_tuple
    
    if(include_training):
        if(not log_exist_ok):
            logged = lognflow(logs_root)
            exp_list_names = logged.get_flist(
                f'denoised*/I4D_denoiser/I4D_denoised/denoised_*.npy')
            if len(exp_list_names)>0:
                return
        logger = lognflow(logs_root, log_dir_prefix = 'denoised4D_TSVD')
    else:
        if(not log_exist_ok):
            logger = lognflow(logs_root)
        else:
            logger = lognflow(log_dir = pretrained_fpath_I4D.parent.parent)    

    logged_ref     = lognflow(ref_dir)
    data4D_noisy   = logged_ref.get_single('noisy.npy')
    data4D_nonoise = logged_ref.get_single('nonoise.npy')
    # if use_pre_denoised_STEM:
    #     denoised_STEM  = logged_ref.get_single('denoised_STEM.npy')
    #     assert denoised_STEM is not None, 'set use_pre_denoised_STEM = False'
    
    data4D_shape = data4D_noisy.shape
    n_x, n_y, n_r, n_c = data4D_shape
    
    data4D_noisy[..., PACBED_mask==0] = 0
    data4D_nonoise[..., PACBED_mask==0] = 0

    mch = mcemtools.data4D_to_frame(data4D_nonoise[
            hyps_I4D['n_prob']//2:n_canvas_patterns - hyps_I4D['n_prob']//2,
            hyps_I4D['n_prob']//2:n_canvas_patterns - hyps_I4D['n_prob']//2])
    im = plt.imshow(mch); plt_colorbar(im)
    logger(logger.log_plt('I4D_denoiser/canvas_nonoise/nonoise', dpi = 4000))  
    
    mch = mcemtools.data4D_to_frame(data4D_noisy[
            hyps_I4D['n_prob']//2:n_canvas_patterns - hyps_I4D['n_prob']//2,
            hyps_I4D['n_prob']//2:n_canvas_patterns - hyps_I4D['n_prob']//2])
    im = plt.imshow(mch); plt_colorbar(im)
    logger(logger.log_plt('I4D_denoiser/canvas_noisy/noisy', dpi = 4000))        

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
    data4D_noisy_shape = data4D_noisy.shape
    
    logger(f'data4D_noisy.mean() : {data4D_noisy.mean()}')
    logger(f'data4D_nonoise.mean() : {data4D_nonoise.mean()}')
    logger(f'rank_info - model name: {rank_info}')
    
    from tk_r_em import load_network, load_sim_test_data

    r_em_nn = load_network(rank_info)
    r_em_nn.summary()
    data4D_noisy = data4D_noisy.swapaxes(1,2).swapaxes(2,3).swapaxes(0,1).swapaxes(1,2)
    data4D_noisy = data4D_noisy.reshape(n_r * n_c, n_x, n_y)
    data4D_noisy = np.expand_dims(data4D_noisy, -1)
    print(f'data4D_noisy.shape: {data4D_noisy.shape}')
    denoised = r_em_nn.predict(data4D_noisy, 64) # x.shape = (n_r * n_c, n_x, n_y, 1)
    print(f'denoised.shape: {data4D_noisy.shape}')
    denoised = denoised.reshape(n_r, n_c, n_x, n_y).swapaxes(1,2).swapaxes(2,3).swapaxes(0,1).swapaxes(1,2)
    
    logger.log_single('I4D_denoiser/I4D_denoised/denoised', denoised)
    frame_denoised = mcemtools.data4D_to_frame(denoised)
    frame_nonoise = mcemtools.data4D_to_frame(data4D_nonoise)
    frame_noisy = mcemtools.data4D_to_frame(data4D_noisy)
    logger.log_single('I4D_denoiser/canvases/denoised', frame_denoised)
    logger.log_single('I4D_denoiser/canvases/nonoise', frame_nonoise)
    logger.log_single('I4D_denoiser/canvases/noisy', frame_noisy)
    logger('-='*16)
    logger_dir = logger.log_dir
    
    del logger
    
    return logger_dir
