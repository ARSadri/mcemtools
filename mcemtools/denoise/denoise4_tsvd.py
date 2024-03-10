import time
import numpy as np
import matplotlib.pyplot as plt
from lognflow import lognflow, logviewer, plt_colorbar

import mcemtools

def denoise4_tsvd(
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
    use_pre_denoised_STEM,
    STEM_denoiser_model_type,
    rank_info):

    FLAG_train_STEM, FLAG_train_I4D = include_training
    include_training = any(include_training)
    pretrained_fpath_STEM, pretrained_fpath_I4D = pretrained_fpaths_tuple
    
    if(include_training):
        if(not log_exist_ok):
            logged = logviewer(logs_root)
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

    logged_ref     = logviewer(ref_dir)
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
    logger(f'rank_info: {rank_info}')
    
    from mcemtools.tensor_svd import scree_plots
    screes = scree_plots(data4D_noisy.reshape(data4D_noisy.shape[0],data4D_noisy.shape[1],data4D_noisy.shape[2]*data4D_noisy.shape[3]))
    plt.plot(screes[0], '-*', label = 'x direction')
    plt.plot(screes[1], '-*', label = 'y direction')
    plt.plot(screes[2], '-*', label = 'pixels')
    plt.legend()
    logger.log_plt('screes', dpi = 1000)
    
    logger.log_single('screes_0', screes[0])
    logger.log_single('screes_1', screes[1])
    logger.log_single('screes_2', screes[2])
    
    err_all = []
    _nx_ny_npix_all = []
    for _n_x in rank_info.n_x:
        for _n_y in rank_info.n_y:
            for _n_pix in rank_info.n_pix:
                print(f'{_n_x}/{rank_info.n_x[-1]}',
                      f'{_n_y}/{rank_info.n_y[-1]}',
                      f'{_n_pix}/{rank_info.n_pix[-1]}')
                denoised = mcemtools.tensor_svd.tensor_svd_denoise(
                    data4D_noisy, rank = (_n_x, _n_y, _n_pix))
                denoised = denoised.reshape(*data4D_noisy_shape)
                current_Err = ((denoised - data4D_nonoise)**2).mean()**0.5
                err_all.append(current_Err)
                _nx_ny_npix_all.append([_n_x, _n_y, _n_pix])
    err_all = np.array(err_all)
    logger(f'ranks are: {rank_info.n_pix}')
    logger(f'errors vs ranks are: {err_all}')
    ind = np.argmin(err_all)
    logger(f'best rank is {ind}: {_nx_ny_npix_all[ind]}')
    _n_x, _n_y, _n_pix = _nx_ny_npix_all[ind]

    denoised = mcemtools.tensor_svd.tensor_svd_denoise(
            data4D_noisy, rank = (_n_x, _n_y, _n_pix))
    denoised = denoised.reshape(*data4D_noisy_shape)
    logger.log_single('I4D_denoiser/I4D_denoised/denoised', denoised)
    frame_denoised = mcemtools.data4D_to_frame(denoised)
    frame_nonoise = mcemtools.data4D_to_frame(data4D_nonoise)
    frame_noisy = mcemtools.data4D_to_frame(data4D_noisy)
    logger.log_single('I4D_denoiser/canvases/denoised', frame_denoised)
    logger.log_single('I4D_denoiser/canvases/nonoise', frame_nonoise)
    logger.log_single('I4D_denoiser/canvases/noisy', frame_noisy)
    logger('--\|/'*16)
    logger_dir = logger.log_dir
    
    del logger
    
    return logger_dir
