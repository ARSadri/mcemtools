#!/usr/bin/env python

import pathlib
import numpy as np
from   addict import Dict
from   lognflow import lognflow, logviewer, plt_imshow
import matplotlib.pyplot as plt

import mcemtools
from mcemtools import denoise4net
from mcemtools.data import np_random_poisson_no_zeros

def criterion_I4D_LAGMUL(kcnt, n_ksweeps): 
    k_ratio = kcnt/n_ksweeps
    #LAGMUL: 1 is for Gaussian and 0 is for Poisson
    if(  (0   < k_ratio) | (k_ratio <= 2/3)):
        LAGMUL = 1
    elif((2/3 < k_ratio) | (k_ratio <=   1)):
        LAGMUL = 0
    return LAGMUL

def make_problem(FLAG_make_problem,
                 exp_noisy_fpath, 
                 exp_nonoise_fpath,
                 ref_dir, 
                 problem_args):

    fpath = ref_dir / 'data4D_shape.npy'
    if fpath.is_file() & (not FLAG_make_problem):
        return fpath
    logger = lognflow(log_dir = ref_dir)
    data4D_noisy_diffused = None
    if(0):
        ...
    elif(0):
        n_probes = problem_args['n_probes']
        np.random.seed(problem_args.noise_seed)
        
        if (problem_args['use_ref_nonoise']):
            data4D_nonoise = np.load(problem_args['ref_nonoise_fpath'])
            print('using ref nonoise')
            print(problem_args['ref_nonoise_fpath'])
            print(f'original data4D_nonoise.shape: {data4D_nonoise.shape}')
        else:
            data4D_nonoise = np.load(exp_nonoise_fpath)
            if n_probes>0:
                data4D_nonoise = data4D_nonoise[:n_probes, :n_probes]
            data4D_nonoise = mcemtools.masking.crop_or_pad(
                data4D_nonoise, (data4D_nonoise.shape[0],
                                 data4D_nonoise.shape[1],
                                 128, 128))    
            data4D_nonoise = mcemtools.bin_4D(data4D_nonoise, n_pix_in_bin = 2)
            data4D_nonoise = mcemtools.bin_4D(data4D_nonoise, n_pix_in_bin = 2)
            data4D_nonoise *= problem_args.Ne
        
        if problem_args['use_ref_noisy']:
            data4D_noisy = np.load(problem_args['ref_noisy_fpath'])
            print('using ref noisy')
            print(problem_args['ref_noisy_fpath'])
        else:
            data4D_noisy = np_random_poisson_no_zeros(data4D_nonoise)
        
        if problem_args['use_old_denoised']:
            data4D_noisy_diffused = data4D_noisy.copy()
            print('using old denoised')
            print(f'problem_args.ref_denoised_fpath: {problem_args.ref_denoised_fpath}')
            print(f'problem_args.denprog: {problem_args.denprog}')
            denoised = np.load(problem_args['ref_denoised_fpath'])
            
            factor = problem_args.denprog
                        
            data4D_noisy_diffused = data4D_noisy_diffused * (
                1 - factor) + denoised * factor
        
            print('data4D_noisy_diffused made')        
    elif(0):
        n_probes = problem_args['n_probes']
        np.random.seed(problem_args.noise_seed)
        
        if (problem_args['use_ref_nonoise']):
            data4D_nonoise = np.load(problem_args['ref_nonoise_fpath'])
            print('using ref nonoise')
            print(problem_args['ref_nonoise_fpath'])
            print(f'original data4D_nonoise.shape: {data4D_nonoise.shape}')
        else:
            data4D_nonoise = np.load(exp_nonoise_fpath)
            if n_probes>0:
                data4D_nonoise = data4D_nonoise[:n_probes, :n_probes]
            data4D_nonoise = mcemtools.bin_4D(data4D_nonoise, n_pix_in_bin = 2)
            data4D_nonoise = mcemtools.bin_4D(data4D_nonoise, n_pix_in_bin = 2)
            data4D_nonoise = mcemtools.bin_4D(data4D_nonoise, n_pix_in_bin = 2)
            # data4D_nonoise = mcemtools.bin_4D(data4D_nonoise, n_pix_in_bin = 2)
            data4D_nonoise *= problem_args.Ne
        
        if problem_args['use_ref_noisy']:
            data4D_noisy = np.load(problem_args['ref_noisy_fpath'])
            print('using ref noisy')
            print(problem_args['ref_noisy_fpath'])
        else:
            data4D_noisy = np_random_poisson_no_zeros(data4D_nonoise)
        
        if problem_args['use_old_denoised']:
            data4D_noisy_diffused = data4D_noisy.copy()
            print('using old denoised')
            print(f'problem_args.ref_denoised_fpath: {problem_args.ref_denoised_fpath}')
            print(f'problem_args.denprog: {problem_args.denprog}')
            denoised = np.load(problem_args['ref_denoised_fpath'])
            
            factor = problem_args.denprog
                        
            data4D_noisy_diffused = data4D_noisy_diffused * (
                1 - factor) + denoised * factor
        
            print('data4D_noisy_diffused made')
        
    elif(0):
        logged = logviewer(exp_nonoise_fpath.parent)
        data4D_nonoise = logged.get_single(exp_nonoise_fpath.name)['I4D_no_noise']
        data4D_nonoise = data4D_nonoise.swapaxes(1,2).swapaxes(2,3).swapaxes(0,1).swapaxes(1,2)
        # data4D_nonoise = data4D_nonoise[:, :, :-1, :-1]
        data4D_nonoise = np.tile(data4D_nonoise, (4, 4, 1, 1))
        data4D_nonoise *= problem_args.Ne
        data4D_noisy = np_random_poisson_no_zeros(data4D_nonoise)
        
    elif(0):
        data4D_nonoise = np.load(
            exp_nonoise_fpath.parent / 'nonoise_64_kernel.npy')
        data4D_noisy = np.load(
            exp_nonoise_fpath.parent / 'denoised_64_kernel.npy')
        
    elif(0):
        np.random.seed(problem_args.noise_seed) 

        if (problem_args['use_ref_nonoise']):
            data4D_nonoise = np.load(problem_args['ref_nonoise_fpath'])
            print('using ref nonoise')
            print(problem_args['ref_nonoise_fpath'])
        else:
            data4D_nonoise = np.load(exp_nonoise_fpath)
            _data4D_nonoise = mcemtools.bin_4D(data4D_nonoise, 1, 2)
            _data4D_nonoise = mcemtools.bin_4D(_data4D_nonoise, 1, 2)
            max_x, max_y, n_rr, m_cc = _data4D_nonoise.shape
            n_probes = (problem_args['n_probes'], problem_args['n_probes'])
            skip = (8, 8)
            probe_noise_std = 0  # remember that 4.129 / 128 = 0.0322578125
            data4D_nonoise = np.zeros(
                (n_probes[0], n_probes[1], n_rr, m_cc), dtype='float32')
            for rcnt in range(0, data4D_nonoise.shape[0], skip[0]):
                for ccnt in range(0, data4D_nonoise.shape[0], skip[1]):
    
                    inds_c, inds_r = np.meshgrid(
                        np.arange(skip[0]), np.arange(skip[1]))
                    inds_r = (inds_r * max_x / skip[0] + probe_noise_std * (
                        np.random.randn(*inds_r.shape) )).astype('int')
                    inds_c = (inds_c * int(max_y / skip[1]) + probe_noise_std * (
                        np.random.randn(*inds_c.shape) )).astype('int')
                    inds_r[inds_r >= max_x] = 2 * max_x - inds_r[inds_r >= max_x] - 1
                    inds_c[inds_c >= max_y] = 2 * max_y - inds_c[inds_c >= max_y] - 1
                    inds_r[inds_r <    0] =         - inds_r[inds_r < 0]
                    inds_c[inds_c <    0] =         - inds_c[inds_c < 0]
                    
                    data4D_nonoise[rcnt:rcnt+skip[0], ccnt:ccnt+skip[1]] = \
                        _data4D_nonoise[inds_r, inds_c].copy()   
            
            # data4D_nonoise = mcemtools.bin_4D(data4D_nonoise, 1, 2)
            # mcemtools.markimage(data4D_nonoise.sum((0,1)).squeeze()); exit()
            # cr = data4D_nonoise.shape[2]//2
            # cc = data4D_nonoise.shape[3]//2
            # data4D_nonoise = data4D_nonoise[:, :, cr-32:cr+32, cc-32:cc+32]
            # data4D_nonoise = np.tile(data4D_nonoise, (64, 64, 1, 1))
            # data4D_nonoise = data4D_nonoise[:problem_args.ps*16*4:problem_args.ps,
            #                                 :problem_args.ps*16*4:problem_args.ps]
            data4D_nonoise *= problem_args.Ne
        
        if problem_args['use_ref_noisy']:
            data4D_noisy = np.load(problem_args['ref_noisy_fpath'])
            print('using ref noisy')
            print(problem_args['ref_noisy_fpath'])

        else:
            data4D_noisy = np_random_poisson_no_zeros(data4D_nonoise)
        
        if problem_args['use_old_denoised']:
            print('using ref denoised')
            print(problem_args['denoised_fpath'])
            print(problem_args.denprog)
            denoised = np.load(problem_args['denoised_fpath'])
            
            # data4D_noisy = noisy.copy()
            data4D_noisy = noisy * (
                1 - problem_args.denprog) + denoised * problem_args.denprog
            
        
    elif(1):
        n_probes = problem_args['n_probes']
        logged = logviewer(exp_nonoise_fpath.parent)
        data4D_nonoise = logged.get_single(
            exp_nonoise_fpath.name, suffix = '.npy', verbose = True)
        
        data4D_nonoise = data4D_nonoise[:n_probes, :n_probes]
        # data4D_nonoise = mcemtools.bin_4D(data4D_nonoise, 1, 2)
        data4D_nonoise *= problem_args.Ne
        data4D_noisy = np_random_poisson_no_zeros(data4D_nonoise)
        n_x, n_y, n_r, n_c = data4D_noisy.shape
        # data4D_nonoise = data4D_nonoise[
        #     :, :, n_r//2 - 8: n_r//2 + 8, 
        #           n_c//2 - 8: n_c//2 + 8]
        # data4D_noisy = data4D_noisy[
        #     :, :, n_r//2 - 8: n_r//2 + 8, 
        #           n_c//2 - 8: n_c//2 + 8]
        # print(f'data4D_nonoise.shape: {data4D_nonoise.shape}')
        # if exp_noisy_fpath is not None:
        #     denoised = np.load(exp_noisy_fpath)
        #     noisy = np.load(problem_args['ref_noisy_fpath'])
        #     data4D_noisy = noisy * (1 - problem_args[
        #         'denprog']) + denoised * problem_args.denprog

    elif(0):
        logged = logviewer(exp_nonoise_fpath.parent)
        
        def read_raw(fpath):
            return mcemtools.load_raw(fpath, (256, 256), (128, 128))
        
        data4D_nonoise = logged.get_single(
            exp_nonoise_fpath.name, read_func = read_raw)
        data4D_nonoise[data4D_nonoise < 30] = 0
        data4D_nonoise = data4D_nonoise.astype('float32') / 575.0
        
        # mcemtools.viewer_4D(data4D_nonoise, logger = print); exit()
        # data4D_nonoise = data4D_nonoise[:, :, 64-24:64+24, 64-24:64+24]
        logged = logviewer(exp_noisy_fpath.parent)
        data4D_noisy = logged.get_single(
            exp_noisy_fpath.name, read_func = read_raw)
        data4D_noisy[data4D_noisy < 30] = 0
        data4D_noisy = data4D_noisy.astype('float32') / 575.0
        # data4D_noisy = data4D_noisy[:, :, 64-24:64+24, 64-24:64+24]

        data4D_nonoise = data4D_nonoise[83:179, 38:144]
        data4D_noisy = data4D_noisy[83:179, 38:144]

        data4D_noisy = mcemtools.bin_4D(data4D_noisy, 1, 2)
        data4D_nonoise = mcemtools.bin_4D(data4D_nonoise, 1, 2)

        # mcemtools.viewer_4D(data4D_noisy)
        # mcemtools.viewer_4D(data4D_nonoise)
        # exit()

        print(f'Ne: {data4D_noisy.sum((2,3)).mean()}')

    elif(0):
        data4D_noisy = mcemtools.load_raw(
            exp_fpath, scanSize = (256,256), detSize = (128,128))
        data4D_noisy = data4D_noisy.astype('float32')/575.0
        # mcemtools.markimage(data4D_noisy.sum(1).sum(0).squeeze()), exit()

        if(1):
            data4D_nonoise = data4D_noisy.copy()
        else:
            data4D_nonoise_fpath = pathlib.Path(
                r'./../mcemtools_data/Alireza_alcu/a2_14p5Mx_194pmrad_2500/scan_x256_y256.raw')
            data4D_nonoise = mcemtools.load_raw(
               data4D_nonoise_fpath, scanSize = (256,256), detSize = (128,128))
            data4D_nonoise = data4D_nonoise.astype('float32')/575.0
            # mcemtools.markimage(data4D_nonoise.sum(1).sum(0).squeeze()), exit()
            data4D_nonoise = data4D_nonoise[..., 65-32:65+32, 64-32:64+32]
    elif(0):
        logged = logviewer(exp_nonoise_fpath.parent)
        data4D_nonoise = logged.get_single(exp_nonoise_fpath.name
                                           ).astype('float32')
        data4D_noisy = data4D_nonoise.copy()
    elif(0):
        logged = logviewer(exp_nonoise_fpath.parent)
        data4D_nonoise = logged.get_single(
            exp_nonoise_fpath.name).astype('float32')
        data4D_nonoise[:, :, data4D_nonoise.sum((0, 1)) > 5000] = 0
        
        
        # mcemtools.markimage(data4D_nonoise.sum((0, 1))); plt.show(); exit()
        
        # plt.hist(data4D_nonoise.ravel(), 1000); plt.show()
        # exit()
        
        # com_x, com_y = mcemtools.centre_of_mass_4D(data4D_nonoise)
        # plt_imshow(com_x + 1j * com_y, cmap = 'complex'); plt.show(); exit()
        
                   
        data4D_nonoise = data4D_nonoise / 32
        side_hlength = 96
        data4D_nonoise = data4D_nonoise[..., 123 - side_hlength : 123 + side_hlength, 
                                             125 - side_hlength : 125 + side_hlength]
        # data4D_nonoise = data4D_nonoise[..., 126 - side_hlength : 126 + side_hlength, 
        #                                      140 - side_hlength : 140 + side_hlength]
        
        # mcemtools.viewer_4D(data4D_nonoise);exit()
        
        # data4D_nonoise = mcemtools.masking.crop_or_pad(
        #     data4D_nonoise, (data4D_nonoise.shape[0], data4D_nonoise.shape[1],
        #                      128, 128))
        data4D_nonoise = mcemtools.bin_4D(data4D_nonoise, n_pix_in_bin = 2)
        data4D_nonoise = mcemtools.bin_4D(data4D_nonoise, n_pix_in_bin = 2)
        # data4D_nonoise = mcemtools.bin_4D(data4D_nonoise, n_pix_in_bin = 2)
        
        # mask2d = mcemtools.annular_mask(data4D_nonoise.sum((0, 1)).shape,
        #                                 radius = 3200, in_radius = 10)
        # import RobustGaussianFittingLibrary as rgflib
        # bck, prof = rgflib.fitBackgroundRadially(
        #    data4D_nonoise.sum((0, 1)), optIters = 1, return_vecMP = True)
        # import matplotlib.pyplot as plt
        # plt.plot(prof[0]), plt.show(); exit()
        data4D_noisy = data4D_nonoise.copy()
        
    problem_PACBED_mask = True
    if problem_PACBED_mask:
        PACBED_mask = mcemtools.annular_mask((data4D_noisy.shape[2], 
                                              data4D_noisy.shape[3]), 
                                              radius = 16,
                                              in_radius = None)
        logger.log_single('premask_noisy', data4D_noisy, time_tag = False)
        logger.log_single('premask_nonoise', data4D_noisy, time_tag = False)
        data4D_noisy[..., PACBED_mask == 0] = 0
        data4D_nonoise[..., PACBED_mask == 0] = 0

    print(f'data4D_noisy.shape:{data4D_noisy.shape}')
    print(f'data4D_nonoise.shape:{data4D_nonoise.shape}')
    
    logger.log_single('noisy', data4D_noisy, time_tag = False)
    logger.log_single('nonoise', data4D_nonoise, time_tag = False)
    if data4D_noisy_diffused is not None:
        if problem_PACBED_mask:
            data4D_noisy_diffused[..., PACBED_mask == 0] = 0
        logger.log_single('data4D_noisy_diffused', 
                          data4D_noisy_diffused, time_tag = False)

    STEM, PACBED = mcemtools.sum_4D(data4D_noisy)
    logger.log_imshow('noisy_STEM', STEM, time_tag = False)
    logger.log_imshow('noisy_PACBED', PACBED, time_tag = False)

    STEM, PACBED = mcemtools.sum_4D(data4D_nonoise)
    logger.log_imshow('nonoise_STEM', STEM, time_tag = False)
    logger.log_imshow('nonoise_PACBED', PACBED, time_tag = False)
    
    com_x, com_y = mcemtools.centre_of_mass_4D(data4D_noisy)
    logger.log_imshow(
        'noisy_CoM',com_x + 1j * com_y, cmap = 'complex', time_tag = False)

    com_x, com_y = mcemtools.centre_of_mass_4D(data4D_nonoise)
    logger.log_imshow(
        'nonoise_CoM',com_x + 1j * com_y, cmap = 'complex', time_tag = False)
    
    #This must be the last thing this function does  
    return logger.log_single(
        'data4D_shape.npy', np.array(data4D_noisy.shape),time_tag = False)
    
def get_problem_settings(problem_args):
    ref_dir = problem_args.ref_dir

    fpath = ref_dir / 'data4D_shape.npy'
    assert fpath.is_file(), 'You should have made the problem first'
    data4D_shape = np.load(fpath)
    n_x, n_y, n_r, n_c = data4D_shape

    hyps_STEM = Dict(
        learning_rate = 1e-5,
        learning_momentum = 1e-5,
        mbatch_size = 4,
        n_epochs = 8,
        n_segments = 1,
        win_length = 32,
        skip_length = 1,
        n_kernels = 4,
        n_kSweeps = 4,
        infer_size_STEM = 50,
        mask_rate = 0.9,
        )
    
    hyps_CoM = Dict(
        learning_rate = 1e-4,
        learning_momentum = 1e-5,
        mbatch_size = 4,
        n_epochs = 8,
        n_segments = 1,
        win_length = 32,
        skip_length = 1,
        n_kernels = 4,
        n_kSweeps = 4,
        infer_size_CoM = 50,
        mask_rate = 0.5,
        denoise_amp_angle = True,
        )
    
    n_refine_steps = 2
    hyps_I4D = Dict(
        n_prob = 3,                                     ###
        learning_rate = problem_args.learning_rate,     ####
        learning_momentum = 1e-4,                       #
        mbatch_size = problem_args.mbatch_size,         #
        n_epochs = 8,                                   ##
        n_segments = 1,                                 #                          
        n_kernels = problem_args.n_kernels,             #####
        infer_size_I4D = 50,                            #
        PAC_loss_factor = 0.015,                        ##
        mSTME_loss_factor = 0.005,                      ##
        n_refine_steps = n_refine_steps,
        n_ksweeps = 4,
        n_ksweeps_last = 4,
        learning_rate_decay = 0.1**(1/(n_refine_steps - 1)),
        learning_momentum_decay = 0.1**(1/(n_refine_steps - 1)),
        refine_by_labels = False,
        reset_on_refine = True,
        test_mode = False,
        use_mu_eaxct = False,
        )
    
    #### trainable area in STEM and PACBED #####################################
    # mcemtools.viewer_4D(np.load(logs_root / 'ref/noisy.npy'), logger = print)
    PACBED_mask = mcemtools.annular_mask((data4D_shape[2], 
                                          data4D_shape[3]), 
                                          radius = 16,
                                          in_radius = None)
    denoise_STEM_mask = mcemtools.annular_mask((data4D_shape[2],
                                                data4D_shape[3]),
                                                radius = None,
                                                in_radius = None)
    if(0):
        training_area_r_start = 0
        training_area_r_end = 128    #will be summed up with WinX
        training_area_c_start = 0
        training_area_c_end = 128    #will be summed up with WinX
        trainable_area_STEM2D = np.zeros((n_x, n_y))
        trainable_area_STEM2D[
            training_area_r_start : training_area_r_end + hyps_STEM['win_length'],
            training_area_c_start : training_area_c_end + hyps_STEM['win_length']
            ] = 1
    elif(0):
        data4D_noisy = np.load(ref_dir / 'noisy.npy')
        PACBED_mask_4D = mcemtools.mask2D_to_4D(PACBED_mask, data4D_noisy.shape)
        noisy_STEM, _ = mcemtools.sum_4D(data4D_noisy, PACBED_mask_4D)
        trainable_area_STEM2D[
            noisy_STEM < np.percentile(noisy_STEM.ravel(), 80)] = 1
    elif(0):
        trainable_area_STEM2D = 1 - np.load(ref_dir / 'mask.npy')
    else:
        trainable_area_STEM2D = np.ones((n_x, n_y))
        
    if(0):
        training_area_r_start = 0
        training_area_r_end = 128    #will be summed up with WinX
        training_area_c_start = 0
        training_area_c_end = 128    #will be summed up with WinX
        trainable_area_I4D = np.zeros((n_x, n_y))
        trainable_area_I4D[training_area_r_start : training_area_r_end,
                           training_area_c_start : training_area_c_end] = 1
    elif(0):
        data4D_noisy = np.load(ref_dir / 'noisy.npy')
        PACBED_mask_4D = mcemtools.mask2D_to_4D(PACBED_mask, data4D_noisy.shape)
        noisy_STEM, _ = mcemtools.sum_4D(data4D_noisy, PACBED_mask_4D)
        trainable_area_I4D[
            noisy_STEM < np.percentile(noisy_STEM.ravel(), 80)] = 1
    elif(0):
        noisy_STEM = np.load(ref_dir / 'denoised_STEM.npy')
        trainable_area_I4D = np.zeros(noisy_STEM.shape)
        trainable_area_I4D[
            noisy_STEM < np.percentile(noisy_STEM.ravel(), 30)] = 1
        
        trainable_area_I4D = mcemtools.remove_islands_by_size(
            trainable_area_I4D, min_n_pix=10)
        import scipy.ndimage
        trainable_area_I4D = scipy.ndimage.binary_dilation(trainable_area_I4D)
        trainable_area_I4D = scipy.ndimage.binary_erosion(trainable_area_I4D)
        # trainable_area_I4D = 1 - trainable_area_I4D
    elif(0):
        trainable_area_I4D = 1 - np.load(ref_dir / 'mask.npy')
    else:
        trainable_area_I4D = np.ones((n_x, n_y))

    problem_settings = dict(
        hyps_STEM             = hyps_STEM,
        hyps_I4D              = hyps_I4D,
        hyps_CoM              = hyps_CoM,
        trainable_area_STEM2D = trainable_area_STEM2D,
        trainable_area_I4D    = trainable_area_I4D,
        PACBED_mask           = PACBED_mask,
        denoise_STEM_mask     = denoise_STEM_mask,
        repeat_by_scattering  = [2, 1],
        n_canvas_patterns     = 33,
        )

    return problem_settings

if __name__ == '__main__':

    data_dir = r'./../mcemtools_data/SrTiO3/elec_True_mag_False_ps_1.00_th_240_app_19_df_50.0'
    logged_data = logviewer(log_dir = data_dir)
    logs_root = pathlib.Path(r'./../mcemtools_logs/STO_16eA')
    
    exp_flist = []
    material_name_list = []

    more_files = logged_data.get_flist('data.npy')
    # more_files = logged_data.get_flist('Ti_pre1_drift.npy')
    for _ in more_files:
        exp_flist.append(_)
        material_name_list.append('STO_16eA')

    # more_files = logged_data.get_flist('SrTiO3_probe_spacings_alpha/pyms_n_03_ps_0.73_df_0.0_200A_app_*/data.npy')
    # for _ in more_files:
    #     exp_flist.append(_)
    #     material_name_list.append('SrTiO3')
    #
    # more_files = logged_data.get_flist('Y2Ti2O7_probe_spacings_alpha/pyms_n_03_ps_0.63_df_0.0_200A_app_*/data.npy')
    # for _ in more_files:
    #     exp_flist.append(_)
    #     material_name_list.append('Y2Ti2O7')


    # exp_flist = [exp_flist[0], exp_flist[10]]
    # material_name_list = [material_name_list[0], material_name_list[10]]
    # exp_flist = [exp_flist[-1]]
    # material_name_list = [material_name_list[-1]]
    # exp_flist = exp_flist[6:9]
    # material_name_list = material_name_list[6:9]
    print(exp_flist)
    
    #### files paths and general settings ######################################
    model_type                = 'UNET' # 'UNET' or 'TSVD'
    nn_init_try_max           = 5
    Ne_list                   = [1600]#16, 32, 64, 128, 256, 512]
    n_probes_list             = [32]
    n_kernels_list            = [16]#2, 4, 8, 16, 32, 64, 128]
    denoiser_progress_list    = np.arange(1)/1
    learning_rate_list        = [1e-3]
    mbatch_size_list          = [2]
    noise_seed_list           = [0]
    FLAG_make_problem         = False
    log_exist_ok              = True
    FLAG_train_I4D            = True
    use_classes_by_scattering = True
    use_repeat_by_scattering  = True
    log_denoised_every_sweep  = False

    STEM_denoiser_model_type  = 'UNET' # 'UNET' or 'TSVD'
    FLAG_train_STEM           = False
    use_pre_denoised_STEM     = False
    FLAG_denoise_STEM         = False
    denoise_STEM_for_I4D      = False

    CoM_denoiser_model_type  = 'UNET' # 'UNET' or 'TSVD'
    FLAG_train_CoM           = False
    use_pre_denoised_CoM     = False
    FLAG_denoise_CoM         = False
    denoise_CoM_for_I4D      = False

    if STEM_denoiser_model_type  == 'TSVD':
        rank_info = Dict(n_x = 28, n_y = 28, n_pix = 40)
    if CoM_denoiser_model_type  == 'TSVD':
        rank_info = Dict(n_x = 28, n_y = 28, n_pix = 40)
    if model_type == 'TSVD':
        denoiser_progress_list = [0.0]
        rank_info = Dict(n_x =   [56], #list(range(10, 41, 5)), 
                         n_y =   [56], #list(range(10, 31, 5)), 
                         n_pix = [64], #list(range(10, 80, 10))
                         )
    else:
        rank_info = None
    
    if(0):  # to start from a denprog
        
        exp_dir = logs_root / r'SrTiO3_pyms_n_01_ps_0.15_df_0.0_117A_app_21.4_Ne_8'
        logged_done = logviewer(exp_dir)
        ref_nonoise_fpath = logged_done.get_flist('integ_0.0/ref/nonoise.npy')[-1]
        ref_noisy_fpath =  logged_done.get_flist('integ_0.0/ref/noisy.npy')[-1]
        integ_cnt = 0
        ref_denoised_fpath = logged_done.get_flist(
            f'integ_0.{integ_cnt}/denoised4D_UNet_*/I4D_denoiser/I4D_denoised/denoised*.npy')[-1]
        denoiser_progress_list = denoiser_progress_list[integ_cnt + 1:]
        assert ref_denoised_fpath.is_file() & ref_nonoise_fpath.is_file() & \
               ref_noisy_fpath.is_file()
    
    #--------------------------------------------------------------------------#
    #### running experiments ###################################################
    assert exp_flist, 'There are no files'
    logs_root = pathlib.Path(logs_root)
    logger_for_all = lognflow(log_dir = logs_root)
    logger_for_all(f'There are {len(exp_flist)} files for analysis.')
    exp_noisy_fpath = None
    FLAG_make_problem_cpy = FLAG_make_problem
    for exp_nonoise_fpath, material_name in zip(exp_flist, material_name_list):
        for learning_rate in learning_rate_list:
            for mbatch_size in mbatch_size_list:
                for noise_seed in noise_seed_list:
                    for Ne in Ne_list:
                        for n_probes in n_probes_list:
                            for n_kernels in n_kernels_list:
                                for denprog in denoiser_progress_list:
                                    exp_dir = None
                                    exp_nonoise_dir = exp_nonoise_fpath.parent

                                    app = 19#float(exp_nonoise_dir.name.split('app_')[-1])
                                    # Ne = int(Ne * app**2)
                                    
                                    FLAG_make_problem = FLAG_make_problem_cpy
                                    _noise_seed = int(noise_seed)
                                    
                                    exp_name = ''
                                    exp_name += material_name + '_' + exp_nonoise_dir.name
                                    exp_name += f'_Ne_{Ne}'
                                    # exp_name += f'_n_k_{n_kernels}'
                                    # exp_name += f'/nprob_{n_probes}'
                                    # exp_name += f'/integ_{denprog}'
                                    
                                    logger_for_all(f'Processing: {exp_nonoise_dir}', flush = True)
                                    
                                    _logs_root = logs_root / f'{exp_name}'
                                    ref_dir = _logs_root / 'ref'

                                    pretrained_STEM_fpath = ref_dir / 'STEM_model.torch'
                                    pretrained_I4D_fpath = ref_dir / 'I4D_model.torch'
                                    
                                    if denprog == 0:
                                        ref_nonoise_fpath = ref_dir / 'nonoise.npy'
                                        ref_noisy_fpath = ref_dir / 'noisy.npy'
                                        ref_denoised_fpath = ''
                                        use_ref_noisy = False
                                        use_ref_nonoise = False
                                        use_old_denoised = False
                                    else:
                                        use_ref_noisy = True
                                        use_ref_nonoise = True
                                        use_old_denoised = True
                                    

                                    if denprog == denoiser_progress_list[-1]:
                                        log_denoised_every_sweep = True
                                    
                                    problem_args = Dict(Ne                      = Ne,
                                                        n_probes                = n_probes,
                                                        n_kernels               = n_kernels,
                                                        learning_rate           = learning_rate,
                                                        mbatch_size             = mbatch_size,
                                                        denoiser_progress_list  = denoiser_progress_list,
                                                        noise_seed              = _noise_seed,
                                                        denprog                 = denprog,
                                                        use_ref_nonoise         = use_ref_nonoise,
                                                        use_old_denoised        = use_old_denoised,
                                                        use_ref_noisy           = use_ref_noisy,
                                                        ref_nonoise_fpath       = ref_nonoise_fpath,
                                                        ref_denoised_fpath      = ref_denoised_fpath,
                                                        ref_noisy_fpath         = ref_noisy_fpath,
                                                        ref_dir                 = ref_dir)
                            
                                    experiment_settings = Dict(
                                        logs_root               = _logs_root,
                                        exp_name                = exp_name,
                                        ref_dir                 = ref_dir,
                                        include_training        = (FLAG_train_STEM,
                                                                   FLAG_train_I4D),
                                        pretrained_fpaths_tuple = (pretrained_STEM_fpath, 
                                                                   pretrained_I4D_fpath),
                                        FLAG_denoise_STEM       = FLAG_denoise_STEM,                          
                                        log_exist_ok            = log_exist_ok)
                                    
                                    more_settings = Dict(
                                        denoise_STEM_for_I4D      = denoise_STEM_for_I4D,
                                        log_denoised_every_sweep  = log_denoised_every_sweep,
                                        criterion_I4D_LAGMUL      = criterion_I4D_LAGMUL,
                                        use_classes_by_scattering = use_classes_by_scattering,
                                        use_repeat_by_scattering  = use_repeat_by_scattering,
                                        use_pre_denoised_STEM     = use_pre_denoised_STEM,
                                        STEM_denoiser_model_type  = STEM_denoiser_model_type,
                                        rank_info                 = rank_info,
                                        FLAG_train_CoM            = FLAG_train_CoM,      
                                        use_pre_denoised_CoM      = use_pre_denoised_CoM,
                                        FLAG_denoise_CoM          = FLAG_denoise_CoM,
                                        denoise_CoM_for_I4D       = denoise_CoM_for_I4D,
                                        CoM_denoiser_model_type   = CoM_denoiser_model_type,
                                        )
        
                                    nn_init_try_cnt = 0
                                    while nn_init_try_cnt < nn_init_try_max:
                                        make_problem(FLAG_make_problem,
                                                     exp_noisy_fpath,
                                                     exp_nonoise_fpath,
                                                     ref_dir, 
                                                     problem_args)
                                        problem_settings = get_problem_settings(
                                            problem_args = problem_args,
                                            )
                                        try:
                                            if(model_type == 'UNET'):
                                                exp_dir = mcemtools.denoise4net(
                                                    **experiment_settings,
                                                    **problem_settings,
                                                    **more_settings)
                                            if(model_type == 'TSVD'):
                                                exp_dir = mcemtools.denoise4_tsvd(
                                                    **experiment_settings,
                                                    **problem_settings,
                                                    **more_settings)
                                            
                                            nn_init_try_cnt = nn_init_try_max
                                        except Exception as e:
                                            nn_init_try_cnt += 1
                                            exp_dir = None
                                            FLAG_make_problem = True
                                            problem_args.noise_seed += 1
                                            print(e)
                                            raise e
                                            if(model_type == 'TSVD'):
                                                raise e
                                    assert exp_dir.is_dir(), f'failed exp: {exp_name}'
                                    logged_done = logviewer(exp_dir)
                                    ref_denoised_fpath = logged_done.get_flist(
                                        'I4D_denoiser/I4D_denoised/denoised*.npy')[-1]
                                    assert ref_denoised_fpath.is_file()
                                    logger_for_all(f'{ref_denoised_fpath} done!')