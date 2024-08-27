#!/usr/bin/env python
# coding: utf-8

import pathlib
import mcemtools
import matplotlib.pyplot as plt
from lognflow import plt_imshow, plt_imhist
# ### Phase contrast imaging of twisted bilayer MoS2
# 
# In this tutorial notebook, we will reconstruct conventional virtual detectors, and phase contrast images of MoS2. We will generate these reconstructions:
# * Virtual bright field image
# * Virtual dark field image
# * Differential phase contrast (DPC)
# * Parallax depth sectioning
# * ptychography (single slice)
# 
# We have already applied the experimental calibrations, so the only other piece of information required is the accelerating voltage of 80 kV.
# 
# ### Downloads
# 
# * [MoS2 twisted bilayer 4D-STEM dataset at bin 2](https://drive.google.com/file/d/1C3tXV5BXz0JXbmyu3wTu3NutuIYYsN8A/view?usp=sharing)
# 
# ### Acknowledgements
# 
# This tutorial was created by the py4DSTEM instructor team:
# - Colin Ophus (clophus@lbl.gov)
# - Georgios Varnavides (gvarnavides@berkeley.edu)
# - Stephanie Ribet (sribet@lbl.gov)
# 
# The 4D-STEM dataset used here was recorded by Zhen Chen in the Muller group, at Cornell University. The associated manuscript can be found here:
# 
# [Electron ptychography of 2D materials to deep sub-ångström resolution](https://doi.org/10.1038/s41586-018-0298-5)
# 
# Updated 2023 July 12

# In[1]:

import numpy as np
import py4DSTEM
from py4DSTEM.datacube import DataCube
print(py4DSTEM.__version__)

from lognflow import lognflow, logviewer, printprogress

class test:
    """
    This is a docstring no?
    """
    def __init__(self):
        ...
    
    @staticmethod
    def test2():
        ...
        
def py4dstem_ptycho_save(logger, ptycho, param_prefix = ''):    
    ptycho_save_list = [
        'error', 'error_iterations', 'object', 'object_cropped', 
        'object_fft', 'positions', 'probe', 'probe_centered', 'probe_fourier']
    for name in ptycho_save_list:
        try:
            if (len(param_prefix) > 0) & (param_prefix[-1] != '/'):
                param_prefix += '/'
            logger.log_single(f'{param_prefix}/{name}', 
                np.array(ptycho.__getattribute__(name)))
        except:
            print(f'could not log {name}')

if __name__ == '__main__':
    
    project_root = pathlib.Path(r'C:\Alireza\mcemtoolsProject')
    data_root    = project_root / 'mcemtools_data'
    logs_root    = project_root / 'mcemtools_logs'
        
    #original
    exp_log_root =    logs_root / 'pyms_ps_1.00_240A_df_50_Ne_16'
    ref_log_root = exp_log_root / 'pyms_ps_1_Ne_16_repcnt_0.0'
    ref_log_dir  = ref_log_root / 'ref'
    den_log_root = exp_log_root / 'pyms_ps_1_Ne_16_repcnt_0.5'
    den_log_dir  = den_log_root / 'pyms_ps_1_Ne_16_repcnt_0.5_1702938168.942208'
    
    #64
    exp_log_root =    logs_root / 'pyms_ps_1.00_240A_df_50_Ne_64'
    ref_log_root = exp_log_root / 'pyms_ps_1_Ne_64_repcnt_0.0'
    ref_log_dir  = ref_log_root / 'ref'
    den_log_root = exp_log_root / 'pyms_ps_1_Ne_64_repcnt_0.9'
    den_log_dir  = den_log_root / 'pyms_ps_1_Ne_64_repcnt_0.9_1701681072.7668705'
    
    #new 
    exp_log_root =    logs_root / r'STO_16eA_df_n50\STO_16eA_elec_True_mag_False_ps_0.98_th_240_app_19_df_50.0_Ne_16'
    ref_log_dir  = exp_log_root / 'ref'
    den_log_dir  = exp_log_root / 'denoised4D_UNet_1724588931.2537155'

    dataset_to_run_for_list = ['den', 'noi', 'non']
    dataset_to_run_for = dataset_to_run_for_list[0]
    
    n_prob_den = 3//2
    rejected_perc_list = [75]#, 30, 40, 50, 60, 70, 80, 90]
    threshed_perc = 100
    
    logger_ref = lognflow(log_dir = ref_log_dir)
    noisy = logger_ref.logged.get_single('noisy.npy')[n_prob_den:-n_prob_den, n_prob_den:-n_prob_den]
    nonoise = logger_ref.logged.get_single('nonoise.npy')[n_prob_den:-n_prob_den, n_prob_den:-n_prob_den]
    
    nonoise_frame = mcemtools.data4D_to_frame(nonoise[6:20, 6:20])

    def apply_STEM_a4d_to_b4d(data4d_a, data4d_b):
        stem = data4d_b.sum((2, 3))
        stem4d = np.tile(np.expand_dims(np.expand_dims(stem, -1), -1),
                         (1, 1, data4d_b.shape[2], data4d_b.shape[3]))
        data4d_b /= stem4d
        stem = data4d_a.sum((2, 3))
        stem4d = np.tile(np.expand_dims(np.expand_dims(stem, -1), -1),
                         (1, 1, data4d_b.shape[2], data4d_b.shape[3]))
        data4d_b *= stem4d
        return data4d_b
    
    ptycho_dir_name = 'ptycho'
    
    if(dataset_to_run_for == 'non'):
        logger = lognflow(log_dir = ref_log_dir/ 'ptycho_nonoise')
        data4d = nonoise.copy()
        rejected_perc_list = [0]
    if(dataset_to_run_for == 'noi'):
        logger = lognflow(log_dir = ref_log_dir/ 'ptycho_noisy')
        data4d = noisy.copy()
        rejected_perc_list = [0]

    # if(1):
    for rejected_perc in rejected_perc_list:#np.arange(0, 50, 5):
        if(dataset_to_run_for == 'den'):
            ptycho_dir_name = f'ptycho_rej_{rejected_perc}_max_{threshed_perc}'
            
            logger = lognflow(log_dir = den_log_dir, time_tag = False)
            denoised_bf = logger.logged.get_stack_from_names('I4D_denoiser/I4D_denoised/denoised*.npy')
            denoised_bf = logger.logged.get_stack_from_names('I4D_denoiser/I4D_denoised_inter/denoised*.npy')
            denoised_bf = denoised_bf[1:].mean(0)
            # denoised_bf = denoised_bf[-1]
            denoised_bf = denoised_bf[n_prob_den:-n_prob_den, n_prob_den:-n_prob_den]

            # den_frame = mcemtools.data4D_to_frame(denoised_bf[6:20, 6:20])
            # plt_imhist(den_frame);plt.show();exit()
            
            n_x, n_y, n_r, n_c = noisy.shape
    
            if 1 & (rejected_perc > 0):
                denoised = noisy.copy()
                # mask = mcemtools.annular_mask((n_r, n_c), radius = 15, in_radius = None)
                # denoised[:, :, mask == 1] = denoised_bf[:, :, mask == 1].copy()
        
                for xcnt in printprogress(range(n_x), title='thresholding patterns'):
                    for ycnt in range(n_y):
                        patt = denoised_bf[xcnt, ycnt].copy()
                        stat = np.percentile(patt[patt > 0], rejected_perc)
                        patt[patt < stat] = 0
                        stat = np.percentile(patt[patt > 0], threshed_perc)
                        patt[patt > stat] = stat
                        denoised_bf[xcnt, ycnt] = patt.copy()
                denoised[denoised_bf > 0] = denoised_bf[denoised_bf>0]
            else:
                denoised = denoised_bf
            
            # denoised_bf = apply_STEM_a4d_to_b4d(nonoise, denoised_bf)

            
            # mask = mcemtools.annular_mask((n_r, n_c), radius = 9, in_radius = 6)
            # denoised[:, :, mask == 1] = denoised_bf[:, :, mask == 1].copy()
            
            # # plt.plot(denoised_bf[:, :, mask == 1].ravel(), 
            # #          nonoise[:, :, mask == 1].ravel(), '*')
            #
            # vec1 = denoised_bf[:, :, mask == 1].ravel().copy()/2
            # vec2 = nonoise[:, :, mask == 1].ravel().copy()
            # logger.log_hist('test', [vec1, vec2], 1000, return_figure = True)
            # # logger.log_hexbin('test', 
            # #                   np.array([vec1, vec2]),
            # #                   gridsize = 1000, return_figure = True)
            #
            # plt.show()
            # exit()
            
            
            # mask = mcemtools.annular_mask((n_r, n_c), radius = 90, in_radius = 10)
            # denoised[:, :, mask == 1] = 0
            # nonoise[:, :, mask == 1] = 0
            
            # mcemtools.viewer_4D(denoised); exit()
            
            # s_bf, _ = mcemtools.sum_4D(denoised)
            # s_nn, _ = mcemtools.sum_4D(nonoise)
            # s_bf = np.expand_dims(s_bf, -1)
            # s_bf = np.expand_dims(s_bf, -1)
            # s_nn = np.expand_dims(s_nn, -1)
            # s_nn = np.expand_dims(s_nn, -1)
            # denoised /= np.tile(s_bf, (1, 1, denoised.shape[2], denoised.shape[3]))
            # denoised *= np.tile(s_nn, (1, 1, denoised.shape[2], denoised.shape[3]))
            
            data4d = denoised.copy()

        comx, comy = mcemtools.centre_of_mass_4D(data4d)
        coms = comx + 1j * comy
        # plt_imhist(data4d[1, 1]); plt.show(); exit()
        # plt_imshow(coms); plt.show()
        logger.log_imshow('coms_complex', coms, cmap = 'complex', time_tag = False)

        r_start, r_end, c_start, c_end = (0, 64, 0, 64)
        data4d = data4d[r_start:r_end, c_start:c_end]
        
        n_r, n_c = (64, 64)
        dsh = data4d.shape
        data4d_cpy = np.zeros((dsh[0], dsh[1], n_r, n_c))
        data4d_cpy[:, :, 
                   n_r // 2 - dsh[2] // 2 : n_r // 2 + dsh[2] // 2,
                   n_c // 2 - dsh[3] // 2 : n_c // 2 + dsh[3] // 2] = data4d.copy()
        data4d = data4d_cpy.copy()
        
        mask2D = mcemtools.annular_mask((n_r, n_c), radius = 8)
        mask4D = mcemtools.mask2D_to_4D(mask2D, data4d.shape)
        
        dataset = DataCube(data=data4d)
        
        defocus = -50
        probe_spacing = 1
        semiangle_cutoff = 19
        BF_rad_pix = 7
        
        dataset.calibration.set_R_pixel_size(probe_spacing)
        dataset.calibration.set_R_pixel_units('A')
        
        Q_pixel_size = semiangle_cutoff/BF_rad_pix
        
        dataset.calibration.set_Q_pixel_size(Q_pixel_size)
        dataset.calibration.set_Q_pixel_units('mrad')
        
        dataset.calibration
        
        dataset.get_dp_mean();
        
        # py4DSTEM.show(
        #     dataset.tree('dp_mean'),
        # )
        
        # logger.log_plt(f'{ptycho_dir_name}/dp_mean')
        
        probe_radius_pixels, probe_qx0, probe_qy0 = dataset.get_probe_size(plot = False)
        expand_BF = 2.0
        
        center = (probe_qx0, probe_qy0)
        radius_BF = probe_radius_pixels + expand_BF
        radii_DF = (probe_radius_pixels + expand_BF, 1e3)
        
        dpc = py4DSTEM.phase.DPC(
            datacube=dataset,
            energy = 300e3,
        ).preprocess()
        dpc.reconstruct(
            max_iter=8,
            store_iterations=True,
            reset=True,
        ).visualize(
            iterations_grid='auto',
            figsize=(12,10)
        );
        logger.log_plt(f'{ptycho_dir_name}/DPCReconstruction')
        
        ptycho = py4DSTEM.phase.MultislicePtychography(
            datacube=dataset,
            num_slices=6,
            slice_thicknesses=40,
            verbose=True,
            energy=300e3,
            defocus=defocus,
            semiangle_cutoff=semiangle_cutoff,
            object_padding_px=(6,6),
            device='gpu',
            # object_type='potential'
        ).preprocess(
            force_com_rotation=0,
            force_com_transpose=False,
            plot_rotation=False,
            plot_center_of_mass = False,
            # dp_mask = mask2D,
        )
        
        ptycho = ptycho.reconstruct(
            reset=True,
            store_iterations=True,
            num_iter = 64,
            identical_slices = 64,
            fix_probe = 64,
            step_size = 0.25,
            max_batch_size = ptycho._num_diffraction_patterns // 64,
        )
        ptycho = ptycho.visualize(
            iterations_grid='auto',
            figsize=(12,8),
        )
        # 
        # Now let's compare the different imaging methods side-by-side!
        logger.log_plt(f'{ptycho_dir_name}/ptycho_plt', time_tag = False)
        py4dstem_ptycho_save(logger, ptycho, param_prefix = ptycho_dir_name)
        
