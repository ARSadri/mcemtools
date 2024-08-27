import sys
import pathlib
import pyms

from pyms.py_multislice import (get_STEM_raster)

import numpy as np
import matplotlib.pyplot as plt
import torch
from lognflow import lognflow, printprogress, plt_colorbar, logviewer
from lognflow.plt_utils import imshow_series, plt_imshow
# from multiprocessing import freeze_support
import mcemtools

from scipy import ndimage 

if __name__ == '__main__':
    # freeze_support()
    # Get crystal
    
    # crystal = pyms.structure.fromfile('Structures/SrTiO3_CeO2_interface.xyz')
    
    # A few maniupulations to remove vaccuum at edges and create a 
    # structure psuedo-periodic
    # crystal = crystal.resize([[0.1,0.76]],axis=[0])
    # crystal = crystal.concatenate(crystal.resize([[0.017,0.99]],
    #                                     axis=[0]).reflect([0]),axis=0)
    
    # Output structure for examination in Vesta 
    # crystal.output_vesta_xtl('manipulated_SrTiO3_CeO2_interface.xtl')
    # tiling = [1,7]
    
    # Note that we have to be careful that we specify the units of
    # the atomic coordinates (cartesian in this case not the default fractional units)
    # and the temperature factor units (root mean squared displacement - urms)
    # crystal = pyms.structure.fromfile('Structures/SrTiO3.xyz',atomic_coordinates='fractional',temperature_factor_units='ums')
    
    # Set up thickness series
    
    thicknesses_list = np.array([[240]])#np.array([[40], [80], [120], [160], [200], [240]])
        
    # Grid size in pixels
    # If you're doing the SrTiO3-CeO2 interface then  for converged, 
    # publication-quality results this gridshape should be doubled
    gridshape = [1024, 1024]
    
    # Probe accelerating voltage in eV
    eV = 300e3
    
    # If only 4D-STEM is requested set detectors to be None
    # detectors = None
    
    # Number of frozen phonon passes
    # Number typically required for a converged experiment
    # nfph = 25
    # Run absorptive calculation instead
    nfph = 1
    nT=0
    
    np.random.seed(0)
    torch.manual_seed(0)
    # 4D-STEM options:
    
    # 4D-STEM with diffraction pattern sizing set by
    # multislice grid
    # FourDSTEM = True
    
    # 4D-STEM with diffraction patterns cropped to
    # 128x128 pixel readout
    
    # Option for more control over the device which performs the 
    # calculation.
    # GPU (CUDA) calculation
    device = torch.device('cuda')
    # Run calculation on your computer's second GPU
    # device = torch.device('cuda:1')
    # CPU only calculation
    # device = torch.device('cpu')
    # ROI=[0.0, 0.0,1, 1]
    # scan_posn = pyms.py_multislice.generate_STEM_raster(
    #                 real_dim, eV, app, tiling=tiling, ROI=ROI)
    tilings_list         = [[2, 2]]
    materials_fname_list = ['SrTiO3']
    materials_name_list  = ['SrTiO3_3']
    FourDSTEM = [[18, 18]]
    prob_spacing_default_list = [3.905/4]#[8.72]
    n_probes = (256, 256)
    n_spacing_list = np.array([1])#np.arange(1, 17).astype('int')
    app_list = [19]#np.floor(np.linspace(1.79, 21.4, 10)*100)/100
    df_list = np.array([float(-50)])#, float(probe_spacing/(app/1000.0))]
    
    data_root = pathlib.Path(r'./../mcemtools_data/')
    magpot_logger = lognflow(log_dir = data_root / r'magpot', time_tag = False)
    
    CoM_label = None
    labels_data = None
    stem_label = None
    stem_mask = None
    batch_size = 64
    Ne = 0
    
    n_epochs = 1
    
    if(0):
        ...
    elif(0):
        def read_bin(fpath):
            return np.fromfile(fpath,dtype=np.float64).reshape(942, 942)
        
        logger_labels = lognflow(
            log_dir = data_root / 'aFe2O3_CoM_data' )
        CoM_label = logger_labels.get_stack_from_files(
            'Main_*_942_942.bin', read_func = read_bin)
        vmin = CoM_label[0].min()
        for _ in range(3):
            DF = CoM_label[_].copy()
            DF = ndimage.rotate(DF, -91, reshape=False)
            DF = np.flip(DF, axis = 1)
            CoM_label[_] = DF.copy()
        # CoM_label = CoM_label[:, 64+12:128-20, 16+5:80-27]
        CoM_label = CoM_label[:, 64+5:64+5+n_probes[0], 16+1:16+1+n_probes[1]]
        n_probes
        stem_label = CoM_label[0].copy()
        
        stem_label /= stem_label.mean()
        stem_label = 1 - stem_label #it is DF
        
        CoM_label = CoM_label[1:].copy()

        CoM_label = CoM_label.swapaxes(0, 1).swapaxes(1, 2)
        print(f'stem_label shape: {stem_label.shape}')
        print(f'CoM_label shape: {CoM_label.shape}')

        # im = plt_imshow(stem_label, vmin = vmin); plt.show(); exit()
        batch_size = 16
        # n_epochs = 100

    elif(0):
        logger_labels = lognflow(
                log_dir = data_root / 'aFe2O3_CoM_recon' )
        labels_data = logger_labels.get_single(
            'elec_True_mag_True_ps_0.10_th_760_app_20_synth/data.npy')
        
        labels_data = labels_data[:n_probes[0], :n_probes[1]]
        batch_size = 16
    
    # import RobustGaussianFittingLibrary as rgflib
    # stem = labels_data.sum((2, 3))
    # mp = rgflib.fitValue(stem.ravel())
    # atoms_mask = mcemtools.remove_islands_by_size(stem < mp[0] - 2 * mp[1], 6)
    #
    # import scipy.ndimage
    # atoms_extent = scipy.ndimage.binary_erosion(atoms_mask)
    # atoms_extent = scipy.ndimage.binary_dilation(atoms_mask, iterations = 2)
    # stem_mask = atoms_extent - atoms_mask
    # stem_mask = stem_mask.astype('float32')
    # stem_mask[stem_mask < 0.01] = 0.015
    # # logger_labels.log_imshow('stem_mask', stem_mask, time_tag = False); exit()
    
    if Ne > 0:
        labels_data = np.random.poisson(labels_data * Ne) / Ne

    
    # labels_data_elec = np.load(data_root / 
    #     r'alphaFe2O3_n1n120\elec_True_mag_False_ps_0.25_th_120_app_21.4\data.npy')
    # mcemtools.viewer_4D(labels_data - labels_data_elec, logger = logger_labels); exit()
    # labels_data = None
    
    CBED_00_all = []
    CBED_11_all = []
    subtracted_all = []
    thickness_all = []
    FLAG_make_pot_mag_all = []
    FLAG_make_pot_elec_all = []
    for material_name, materials_fname, tiling, prob_spacing_default in zip(
        materials_name_list, materials_fname_list,
        tilings_list, prob_spacing_default_list):
    
        xyz_file = pathlib.Path(f'Structures/{materials_fname}.xyz')
        
        crystal = pyms.structure.fromfile(
            str(xyz_file), atomic_coordinates='fractional', 
            temperature_factor_units='ums')
        print(crystal.unitcell)
        # Subslicing of crystal for multislice
        nslices = int(np.ceil(crystal.unitcell[2]/1.9525))
        subslices = np.linspace(1.0/nslices,1.0,nslices)
        nsubslices = len(subslices)
        # subslices = np.array([1.0])
        
        object_scan_size = [
            n_probes[0] * prob_spacing_default * n_spacing_list.max(), 
            n_probes[1] * prob_spacing_default * n_spacing_list.max()]
        object_real_size = crystal.unitcell[:2] * np.asarray(tiling)
        print(f'object_scan_size: {object_scan_size}, '
              f'object_real_size: {object_real_size}')
        for cpeccnt, n_spacings in enumerate(n_spacing_list):
            result = None
            
            probe_spacing = n_spacings * prob_spacing_default
    
            # scan_posn = get_STEM_raster(
            #     object_scan_size = object_scan_size,
            #     object_real_size = object_real_size,
            #     probe_spacing = probe_spacing, 
            #     n_probes = n_probes,
            #     probe_spacing_noise_std = 0)

            ROI_shift_0_to_1 = ((64+12)*probe_spacing/object_real_size[0],
                                (16+5)*probe_spacing/object_real_size[1])

            scan_posn = get_STEM_raster(
                object_real_size = object_real_size,
                probe_spacing = probe_spacing, 
                n_probes = n_probes,
                )
                # ROI_shift_0_to_1 = ROI_shift_0_to_1)
            
            scan_posn_perfect = get_STEM_raster(
                object_real_size = object_real_size,
                probe_spacing = probe_spacing)
            
            scan_posn_tiling = get_STEM_raster(
                object_real_size = object_real_size,
                probe_spacing = (crystal.unitcell[0], crystal.unitcell[1]))
            
            n_spacings_str = '%02d'%n_spacings
            probe_spacing_str = '%02.02f'%probe_spacing
                
            # Probe defocus, an array can be provided for a defocus series
            for df in df_list:
                for app in app_list:
                    for thicknesses in thicknesses_list:
                        for FLAG_make_pot_mag, FLAG_make_pot_elec in zip(
                                # [True, False], [True, True]):
                                # [True, False, True], [True, True, False]):
                                [False], [True]):
                            exp_name =  f'{material_name}'
                            exp_name += f'/elec_{FLAG_make_pot_elec}'
                            exp_name += f'_mag_{FLAG_make_pot_mag}'
                            exp_name += f'_ps_{probe_spacing_str}'
                            exp_name += f'_th_{thicknesses[0]}'
                            exp_name += f'_app_{app}'
                            exp_name += f'_df_{df}'
                            if Ne > 0:
                                exp_name += f'_Ne_{Ne}' 
                            logger = lognflow(
                                log_dir = data_root / exp_name, time_tag = False)
                            
                            # print('Max resolution permitted by the sample grid is {0} mrad'.format(
                            #     pyms.max_grid_resolution(gridshape,real_dim,eV=eV)))
                            # _nyquist_probe_spacing = nyquist_probe_spacing(eV=eV, alpha = app)
                            # logger(f'nyquist probe spacing is: {_nyquist_probe_spacing}')
                            # logger(f'Your probe spacing: {probe_spacing}')
                            
                            if stem_label is not None:
                                logger.log_imshow('label/DF_label', stem_label)
                            if CoM_label is not None:
                                logger.log_imshow('label/CoM_label_complex',
                                    CoM_label[..., 0] + 1j * CoM_label[..., 1], cmap = 'complex')
                                logger.log_imshow('label/CoM_label_abs_angle',
                                    CoM_label[..., 0] + 1j * CoM_label[..., 1])
                                logger.log_imshow('label/CoM_label_x_y',
                                    CoM_label[..., 0] + 1j * CoM_label[..., 1], 
                                    complex_type = 'real_imag')
                            fig = plt.figure(figsize = (8, 8))
                            ax = fig.add_subplot(111)
                            ax.plot(scan_posn_perfect[..., 0].ravel() * object_real_size[0], 
                                    scan_posn_perfect[..., 1].ravel() * object_real_size[1], '.', 
                                    label = 'object')
                            ax.plot(scan_posn[..., 0].ravel() * object_real_size[0], 
                                    scan_posn[..., 1].ravel() * object_real_size[1], '.', 
                                    color = 'red', label = 'ROI')
                            ax.plot(scan_posn_tiling[..., 0].ravel() * object_real_size[0], 
                                    scan_posn_tiling[..., 1].ravel() * object_real_size[1], 'x',
                                    color = 'green', label = 'unit cell')
                            ax.set_aspect('equal', 'box')
                            plt.legend()
                            # plt.show(); exit()
                            logger('plot ready')
                    
                            fpath = logger.log_plt('ROI', dpi = 1000)
                            
                            logger(fpath)
                    
                            fpath = logger.log_single('object_real_size.txt', object_real_size)
                    
                            logger(fpath)
                    
                            fpath = logger.copy('code.py', sys.argv[0])
                            logger(fpath)
                            
                            fpath = logger.log_single('scan_posn_perfect', scan_posn_perfect)
                            logger(fpath)
                            fpath = logger.log_single('scan_posn', scan_posn)
                            logger(fpath)
                            fpath = logger.log_single('scan_posn_tiling', scan_posn_tiling)
                            logger(fpath)
                            
                            detectors = [[0,app/2], [app/2,app],[70,150]]

                            crystal = pyms.structure.fromfile(str(xyz_file),
                                      atomic_coordinates='fractional',
                                      temperature_factor_units='ums')
                            data4d = logger.logged.get_single('data')
                            if(data4d is None):
                                kwargs_dev = dict(
                                    magpot_logger = None,
                                    logger = logger,
                                    FLAG_make_pot_elec = FLAG_make_pot_elec,
                                    FLAG_make_pot_mag = FLAG_make_pot_mag,
                                    lr = None,
                                    betas = None,
                                    n_epochs = 1,
                                    labels_data = None,
                                    stem_mask = None,
                                    stem_label = None,
                                    CoM_label = None,
                                    backprop = False,
                                    det_geo = None,
                                    )
                                result = pyms.STEM_multislice(
                                    crystal,
                                    gridshape,
                                    eV,
                                    app,
                                    thicknesses,
                                    batch_size = batch_size,
                                    subslices=subslices,
                                    device_type=device,
                                    df=df,
                                    nfph=nfph,
                                    nT=nT,
                                    FourD_STEM=FourDSTEM,
                                    PACBED=None,
                                    STEM_images=None,
                                    tiling=tiling,
                                    detector_ranges=None,
                                    scan_posn = scan_posn,
                                    kwargs_dev = kwargs_dev,
                                    seed = np.arange(1000, dtype='int'),
                                    )
                                data4d = result['datacube'][:, :, 1:-1, 1:-1]
                                # data4d.sum().backward()
                                
                                try:
                                    data4d = data4d.detach().cpu().numpy().astype('float32')
                                except:
                                    pass
                                n_r = data4d.shape[-1]
                                data4d = mcemtools.bin_4D(data4d, 1, 1)
                                logger.log_single(f'data', data4d)
                            comx, comy = mcemtools.centre_of_mass_4D(data4d)
                            com = comx + 1j * comy
                            logger.log_imshow(f'com_complex', com, cmap = 'complex')
                            logger.log_imshow(f'com_abs_angle', com, complex_type = 'abs_angle')
                            logger.log_imshow(f'com_real_imag', com, complex_type = 'real_imag')
                            s, p = mcemtools.sum_4D(data4d)
                            logger.log_imshow(f'STEM', s)
                            logger.log_imshow(f'PACBED', p)
                            
                            mask2d = mcemtools.annular_mask(
                                (data4d.shape[2], data4d.shape[3]), in_radius = 32)
                            mask4d = mcemtools.mask2D_to_4D(mask2d, data4d.shape)
                            s, p = mcemtools.sum_4D(data4d, mask4d)
                            logger.log_imshow(f'STEM_DF', s)
                            logger.log_imshow(f'PACBED_DF', p)
                            
                            logger(f'FourDSTEM:{FourDSTEM}')
                            logger(f'data4d.shape:{data4d.shape}')
                            frame = mcemtools.data4D_to_frame(data4d)
                            bfmask2d = mcemtools.annular_mask(
                                (data4d.shape[2], data4d.shape[3]), radius = 16)
                            vmin = data4d[:, :, bfmask2d == 1].min()
                            vmax = data4d[:, :, bfmask2d == 1].max()
                            logger.log_imshow(
                                'frame', frame, dpi = 2000, colorbar = False,
                                vmin = vmin, vmax = vmax)
                            logger('probe spacing generated')
                            print('cuda test passed')
