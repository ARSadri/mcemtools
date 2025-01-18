import pathlib
import torch
import pyms
import utils
from copy import copy
from pyms.Probe import aberration
from itertools import product
import mcemtools
from lognflow import lognflow, printprogress, printv
from lognflow.plt_utils import (
    np, plt, plt_imshow, plt_colorbar, plt_contours, plt_imhist)

if __name__ == '__main__':
    data_root = pathlib.Path(r'./pyms_data')
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cuda')
    GPU_streaming = False
    CFG_absorptive = True
    if CFG_absorptive:
        nfph = 1
        nT   = 0
    else:
        nfph = 20
        nT   = 2 * nfph
    gridshape         = [1024, 1024]
    eV                = 300e3
    thicknesses_list  = [[240]]
    tiling            = [4, 4]
    materials_fname   = 'SrTiO3'
    material_name     = 'SrTiO3_3'
    FourD_STEM        = [[32, 32]]
    n_probes          = (32, 32)
    probe_spacing     = 0.25
    
    app               = 19
    df_list           = [0]
    DF_radius         = 8
    BF_radius         = 7

    spatInc = None

    aberrations     = []
    # aberrations.append(aberration("C10", "C1", "Defocus          ", 1, 0.0, 1, 0))
    # aberrations.append(aberration("C12", "A1", "2-Fold astig.    ", 0, 0, 1, 2))
    # aberrations.append(aberration("C23", "A2", "3-Fold astig.    ", 0, 0, 2, 3))
    # aberrations.append(aberration("C21", "B2", "Axial coma       ", 0, 0, 2, 1))
    # aberrations.append(aberration("C30", "C3", "3rd order spher. ", 0, 0, 3, 0))
    
    # aberrations.append(aberration("C34", "A3", "4-Fold astig.    ", 0.0, 0.0, 3, 4))
    # aberrations.append(aberration("C32", "S3", "Axial star aber. ", 0.0, 0.0, 3, 2))
    # aberrations.append(aberration("C45", "A4", "5-Fold astig.    ", 0.0, 0.0, 4, 5))
    # aberrations.append(aberration("C43", "D4", "3-Lobe aberr.    ", 0.0, 0.0, 4, 3))
    # aberrations.append(aberration("C41", "B4", "4th order coma   ", 0.0, 0.0, 4, 1))
    # aberrations.append(aberration("C50", "C5", "5th order spher. ", 0.0, 0.0, 5, 0))
    # aberrations.append(aberration("C56", "A5", "6-Fold astig.    ", 0.0, 0.0, 5, 6))
    # aberrations.append(aberration("C52", "S5", "5th order star   ", 0.0, 0.0, 5, 2))
    # aberrations.append(aberration("C54", "R5", "5th order rosette", 0.0, 0.0, 5, 4))
    
    # specimen_tilt_list = list(product([-4, 0, 4], [-4, 0, 4]))
    specimen_tilt_list = [[0, 0]]
    
    xyz_file = pathlib.Path(f'{materials_fname}.xyz')
    crystal = pyms.structure.fromfile(
        str(xyz_file), atomic_coordinates='fractional', 
        temperature_factor_units='ums')
    crystal_original = copy(crystal)

    nslices = int(np.ceil(crystal.unitcell[2]/1.9525))
    subslices = np.linspace(1.0/nslices,1.0,nslices)
    nsubslices = len(subslices)
    # subslices = np.array([1.0])
    
    object_real_size = crystal.unitcell[:2] * np.asarray(tiling)
    
    printv(object_real_size)
    
    scan_posn = utils.get_STEM_raster(
        object_real_size = object_real_size,
        n_probes = n_probes,
        probe_spacing = probe_spacing, 
        )
    printv(scan_posn)

    if 1:
        batch_size = 8
        Ne         = 0
        n_epochs   = 1
        det_geo    = None
     
    thicknesses = thicknesses_list[0]
    specimen_tilt = specimen_tilt_list[0]
    df = df_list[0]
    printv(df)
    printv(specimen_tilt)
    printv(thicknesses)
    print('='*20, flush = True)
    crystal = copy(crystal_original)
    exp_name =  f'{material_name}/{material_name}'
    # exp_name += f'_ps_{probe_spacing:02.02f}'
    exp_name += f'_th_{thicknesses[0]:.1f}'
    # exp_name += f'_app_{app}'
    exp_name += f'_df_{df}'
    # if (specimen_tilt[0] != 0) | (specimen_tilt[1] != 0):
    exp_name += f'_tilt_{specimen_tilt[0]}_{specimen_tilt[1]}'
    if Ne > 0:
        exp_name += f'_Ne_{Ne}' 
    if spatInc is not None:
        spatinc_value = float(spatInc['s1']*probe_spacing)
        exp_name += f'_spat_{spatinc_value:.1f}'
    # if nfph > 1:
    #     exp_name += f'_nfph_{nfph}'
    # if not backprop:
    #     exp_name += f'_no_opt'
    # exp_name += f'_detsz_{det_geo_size}'
    logger = lognflow(log_dir = data_root / exp_name, time_tag = False)
    logger.log_code()
    logger.copy(xyz_file.name, xyz_file)

    fig, ax = utils.plot_scan_positions(
        atoms_xyz = crystal.atoms[:, :3],
        atoms_Z = crystal.atoms[:, 3],
        unitcell = crystal.unitcell,
        object_real_size = object_real_size,
        probe_spacing = probe_spacing,
        scan_posn = scan_posn,
        markersize = 0.4,
        atom_markersize = 8,
        text_size = 1)

    logger.savefig('ROI', dpi = 2000)
    logger.save('object_real_size.txt', object_real_size)
    logger.save('scan_posn', scan_posn)

    data4d = logger.load('data4d.npy')
    
    if (data4d is None) | 0:
        
        result = pyms.STEM_multislice(
            crystal,
            gridshape,
            eV,
            app,
            thicknesses,
            batch_size      = batch_size,
            subslices       = subslices,
            device_type     = device,
            df              = df,
            nfph            = nfph,
            nT              = nT,
            FourD_STEM      = FourD_STEM,
            PACBED          = None,
            STEM_images     = None,
            tiling          = tiling,
            detector_ranges = None,
            scan_posn       = scan_posn,
            seed            = None,
            specimen_tilt   = specimen_tilt,
            aberrations     = aberrations,
            )
    
        data4d = result['datacube']
        logger.save('data4d', data4d, time_tag = False)
        
    if 1:
        logger.imshow('data4d_fft2', 
                      np.log(np.abs(np.fft.fftshift(np.fft.fft2(data4d.sum((2, 3)))))**2))

        comx, comy = mcemtools.centre_of_mass_4D(data4d)
        com = comx + 1j * comy
        logger.imshow(f'com_complex', com, cmap = 'complex')
        logger.imshow(f'com_abs_angle', com)
        logger.imshow(f'com_real_imag', com, cmap = 'real_imag')
        s, p = mcemtools.sum_4D(data4d)
        sym_stem = s.copy()
        logger.imshow(f'STEM', s)
        logger.imshow(f'PACBED', p)
        
        mask2d = mcemtools.annular_mask((data4d.shape[2], data4d.shape[3]), in_radius = DF_radius, radius = np.inf)
        mask4d = mcemtools.mask2D_to_4D(mask2d, data4d.shape)
        s, p = mcemtools.sum_4D(data4d, mask4d)
        logger.imshow(f'STEM_DF', s)
        logger.imshow(f'PACBED_DF', p)
        
        logger(f'FourD_STEM:{FourD_STEM}')
        logger(f'data4d.shape:{data4d.shape}')
        frame = mcemtools.data4D_to_frame(data4d[:8,:8])
        bfmask2d = mcemtools.annular_mask(
            (data4d.shape[2], data4d.shape[3]), radius = BF_radius)
        mask4d = mcemtools.mask2D_to_4D(bfmask2d, data4d.shape)
        s, p = mcemtools.sum_4D(data4d, mask4d)
        logger.imshow(f'STEM_BF', s)
        logger.imshow(f'PACBED_BF', p)
        vmin = data4d[:, :, bfmask2d == 1].min()
        vmax = data4d[:, :, bfmask2d == 1].max()
        logger.imshow('frame', frame, dpi = 1000, colorbar = False,
                      vmin = vmin, vmax = vmax)

    # I choose 1 because the probe spacing is 0.25A which will be the std of
    # spatial incoherence of 0.25A.
    # I found out by calculation that the Tokyo Uni 4DSTEM machine (presented in the 
    # anti-ferromagnetic Nature paper) introduces 0.4A spatial incoherence.)
    spatInc = dict(model = 'Gaussian_Lorentzian', angle = 0,
                   s1 = 1, s2 = 1, gamma_x = 1, gamma_y = 1) 
     
    if spatInc is not None:
        data4d = mcemtools.analysis.spatial_incoherence_4D(data4d, spatInc).astype('float32')
        logger.save(f'spatInc/spatInc', spatInc)
        logger.save(f'spatInc/data', data4d)
        
        comx, comy = mcemtools.centre_of_mass_4D(data4d)
        com = comx + 1j * comy
        logger.imshow(f'spatInc/com_complex', com, cmap = 'complex')
        logger.imshow(f'spatInc/com_abs_angle', com)
        logger.imshow(f'spatInc/com_real_imag', com, cmap = 'real_imag')
        s, p = mcemtools.sum_4D(data4d)
        sym_stem = s.copy()
        logger.imshow(f'spatInc/STEM', s)
        logger.imshow(f'spatInc/PACBED', p)

        mask2d = mcemtools.annular_mask(
            (data4d.shape[2], data4d.shape[3]), in_radius = DF_radius)
        mask4d = mcemtools.mask2D_to_4D(mask2d, data4d.shape)
        s, p = mcemtools.sum_4D(data4d, mask4d)
        logger.imshow(f'spatInc/STEM_DF', s)
        logger.imshow(f'spatInc/PACBED_DF', p)

        logger(f'spatInc FourD_STEM:{FourD_STEM}')
        logger(f'spatInc data4d.shape:{data4d.shape}')
        frame = mcemtools.data4D_to_frame(data4d[:8,:8])
        bfmask2d = mcemtools.annular_mask(
            (data4d.shape[2], data4d.shape[3]), radius = BF_radius)
        mask4d = mcemtools.mask2D_to_4D(bfmask2d, data4d.shape)
        s, p = mcemtools.sum_4D(data4d, mask4d)
        logger.imshow(f'spatInc/STEM_BF', s)
        logger.imshow(f'spatInc/PACBED_BF', p)
        vmin = data4d[:, :, bfmask2d == 1].min()
        vmax = data4d[:, :, bfmask2d == 1].max()
        logger.imshow('spatInc/frame', frame, dpi = 1000, colorbar = False,
                      vmin = vmin, vmax = vmax)
