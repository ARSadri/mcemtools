import numpy as np
from .masking import annular_mask, mask2D_to_4D, image_by_windows
from lognflow import printprogress, lognflow
from skimage.transform import warp_polar
import scipy
from itertools import product
import torch
import mcemtools
import torch.nn.functional as F
from typing import Tuple

def interpolate_surface(grid_locations, values, resolution=None, method='cubic'):
    from scipy.interpolate import griddata

    x = grid_locations[:, 0]
    y = grid_locations[:, 1]

    if resolution is None:
        dx = np.abs(np.diff(np.sort(np.unique(x))))
        dy = np.abs(np.diff(np.sort(np.unique(y))))
        min_dx = dx[dx > 0].min() if np.any(dx > 0) else 1.0
        min_dy = dy[dy > 0].min() if np.any(dy > 0) else 1.0
        resolution = 0.1 * min(min_dx, min_dy)

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    grid_x, grid_y = np.meshgrid(
        np.arange(x_min, x_max + resolution, resolution),
        np.arange(y_min, y_max + resolution, resolution)
    )

    grid_z = griddata(grid_locations, values, (grid_x, grid_y), method=method)
    extent = (x_min, x_max, y_min, y_max)
    return grid_x, grid_y, grid_z, extent

def Lorentzian_2dkernel(filter_size, gamma_x=1, gamma_y=1, angle=0):
    """
    Generate a 2D Lorentzian kernel with specified parameters.
    
    Parameters:
    -----------
    filter_size : int
        Size of the kernel (square grid).
    gamma_x : float, optional
        Scale parameter (half-width at half-maximum) along the x-axis. Default is 1.
    gamma_y : float, optional
        Scale parameter (half-width at half-maximum) along the y-axis. Default is 1.
    angle : float, optional
        Rotation angle (in degrees) for the kernel. Default is 0.
    
    Returns:
    --------
    kern2d : ndarray
        Normalized 2D Lorentzian kernel.
    """
    # Rotation matrix
    theta = np.deg2rad(angle)
    R = np.array([[np.cos(theta), -np.sin(theta)], 
                  [np.sin(theta),  np.cos(theta)]])
    
    # Create meshgrid
    lim = filter_size // 2 + (filter_size % 2) / 2
    x = np.linspace(-lim, lim, filter_size)
    y = np.linspace(-lim, lim, filter_size)
    X, Y = np.meshgrid(x, y)
    
    # Rotate coordinates
    coords = np.stack([X.flatten(), Y.flatten()], axis=0)
    rotated_coords = R @ coords
    X_rot, Y_rot = rotated_coords[0, :].reshape(X.shape), rotated_coords[1, :].reshape(Y.shape)
    
    # Compute Lorentzian kernel
    kern2d = 1 / (1 + (X_rot / gamma_x)**2 + (Y_rot / gamma_y)**2)
    return kern2d / kern2d.sum()


def Gaussian_2dkernel(filter_size, s1=1, s2=1, angle=0):
    """
    Generate a 2D Gaussian kernel with specified parameters.
    
    Parameters:
    -----------
    filter_size : int
        Size of the kernel (square grid).
    s1 : float, optional
        Standard deviation along the first axis. Default is 1.
    s2 : float, optional
        Standard deviation along the second axis. Default is 1.
    angle : float, optional
        Rotation angle (in degrees) for the kernel. Default is 0.
    
    Returns:
    --------
    kern2d : ndarray
        Normalized 2D Gaussian kernel.
    """
    # Define the covariance matrix
    cov_matrix = np.array([[s1**2,   0  ], 
                           [  0  , s2**2]])
    
    # Rotation matrix
    theta = np.deg2rad(angle)
    R = np.array([[np.cos(theta), -np.sin(theta)], 
                  [np.sin(theta),  np.cos(theta)]])
    
    # Rotate the covariance matrix
    cov_matrix_rotated = R @ cov_matrix @ R.T
    
    # Create meshgrid
    lim = filter_size // 2 + (filter_size % 2) / 2
    x = np.linspace(-lim, lim, filter_size)
    y = np.linspace(-lim, lim, filter_size)
    X, Y = np.meshgrid(x, y)
    
    # Create the Gaussian kernel
    pos = np.dstack((X, Y))
    rv = scipy.stats.multivariate_normal([0, 0], cov_matrix_rotated)
    kern2d = rv.pdf(pos)
    
    return kern2d / kern2d.sum()

def spatial_incoherence_4D(data4d, spatInc_params, return_filter = False):
    """
    Apply spatial incoherence filtering to 4D-STEM data using Gaussian and/or Lorentzian filters.

    Parameters:
    -----------
    data4d : ndarray
        Input 4D dataset of shape (n_x, n_y, n_r, n_c), where:
        - n_x, n_y: Spatial dimensions.
        - n_r, n_c: Detector dimensions.
    spatInc_params : dict
        Parameters for the spatial incoherence filter. Expected keys include:
        - 'model' : str or list of str
            Specifies the filter type(s) to apply. Options are 'Gaussian', 'Lorentzian', or both.
        - 's1', 's2' : float, optional
            Parameters for the Gaussian filter (e.g., standard deviations along principal axes).
        - 'gamma_x', 'gamma_y' : float, optional
            Parameters for the Lorentzian filter (e.g., scale factors along principal axes).
        - 'angle' : float, optional
            Rotation angle for the filters (applies to both Gaussian and Lorentzian filters).
    use_fft : bool, optional
        If True, performs the filtering in Fourier space for efficiency. 
        If False, performs the filtering in real space. Default is False.

    Returns:
    --------
    filtered_data4d : ndarray
        The filtered 4D dataset, of the same shape as the input `data4d`.

    Notes:
    ------
    - In the Fourier-space approach (`use_fft=True`), the combined filter is computed and applied 
      in Fourier space for faster computation on large datasets.
    - In the real-space approach (`use_fft=False`), the filtering is performed directly using 
      window-based operations, which may be slower but avoids FFT artifacts.
    - Filters are normalized before application to ensure the total weight is 1.
    - If both 'Gaussian' and 'Lorentzian' models are specified in `spatInc_params['model']`, the 
      filters are combined (summed) before normalization.

    Example Usage:
    --------------
    >>> spatInc_params = {
    >>>     'model': ['Gaussian', 'Lorentzian'],
    >>>     's1': 1.5,
    >>>     's2': 2.0,
    >>>     'gamma_x': 0.8,
    >>>     'gamma_y': 0.9,
    >>>     'angle': 45
    >>> }
    >>> filtered_data = spatial_incoherence_4D(data4d, spatInc_params, use_fft=True)
    """
    n_x, n_y, n_r, n_c = data4d.shape
    weight = spatInc_params['weight']
    assert ((0 <= weight) & (weight <= 1)), \
        'the weight between Gaussian and Lorentzian should be a number between 0 and 1'

    if not 's1' in spatInc_params:
        spatInc_params['s1'] = spatInc_params['s']
        spatInc_params['s2'] = spatInc_params['s']
        spatInc_params['angle'] = 0
        spatInc_params['gamma_x'] = spatInc_params['gamma']
        spatInc_params['gamma_y'] = spatInc_params['gamma']
        
    if not ('model' in spatInc_params):
        spatInc_params['model'] = 'Gaussian'
        weight = 1
    
    filter = np.zeros((n_x, n_y))
    if 'Gaussian' in spatInc_params['model']:
        gaussian_filter = Gaussian_2dkernel(
            np.maximum(data4d.shape[0], data4d.shape[1]), 
            spatInc_params['s1'], spatInc_params['s2'], spatInc_params['angle'])
        gaussian_filter = mcemtools.masking.crop_or_pad(
            gaussian_filter, (data4d.shape[0], data4d.shape[1]))
        gaussian_filter = weight * gaussian_filter / gaussian_filter.sum()
        from lognflow import plt_imshow
        filter += gaussian_filter
    else:
        weight = 0
    if 'Lorentzian' in spatInc_params['model']:
        Lorentzian_filter = Lorentzian_2dkernel(
            np.maximum(data4d.shape[0], data4d.shape[1]), 
            spatInc_params['gamma_x'], spatInc_params['gamma_y'], spatInc_params['angle'])
        Lorentzian_filter = mcemtools.masking.crop_or_pad(
            Lorentzian_filter, (data4d.shape[0], data4d.shape[1]))
        Lorentzian_filter = (1-weight) * Lorentzian_filter / Lorentzian_filter.sum() 
        filter += Lorentzian_filter

    data4d_ft = scipy.fft.fftn(data4d, axes=(0, 1))
    filter_ft = scipy.fft.fftn(filter, s=(n_x, n_y))
    result_ft = data4d_ft * filter_ft[:, :, None, None]
    filtered_data4d = scipy.fft.ifftn(result_ft, axes=(0, 1)).real

    if return_filter:
        return filtered_data4d, filter
    else:
        return filtered_data4d

def spatial_incoherence_4D_real(data4d, spatInc_params, use_fft = True, return_filter = False,
                           weight = 1):
    """
    Apply spatial incoherence filtering to 4D-STEM data using Gaussian and/or Lorentzian filters.

    Parameters:
    -----------
    data4d : ndarray
        Input 4D dataset of shape (n_x, n_y, n_r, n_c), where:
        - n_x, n_y: Spatial dimensions.
        - n_r, n_c: Detector dimensions.
    spatInc_params : dict
        Parameters for the spatial incoherence filter. Expected keys include:
        - 'model' : str or list of str
            Specifies the filter type(s) to apply. Options are 'Gaussian', 'Lorentzian', or both.
        - 's1', 's2' : float, optional
            Parameters for the Gaussian filter (e.g., standard deviations along principal axes).
        - 'gamma_x', 'gamma_y' : float, optional
            Parameters for the Lorentzian filter (e.g., scale factors along principal axes).
        - 'angle' : float, optional
            Rotation angle for the filters (applies to both Gaussian and Lorentzian filters).
    use_fft : bool, optional
        If True, performs the filtering in Fourier space for efficiency. 
        If False, performs the filtering in real space. Default is False.

    Returns:
    --------
    filtered_data4d : ndarray
        The filtered 4D dataset, of the same shape as the input `data4d`.

    Notes:
    ------
    - In the Fourier-space approach (`use_fft=True`), the combined filter is computed and applied 
      in Fourier space for faster computation on large datasets.
    - In the real-space approach (`use_fft=False`), the filtering is performed directly using 
      window-based operations, which may be slower but avoids FFT artifacts.
    - Filters are normalized before application to ensure the total weight is 1.
    - If both 'Gaussian' and 'Lorentzian' models are specified in `spatInc_params['model']`, the 
      filters are combined (summed) before normalization.

    Example Usage:
    --------------
    >>> spatInc_params = {
    >>>     'model': ['Gaussian', 'Lorentzian'],
    >>>     's1': 1.5,
    >>>     's2': 2.0,
    >>>     'gamma_x': 0.8,
    >>>     'gamma_y': 0.9,
    >>>     'angle': 45
    >>> }
    >>> filtered_data = spatial_incoherence_4D(data4d, spatInc_params, use_fft=True)
    """
    n_x, n_y, n_r, n_c = data4d.shape

    if not ('model' in spatInc_params):
        spatInc_params['model'] = 'Gaussian'
        weight = 1
    
    if use_fft:
        filter = np.zeros((n_x, n_y))
        if 'Gaussian' in spatInc_params['model']:
            gaussian_filter = weight * Gaussian_2dkernel(
                np.maximum(data4d.shape[0], data4d.shape[1]), 
                spatInc_params['s1'], spatInc_params['s2'], spatInc_params['angle'])
            filter += mcemtools.masking.crop_or_pad(
                gaussian_filter, (data4d.shape[0], data4d.shape[1]))
        else:
            weight = 0
        if 'Lorentzian' in spatInc_params['model']:
            Lorentzian_filter = (1-weight) * Lorentzian_2dkernel(
                np.maximum(data4d.shape[0], data4d.shape[1]), 
                spatInc_params['gamma_x'], spatInc_params['gamma_y'], spatInc_params['angle'])
            
            from lognflow.plt_utils import plt_imshow, plt
            
            filter += mcemtools.masking.crop_or_pad(
                Lorentzian_filter, (data4d.shape[0], data4d.shape[1]))
        filter = filter / filter.sum()

        data4d_ft = scipy.fft.fftn(data4d, axes=(0, 1))
        filter_ft = scipy.fft.fftn(filter, s=(n_x, n_y))
        filter_ft_tiled = np.tile(
            filter_ft[:, :, np.newaxis, np.newaxis], (1, 1, n_r, n_c))
        result_ft = data4d_ft * filter_ft_tiled
        filtered_data4d = scipy.fft.ifftn(result_ft, axes=(0, 1)).real
    else:
        filter = np.zeros((n_x, n_y))
        if 'Gaussian' in spatInc_params['model']:
            filter += Gaussian_2dkernel(**spatInc_params)
        if 'Lorentzian' in spatInc_params['model']:
            filter += Lorentzian_2dkernel(**spatInc_params)
        filter = filter / filter.sum()

        gaussian_filter = np.expand_dims(gaussian_filter, -1)
        gaussian_filter = np.expand_dims(gaussian_filter, -1)
        gaussian_filter = np.tile(gaussian_filter, (1, 1, n_r, n_c))
        
        imgbywin = mcemtools.image_by_windows(
            (n_x, n_y), gaussian_filter.shape, skip = (1, 1), method = 'fixed')
        filtered_data4d = np.zeros(
            (imgbywin.grid_shape[0],imgbywin.grid_shape[1], n_r, n_c),
            dtype = data4d.dtype)
        for grc in imgbywin.grid:
            filtered_data4d[grc[0], grc[1]] = (data4d[
                grc[0]:grc[0] + imgbywin.win_shape[0], 
                grc[1]:grc[1] + imgbywin.win_shape[1]] * gaussian_filter).sum((0, 1))

    if return_filter:
        return filtered_data4d, filter
    else:
        return filtered_data4d

def normalize_4D(data4D, weights4D = None, method = 'loop'):
    """
        Note::
            make sure you have set weights4D[data4D == 0] = 0 when dealing with
            Poisson.
    """
    data4D = data4D.copy()
    n_x, n_y, n_r, n_c = data4D.shape

    for x_cnt in range(n_x):
        for y_cnt in range(n_y):
            cbed = data4D[x_cnt, y_cnt]
            if weights4D is not None:
                cbed = cbed[weights4D[x_cnt, y_cnt] > 0]
            cbed -= cbed.mean()
            cbed_std = cbed.std()
            if cbed_std > 0:
                cbed /= cbed_std
            else:
                cbed *= 0
            if weights4D is not None:
                data4D[x_cnt, y_cnt][weights4D[x_cnt, y_cnt] > 0] = cbed.copy()
            else:
                data4D[x_cnt, y_cnt] = cbed.copy()
    return data4D

def calc_ccorr(CBED, args: tuple):
    mask_ang, nang, mflag = args
    
    vec_a = warp_polar(CBED)
    vec_a_n = vec_a[mask_ang > 0]
    vec_a_n_std = vec_a_n.std()
    vec_a_n -= vec_a_n.mean()
    if vec_a_n_std > 0:
        vec_a_n /= vec_a_n_std
    else:
        vec_a_n *= 0
    vec_a[mask_ang > 0] = vec_a_n.copy()

    rot = vec_a.copy()
    corr = np.zeros(nang)
    for _ang in range(nang):
        if mflag:
            vec_a = np.flip(rot.copy(), axis = 0)
        corr[_ang] = ((rot * vec_a)[mask_ang > 0]).sum() 
        rot = np.roll(rot, 1, axis=0)
    return corr

def calc_symm(CBED, args: tuple):
    mask_ang, nang, mflag = args
    
    nang = 360
    
    polar = warp_polar(CBED) #shape:  360, 46 for a 64x64 pattern
    kvec = np.arange(polar.shape[1]) / (nang / 2 / np.pi)
    if mask_ang is not None:
        polar[mask_ang == 0] = 0
    
    """
        perform angular autocorrelation or autoconvolutiuon using Fourier
        correlation theorems.
        note: one difference between the above symmetry measures is the
        presence/absence of the absolute value. The other difference is that
        the symmetry angle is halved for the mirrors, since a similarity
        transform is implied to rotate, perform inversion, then rotate back.
    """
    if mflag == 1: #mirror symmetries
        polar_autocorr = np.real(np.fft.ifft((np.fft.fft(polar,nang,0))**2,nang,0))
    else:          #rotational symmetries
        polar_autocorr = np.real(np.fft.ifft(np.abs(np.fft.fft(polar,nang,0)),nang,0))
    """
        multiply array to account for Jacobian polar r weighting (here kvec). 
        Integrate over radius in the diffraction pattern - one could also
        mask the pattern beforehand, as in ACY Liu's correlogram approach.
    """
    corr = (polar_autocorr*kvec[np.newaxis,:]).sum(1)
    
    """
        notice the deliberate omission of fftshift above.     
        factors of nang and 2*pi are for numerical comparison to the Riemann
        sum integrals in the Cartesian case.
        normalise with respect to no symmetry operation.  For accurate
        normalisation, include otherwise redundant polar coordinate 
        conversion and subsequent squaring.
    """
    corr = corr/((np.abs(polar))*kvec[np.newaxis,:]).sum()
    
    return corr

def SymmSTEM(data4D, mask2D = None, nang = 180, mflag = False, 
             verbose = True, use_multiprocessing = False,
             use_autoconvolutiuon = False):
    # assert not use_autoconvolutiuon, 'autoconvolutiuon is not ready yet!'
    n_x, n_y, n_r, n_c = data4D.shape
    
    if mask2D is not None:
        assert mask2D.shape == (n_r, n_c),\
            'mask2D should have the same shape as'\
            ' (data4D.shape[2], data4D.shape[3])'
        mask_ang = warp_polar(mask2D.copy())
    else:
        mask_ang = warp_polar(np.ones((n_r, n_c)))
    
    inputs_to_share = (mask_ang, nang, mflag)
    
    if use_multiprocessing:
        inputs_to_iter = data4D.reshape((n_x*n_y, n_r, n_c))
        from lognflow import multiprocessor
        corr_ang_auto = multiprocessor(
            calc_symm if use_autoconvolutiuon else calc_ccorr, 
            iterables = (inputs_to_iter, ),
            shareables = inputs_to_share,
            verbose = verbose)
        corr_ang_auto = corr_ang_auto.reshape(
            (n_x, n_y, corr_ang_auto.shape[1]))
        corr_ang_auto /= (mask_ang > 0).sum()
    else:
        corr_ang_auto = np.zeros((n_x, n_y, nang))
        if(verbose):
            pBar = printprogress(
                n_x * n_y, title = f'Symmetry STEM for {n_x * n_y} patterns')
        for i in range(n_x):
            for j in range(n_y):
                if use_autoconvolutiuon:
                    corr = calc_symm(data4D[i, j], inputs_to_share)
                else:
                    corr = calc_ccorr(data4D[i, j], inputs_to_share)
                corr_ang_auto[i,j] = corr.copy()
                if(verbose):
                    pBar()
        corr_ang_auto /= (mask_ang > 0).sum()
    
    return corr_ang_auto

def swirl_and_sum(img):
    _img = np.zeros(img.shape, dtype = img.dtype)
    _img[1:-1, 1:-1] = \
          img[ :-2,  :-2] \
        + img[ :-2, 1:-1] \
        + img[ :-2, 2:  ] \
        + img[1:-1,  :-2] \
        + img[1:-1, 1:-1] \
        + img[1:-1, 2:  ] \
        + img[2:  ,  :-2] \
        + img[2:  , 1:-1] \
        + img[2:  , 2:  ]
    return _img
    
def sum_4D(data4D, weight4D = None):
    """ Annular virtual detector
            Given a 4D dataset, n_x x n_y x n_r x n_c.
            the output is the marginalized images over the n_x, n_y or n_r,n_c
        
        :param data4D:
            data in 4 dimension real_x x real_y x k_r x k_c
        :param weight4D: np.ndarray
            a 4D array, optionally, calculate the sum according to the weights
            in weight4D. If wish to use it as a mask, use 0 and 1.
    """
    if weight4D is not None:
        assert weight4D.shape == data4D.shape,\
            'weight4D should have the same shape as data4D'
    
    I4D_cpy = data4D.copy()
    if weight4D is not None:
        I4D_cpy = I4D_cpy * weight4D
    PACBED = I4D_cpy.sum(1).sum(0).squeeze()
    totI = I4D_cpy.sum(3).sum(2).squeeze()
    return totI, PACBED

def conv_4D_single(grc, sharables):
    imgbywin, data4D = sharables
    return data4D[grc[0]:grc[0] + imgbywin.win_shape[0], 
                  grc[1]:grc[1] + imgbywin.win_shape[1]].sum((0, 1))
    
def conv_4D(data4D, 
            winXY, 
            conv_function = sum_4D, 
            skip = (1, 1), 
            use_mp = True):
    """
        :param conv_function:
            a function that returns a tuple, we will use the second element:
            _, stat = conv_function(data4D)
            This function should return a 2D array at second position in the 
            tuple. For example sum_4D returns sum((0,1)) of the 4D array. 
    """
    imgbywin = image_by_windows(data4D.shape, winXY, skip = skip)
    npts = len(imgbywin.grid)
    if use_mp:
        from lognflow import multiprocessor
        data4D_cpy = multiprocessor(
            conv_4D_single, imgbywin.grid, (imgbywin, data4D), verbose = True)
    else:
        pbar = printprogress(
            len(imgbywin.grid),
            title = f'conv_4D for {len(imgbywin.grid)} windows')
        for gcnt, grc in enumerate(imgbywin.grid):
            gr, gc = grc
            view = data4D[gr:gr + imgbywin.win_shape[0], 
                          gc:gc + imgbywin.win_shape[1]].copy()
            _, stat = conv_function(view)
            if gcnt == 0:
                data4D_cpy = np.zeros((npts, ) + stat.shape, dtype = stat.dtype)
            data4D_cpy[gcnt] = stat.copy()
            pbar()
    data4D_cpy = data4D_cpy.reshape(
        imgbywin.grid_shape + (data4D_cpy.shape[1], data4D_cpy.shape[2]))
    return data4D_cpy

def bin_image(data, factor = 2, logger = None):
    """ bin image rapidly, simply by summing every "factor" number of pixels.
    :param data: 
        must have at least 2 dimensions 
    :param factor:
        data will be binned rapidly by the given factor. it 2 by default.
    :param logger:
        should have a __call__, it is print by default.
    """
    assert factor == int(factor), f'Binning factor must be integer, it is {factor}'
    data_shape = data.shape
    n_x, n_y = data_shape[0], data_shape[1]
    if len(data_shape) > 2:
        data_summed = np.zeros((n_x - factor + 1, n_y - factor + 1, *data_shape[2:]),
                               dtype = data.dtype)
    else:
        data_summed = np.zeros((n_x - factor + 1, n_y - factor + 1), 
                               dtype = data.dtype)
    if logger is not None:
        logger(f'bin_image start for dataset of shape {data_shape}...')
    
    fh = int(factor/2)
    
    for indi, indj in product(list(range(factor)), list(range(factor))):
        rend = -fh + indi
        cend = -fh + indj
        if rend == 0: rend = n_x
        if cend == 0: cend = n_y
        data_summed += data[fh - 1 + indi:rend, fh - 1 + indj:cend].copy()

    data_binned = data_summed[::factor, ::factor]
        
    if logger is not None:
        logger(f'... bin_image done with shape {data_binned.shape}')
    return data_binned

def bin_4D(data4D, 
           n_pos_in_bin: int = 1, n_pix_in_bin: int = 1,
           method_pos: str = 'skip', method_pix: str = 'linear',
           conv_function = sum_4D, skip = (1, 1), logger = None):
    """
    options for methods are: skip, linear and conv
    """
    data4D = data4D.copy()
    if(n_pos_in_bin > 1):
        if(method_pos == 'skip'):
            data4D = data4D[::n_pos_in_bin, ::n_pos_in_bin]
        if(method_pos == 'linear'):
            data4D = bin_image(data4D, n_pos_in_bin, logger = logger)
        if(method_pos == 'conv'):
                data4D = conv_4D(
                    data4D, (n_pos_in_bin, n_pos_in_bin), conv_function,
                    skip = skip)
    if(n_pix_in_bin > 1):
        if(method_pix == 'skip'):
            data4D = data4D[:, :, ::n_pix_in_bin, ::n_pix_in_bin]
        if(method_pix == 'linear'):
            data4D = data4D.swapaxes(
                1,2).swapaxes(0,1).swapaxes(2,3).swapaxes(1,2)
            data4D = bin_image(data4D, n_pix_in_bin, logger = logger)
            data4D = data4D.swapaxes(
                1,2).swapaxes(0,1).swapaxes(2,3).swapaxes(1,2)
        if(method_pix == 'conv'):
            data4D = data4D.swapaxes(
                1,2).swapaxes(0,1).swapaxes(2,3).swapaxes(1,2)
            data4D = conv_4D(
                data4D, (n_pix_in_bin, n_pix_in_bin), conv_function,
                skip = (n_pix_in_bin, n_pix_in_bin))
            data4D = data4D.swapaxes(
                1,2).swapaxes(0,1).swapaxes(2,3).swapaxes(1,2)
    return data4D

def std_4D(data4D, mask4D = None):
    """ Annular virtual detector
            Given a 4D dataset, n_x x n_y x n_r x n_c.
            the output is the marginalized images over the n_x, n_y or n_r,n_c
        
        :param data4D:
            data in 4 dimension real_x x real_y x k_r x k_c
        :param mask4D: np.ndarray
            a 4D array, optionally, calculate the CoM only in the areas 
            where mask==True
    """
    if mask4D is not None:
        assert mask4D.shape == data4D.shape,\
            'mask4D should have the same shape as data4D'
    data4D_shape = data4D.shape
    I4D_cpy = data4D.copy()
    if mask4D is not None:
        I4D_cpy *= mask4D
    PACBED_mu = I4D_cpy.sum((0, 1))
    totI = I4D_cpy.sum((2, 3))
    
    if mask4D is not None:
        mask4D_PACBED = mask4D.sum((0, 1))
        mask4D_totI = mask4D.sum((2, 3))
                                 
        PACBED_mu[mask4D_PACBED > 0] /= mask4D_PACBED[mask4D_PACBED > 0]
        PACBED_mu[mask4D_PACBED == 0] = 0
        
        totI[mask4D_totI > 0] /= mask4D_totI[mask4D_totI > 0]
        totI[mask4D_totI == 0] = 0

    PACBED_mu = np.expand_dims(PACBED_mu, (0, 1))
    PACBED_mu = np.tile(PACBED_mu, (data4D_shape[0], data4D_shape[1], 1, 1))
    _, PACBED_norm = sum_4D((I4D_cpy - PACBED_mu)**2, mask4D)
    PACBED = PACBED_norm.copy()
    if mask4D is not None:
        PACBED[mask4D_PACBED > 0] /= mask4D_PACBED[mask4D_PACBED>0]
        PACBED[mask4D_PACBED == 0] = 0
    PACBED = PACBED**0.5
    
    PACBED[0, 0] = 0
    PACBED[-1, -1] = 2
    
    return totI, PACBED

def CoM_torch(data4D, mask4D = None, normalize = True, 
              row_grid_cube = None, clm_grid_cube = None):
    """ modified from py4DSTEM
    
        I wish they (py4DSTEM authors) had written it as follows.
        Calculates two images - centre of mass x and y - from a 4D data4D.

    Args
    ^^^^^^^
        :param data4D: np.ndarray 
            the 4D-STEM data of shape (n_x, n_y, n_r, n_c)
        :param mask4D: np.ndarray
            a 4D array, optionally, calculate the CoM only in the areas 
            where mask==True
        :param normalize: bool
            if true, subtract off the mean of the CoM images
    Returns
    ^^^^^^^
        :returns: (2-tuple of 2d arrays), the centre of mass coordinates, (x,y)
        :rtype: np.ndarray
    """
    n_x, n_y, n_r, n_c = data4D.shape

    if mask4D is not None:
        assert mask4D.shape == data4D.shape,\
            f'mask4D with shape {mask4D.shape} should have '\
            + f'the same shape as data4D with shape {data4D.shape}.'
    if (row_grid_cube is None) | (clm_grid_cube is None):
        clm_grid, row_grid = np.meshgrid(np.arange(n_c), np.arange(n_r))
        row_grid_cube      = np.tile(row_grid,   (n_x, n_y, 1, 1))
        clm_grid_cube      = np.tile(clm_grid,   (n_x, n_y, 1, 1))
        row_grid_cube = torch.from_numpy(row_grid_cube).to(data4D.device).float()
        clm_grid_cube = torch.from_numpy(clm_grid_cube).to(data4D.device).float()
    
    if mask4D is not None:
        mass = (data4D * mask4D).sum(3).sum(2).float()
        CoMx = (data4D * row_grid_cube * mask4D).sum(3).sum(2).float()
        CoMy = (data4D * clm_grid_cube * mask4D).sum(3).sum(2).float()
    else:
        mass = data4D.sum(3).sum(2).float()
        CoMx = (data4D * row_grid_cube).sum(3).sum(2).float()
        CoMy = (data4D * clm_grid_cube).sum(3).sum(2).float()
        
    CoMx[mass!=0] = CoMx[mass!=0] / mass[mass!=0]
    CoMy[mass!=0] = CoMy[mass!=0] / mass[mass!=0]

    if normalize:
        CoMx -= CoMx.mean()
        CoMy -= CoMy.mean()

    return CoMx.float(), CoMy.float(), row_grid_cube, clm_grid_cube

def CoM_detector(det_resp):
    n_ch, n_r, n_c = det_resp.shape
    cent_x, cent_y = scipy.ndimage.center_of_mass(np.ones((n_r, n_c)) / (n_r * n_c))
    mask_coms = []
    for cnt in range(n_ch):
        mask_com_x, mask_com_y = scipy.ndimage.center_of_mass(det_resp[cnt] / det_resp[cnt].sum())
        mask_com_x -= cent_x
        mask_com_y -= cent_y
        mask_coms.append([mask_com_x, mask_com_y])
    return np.array(mask_coms)

def CoM_channel_torch(data_per_ch, mask_coms):
    com_x_ch = []
    com_y_ch = []
    for cnt, mask_com in enumerate(mask_coms):
        com_x_ch.append(data_per_ch[..., cnt] * mask_com[0])
        com_y_ch.append(data_per_ch[..., cnt] * mask_com[1])
    com_x_ch = torch.cat(
        [_.unsqueeze(-1) for _ in com_x_ch], axis = 1).mean(1, dtype=torch.float32)
    com_y_ch = torch.cat(
        [_.unsqueeze(-1) for _ in com_y_ch], axis = 1).mean(1, dtype=torch.float32)
    return com_x_ch, com_y_ch

def centre_of_mass_4D(data4D, mask4D = None, normalize = True):
    """ modified from py4DSTEM
    
        I wish they (py4DSTEM authors) had written it as follows.
        Calculates two images - centre of mass x and y - from a 4D data4D.

    Args
    ^^^^^^^
        :param data4D: np.ndarray 
            the 4D-STEM data of shape (n_x, n_y, n_r, n_c)
        :param mask4D: np.ndarray
            a 4D array, optionally, calculate the CoM only in the areas 
            where mask==True
        :param normalize: bool
            if true, subtract off the mean of the CoM images
    Returns
    ^^^^^^^
        :returns: (2-tuple of 2d arrays), the centre of mass coordinates, (x,y)
        :rtype: np.ndarray
    """
    n_x, n_y, n_r, n_c = data4D.shape
    data4D_dtype = data4D.dtype

    if mask4D is not None:
        assert mask4D.shape == data4D.shape,\
            f'mask4D with shape {mask4D.shape} should have '\
            + f'the same shape as data4D with shape {data4D.shape}.'
    
    data4D = data4D.copy()
    stem = data4D.mean((2, 3))
    stem = np.expand_dims(np.expand_dims(stem, -1), -1)
    stem = np.tile(stem, (1, 1, n_r, n_c))
    data4D[stem != 0] /= stem[stem != 0]
    data4D[stem == 0] = 0
    
    clm_grid, row_grid = np.meshgrid(np.arange(-n_c//2, n_c//2),
                                     np.arange(-n_r//2, n_r//2))
    row_grid_cube      = np.tile(row_grid,   (n_x, n_y, 1, 1))
    clm_grid_cube      = np.tile(clm_grid,   (n_x, n_y, 1, 1))
    
    if mask4D is not None:
        mass = (data4D * mask4D).sum(3).sum(2).astype(data4D_dtype)
        CoMx = (data4D * row_grid_cube * mask4D).sum(3).sum(2).astype(data4D_dtype)
        CoMy = (data4D * clm_grid_cube * mask4D).sum(3).sum(2).astype(data4D_dtype)
    else:
        mass = data4D.sum(3).sum(2).astype(data4D_dtype)
        CoMx = (data4D * row_grid_cube).sum(3).sum(2).astype(data4D_dtype)
        CoMy = (data4D * clm_grid_cube).sum(3).sum(2).astype(data4D_dtype)
        
    CoMx[mass!=0] = CoMx[mass!=0] / mass[mass!=0]
    CoMy[mass!=0] = CoMy[mass!=0] / mass[mass!=0]

    if normalize:
        CoMx -= CoMx.mean()
        CoMy -= CoMy.mean()

    return CoMx.astype(data4D_dtype), CoMy.astype(data4D_dtype)

def cross_correlation_4D(data4D_a, data4D_b, mask4D = None):
    
    assert data4D_a.shape == data4D_b.shape, \
        'data4D_a should have same shape as data4D_b'
    if mask4D is not None:
        assert mask4D.shape == data4D_a.shape,\
            'mask4D should have the same shape as data4D_a'

    data4D_a = normalize_4D(data4D_a.copy(), mask4D)
    data4D_b = normalize_4D(data4D_b.copy(), mask4D)
    corr_mat, _  = sum_4D(data4D_a * data4D_b, mask4D)
    
    if mask4D is not None:
        mask_STEM = mask4D.sum(3).sum(2)
        corr_mat[mask_STEM>0] /= mask_STEM[mask_STEM>0]
        corr_mat[mask_STEM == 0] = 0
    else:
        corr_mat = corr_mat / data4D_a.shape[2] / data4D_a.shape[3]
    return corr_mat

def locate_atoms(stem, min_distance = 3, min_distance_init = 3,
                 maxfilter_size = 0, reject_too_close = False,
                 rgflib_fitBackground_kwargs = None, logger = None):
    
    n_x, n_y = stem.shape
    
    nSTEM = stem.max() - stem.copy()
    
    from skimage.feature import peak_local_max
    import scipy.ndimage
    
    if rgflib_fitBackground_kwargs is not None:
        try:
            from RobustGaussianFittingLibrary import fitBackground
        except Exception as e:
            print('You need to >>> pip install RobustGaussianFittingLibrary')
            raise e
        if logger is not None: logger('getting mp')
        mp = fitBackground(nSTEM, **rgflib_fitBackground_kwargs)
        if logger is not None: logger('mp calculated!')
        SNR = nSTEM - mp[0]
        mpstd = mp[1]
        SNR[mpstd > 0] /= mpstd[mpstd > 0]
        SNR[mpstd == 0] = 0
        nSTEM = SNR.copy()
    
    if maxfilter_size:
        if logger is not None: logger('max filter!')
        image_max = scipy.ndimage.maximum_filter(
            nSTEM, size=maxfilter_size, mode='constant')
    else:
        image_max = nSTEM.copy()
    if logger is not None: logger('finding peak local max!')
    coordinates = peak_local_max(image_max, min_distance=min_distance_init)
    
    if(reject_too_close):
        inds = []
        if logger is not None: logger('rejecting too close ones!')
        dist_coord_to_com = np.zeros(len(coordinates))
        move_by_com = np.zeros((len(coordinates), 2))
        if logger is not None: pbar = printprogress(len(coordinates),
                                                    print_function = logger)
        for ccnt, coord in enumerate(coordinates):
            coord_0, coord_1 = coord
            r_start = coord_0 - min_distance
            r_end   = coord_0 + min_distance + 1
            c_start = coord_1 - min_distance
            c_end   = coord_1 + min_distance + 1
            
            if ( r_end >= n_x):
                r_end = n_x
                r_start = 2 * coord_0 - r_end
            if ( r_start < 0):
                r_start = 0
                r_end = 2 * coord_0
            if ( c_end >= n_y):
                c_end = n_y
                c_start = 2 * coord_1 - c_end
            if ( c_start < 0):
                c_start = 0
                c_end = 2 * coord_1
            
            local_stem = nSTEM[r_start: r_end, c_start: c_end].copy()

            cy, cx = scipy.ndimage.center_of_mass(local_stem)
            cx += 0.5
            cy += 0.5
            move_by_com[ccnt] = np.array([cx - local_stem.shape[0]/2,
                                          cy - local_stem.shape[1]/2])
            dist_coord_to_com[ccnt] = (
                move_by_com[ccnt, 0]**2 + move_by_com[ccnt, 1]**2)**0.5
            if logger is not None: pbar()
        
        if logger is not None: logger('getting typical distances!')
        try:
            from RobustGaussianFittingLibrary import fitValue
        except Exception as e:
            print('You need to >>> pip install RobustGaussianFittingLibrary')
            raise e
        dist2 = scipy.spatial.distance.cdist(coordinates, coordinates)
        dist2 = dist2 + np.diag(np.inf + np.zeros(coordinates.shape[0]))
        dist2_min = dist2.min(1)
        mP = fitValue(dist2_min, MSSE_LAMBDA = 2.0)
        dist2_threshold = mP[0] / 2
        dist2_threshold = np.minimum(dist2_threshold, dist2.min(1).mean())
        dist2_cpy = dist2.copy()
        
        if logger is not None: logger('keeping those with normal distances!')
        for single_ind, single_dist2 in enumerate(dist2_cpy):
            _tmp = dist_coord_to_com[single_dist2 < dist2_threshold].copy()
            if _tmp.any():
                current_com = dist_coord_to_com[single_ind]
                best_com = _tmp.min()
                if current_com < best_com:
                    inds.append(single_ind)
            else:
                inds.append(single_ind)
        coordinates = coordinates + move_by_com
        coordinates = coordinates[np.array(inds)]
        
    return coordinates

def stem_image_nyquist_interpolation(
        StemImage, xlen, ylen, alpha, Knought, npixout, npiyout):
    """
    Nyquist interpolates a STEM image using Fourier methods.
    STEMImage has real space dimensions ylen and xlen in Angstrom.

    Parameters:
    - StemImage: Input 2D STEM image.
    - xlen, ylen: Real space dimensions in Angstrom.
    - alpha: Probe-forming aperture semiangle in mrad.
    - Knought: Vacuum wavevector (in inverse Angstrom).
    - npixout, npiyout: Number of pixels in the output image (x, y).

    Returns:
    - StemImageInterpolated: Upsampled 2D STEM image.
    """
    npix, npiy = np.shape(StemImage)
    qalpha = Knought * alpha * 1.0e-3
    qband = 2.0 * qalpha
    qnyq = 2.0 * qband

    npixmin = np.ceil(xlen * qnyq)
    npiymin = np.ceil(ylen * qnyq)

    if npix < npixmin or npiy < npiymin:
        print('Input STEM image is insufficiently sampled for Nyquist interpolation')

    ctemp2 = np.fft.fftshift(np.fft.fft2(StemImage))
    ctemp = np.zeros((npixout, npiyout), dtype=complex)

    center_y, center_x = npiyout // 2, npixout // 2
    start_y, start_x = center_y - npiy // 2, center_x - npix // 2
    ctemp[start_x:start_x + npix, start_y:start_y + npiy] = ctemp2

    ctemp = np.fft.ifft2(np.fft.ifftshift(ctemp))
    StemImageInterpolated = np.real(ctemp)

    StemImageInterpolated *= (npixout * npiyout) / (npix * npiy)

    return StemImageInterpolated

def upsample_4d_data(data4d, xlen, ylen, alpha, Knought, npixout, npiyout):
    """
    Upsamples a 4-dimensional dataset in real space.

    Parameters:
    - data4d: Input 4D dataset.
    - xlen, ylen: Real space dimensions in Angstrom.
    - alpha: Probe-forming aperture semiangle in mrad.
    - Knought: Vacuum wavevector (in inverse Angstrom).
    - npixout, npiyout: Number of pixels in the output image (x, y).

    Returns:
    - data4d_upsampled: Upsampled 4D dataset.
    """
    data4d_shape = data4d.shape
    data4d = data4d.reshape(data4d_shape[0], data4d_shape[1], -1)
    data4d_upsampled = np.zeros(
        (npixout, npiyout, data4d.shape[2]), dtype=data4d.dtype)
    
    for pix_cnt in range(data4d.shape[2]):
        data4d_upsampled[:, :, pix_cnt] = stem_image_nyquist_interpolation(
            StemImage=data4d[:, :, pix_cnt].copy(),xlen=xlen, ylen=ylen, 
            alpha=alpha, Knought=Knought,npixout=npixout, npiyout=npiyout)

    data4d_upsampled = data4d_upsampled.reshape(
        npixout, npiyout, data4d_shape[2], data4d_shape[3])

    return data4d_upsampled

def stem_4d_nyquist_interpolation_fourier(
        data4d, xlen, ylen, alpha, Knought, npixout, npiyout):
    """
    Nyquist interpolates a 4D STEM dataset in real space using 4D Fourier methods.
    Each STEM image has real space dimensions ylen and xlen in Angstrom.

    Parameters:
    - data4d: Input 4D STEM dataset (n_x, n_y, n_r, n_c).
    - xlen, ylen: Real space dimensions in Angstrom.
    - alpha: Probe-forming aperture semiangle in mrad.
    - Knought: Vacuum wavevector (in inverse Angstrom).
    - npixout, npiyout: Number of pixels in the output image (x, y).

    Returns:
    - data4d_upsampled: Upsampled 4D STEM dataset (npixout, npiyout, n_r, n_c).
    """
    n_x, n_y, n_r, n_c = data4d.shape

    # Ensure the output size in real space (npixout, npiyout) is valid
    if npixout < n_x or npiyout < n_y:
        raise ValueError(f"Output dimensions ({npixout}, {npiyout}) must be >= real-space input dimensions ({n_x}, {n_y}).")

    # Compute Nyquist parameters
    qalpha = Knought * alpha * 1.0e-3
    qband = 2.0 * qalpha
    qnyq = 2.0 * qband

    npixmin = np.ceil(xlen * qnyq)
    npiymin = np.ceil(ylen * qnyq)

    if n_c < npixmin or n_r < npiymin:
        print('Warning: Input 4D STEM dataset is insufficiently sampled for Nyquist interpolation.')

    # Perform the 4D Fourier transform
    ctemp2 = np.fft.fftshift(np.fft.fftn(data4d, axes=(2, 3)), axes=(2, 3))

    ctemp = mcemtools.masking.crop_or_pad(ctemp2, (npixout, npiyout, n_r, n_c))

    # # Create a larger 4D array to hold the interpolated Fourier components
    # ctemp = np.zeros((npixout, npiyout, n_r, n_c), dtype=complex)
    #
    # # Compute insertion indices for the 4D FFT data
    # pad_y = (npiyout - n_y) // 2
    # pad_x = (npixout - n_x) // 2
    #
    # start_y = max(0, pad_y)  # Prevent negative indices
    # start_x = max(0, pad_x)
    # end_y = start_y + n_y
    # end_x = start_x + n_x
    #
    # # Verify compatibility
    # if (end_y - start_y != n_y) or (end_x - start_x != n_x):
    #     raise ValueError("Mismatch between insertion region and real-space input dimensions.")
    #
    # # Insert the FFT data into the center of the larger array
    # ctemp[start_y:end_y, start_x:end_x, :, :] = ctemp2

    # Perform the inverse 4D FFT and shift back
    ctemp = np.fft.ifftn(np.fft.ifftshift(ctemp, axes=(2, 3)), axes=(2, 3))
    data4d_upsampled = np.real(ctemp)

    # Normalize intensity
    data4d_upsampled *= (npixout * npiyout) / (n_x * n_y)

    return data4d_upsampled

def force_stem_4d(a4d, b4d):
    """ force stem
        force the stem image of the dataset a to be the stem image of the dataset b.
    """
    
    stem = a4d.mean((2, 3))
    stem = np.expand_dims(np.expand_dims(stem, -1), -1)
    stem = np.tile(stem, (1, 1, a4d.shape[2],a4d.shape[3]))
    a4d[stem != 0] /= stem[stem != 0]
    a4d[stem == 0] = 0
    stem = b4d.mean((2, 3))
    stem = np.expand_dims(np.expand_dims(stem, -1), -1)
    stem = np.tile(stem, (1, 1, a4d.shape[2],a4d.shape[3]))
    a4d[stem != 0] *= stem[stem != 0]
    a4d[stem == 0] = 0
    return a4d

def compute_pixel_histograms(images, bins):
    """
    Compute per-pixel histograms across a stack of images.

    Each pixel position (row, column) in the image stack is analyzed
    independently to estimate how its intensity values are distributed
    across the provided bins. The result is a 3D array representing
    normalized histograms (probability distributions) for every pixel.

    Parameters
    ----------
    images : np.ndarray
        Array of shape (n_images, height, width) or (n_images, ...),
        containing a stack of 2D (or higher-dimensional) images.
        Each image should have the same shape and dtype (numeric).
    bins : np.ndarray
        Array of bin edges (length = num_bins + 1) defining the histogram
        intervals, e.g. from `np.linspace(min_val, max_val, num_bins + 1)`.

    Returns
    -------
    histograms : np.ndarray
        Array of shape (num_bins, height, width), containing the normalized
        per-pixel histograms.
        Each element `histograms[b, i, j]` gives the fraction of images
        whose pixel at position (i, j) falls into bin `b`.

    Notes
    -----
    - The histograms are normalized by the number of input images
      (`n_images`), so the sum across bins for each pixel equals 1.0.
    - Uses `np.digitize` internally, so the last bin includes values
      that equal its right edge.

    Example
    -------
    >>> images = np.random.rand(100, 64, 64)
    >>> bins = np.linspace(0, 1, 11)
    >>> hists = compute_pixel_histograms(images, bins)
    >>> hists.shape
    (10, 64, 64)
    """
    n_images = len(images)
    num_bins = len(bins) - 1

    # Initialize histogram array: (num_bins, height, width)
    histograms = np.zeros((num_bins,) + images.shape[1:], dtype=int)

    # Assign each pixel in all images to a bin index (0 to num_bins-1)
    binned = np.digitize(images, bins=bins) - 1

    # Count how often each pixel falls into each bin
    for b in range(num_bins):
        histograms[b] = np.sum(binned == b, axis=0)

    # Normalize to convert counts to probabilities
    histograms = histograms.astype('float32') / float(n_images)

    return histograms.astype('float32')

def find_cdf_divisions(cdf, x_vals, M):
    """
    Divide a CDF into M equally spaced probability intervals and find
    the corresponding x-value thresholds.

    Parameters
    ----------
    cdf : np.ndarray
        Monotonically increasing array of CDF values (between 0 and 1).
    x_vals : np.ndarray
        Corresponding x-values for the CDF.
    M : int
        Number of desired bins.

    Returns
    -------
    targets : np.ndarray
        Target CDF values (quantiles) at which thresholds are determined.
    thresholds : np.ndarray
        Corresponding x-values that divide the data into M equal-probability bins.
    """
    # Define target CDF values (avoid exactly 0 and 1 to stay within interpolation range)
    targets = np.linspace(1 / M, 1 - 1 / M, M - 1)
    
    # Interpolate to find x thresholds corresponding to those CDF levels
    thresholds = np.interp(targets, cdf, x_vals)
    
    return targets, thresholds

def test_find_cdf_divisions():
    # ==== Example: bimodal distribution ====

    # Generate data: two normal distributions (each 10,000 samples)
    np.random.seed(0)
    data1 = np.random.normal(0, 0.5, 10_000)
    data2 = np.random.normal(2, 0.5, 10_000)
    data = np.concatenate([data1, data2])

    # Sort data for CDF computation
    x_vals = np.sort(data)
    cdf = np.arange(1, len(x_vals) + 1) / len(x_vals)

    # Find divisions using CDF-based method
    M = 16  # number of desired bins
    targets, thresholds = find_cdf_divisions(cdf, x_vals, M)

    # ==== Plot CDF with target lines ====
    plt.figure(figsize=(8, 4))
    plt.plot(x_vals, cdf, label="CDF", color='C0')
    for t, thr in zip(targets, thresholds):
        plt.axhline(t, color='gray', linestyle='--', linewidth=0.8)
        plt.axvline(thr, color='red', linestyle='--', linewidth=1)
    plt.title("CDF and Equal-Probability Divisions")
    plt.xlabel("x")
    plt.ylabel("CDF")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # ==== Plot histogram and overlay thresholds ====
    plt.figure(figsize=(8, 4))
    plt.hist(data, bins=100, density=True, color='lightblue', edgecolor='k')
    for thr in thresholds:
        plt.axvline(thr, color='red', linestyle='--', linewidth=1)
    plt.title("Histogram with Equal-Probability Bin Thresholds")
    plt.xlabel("x")
    plt.ylabel("Density")
    plt.grid(True)
    plt.tight_layout()

    plt.show()

def get_cc(vec_a: torch.Tensor, vec_b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Cross-correlation between two 1D vectors (zero-mean, unit-std).
    Matches user's definition but stabilized a bit to avoid div-by-zero.
    """
    va = vec_a.view(-1).float()
    vb = vec_b.view(-1).float()
    a_std = va.std(unbiased=False)
    b_std = vb.std(unbiased=False)
    if a_std.item() == 0 or b_std.item() == 0:
        return torch.tensor(0.0, device=va.device)
    vec_1 = (va - va.mean()) / (a_std + eps)
    vec_2 = (vb - vb.mean()) / (b_std + eps)
    return (vec_1 * vec_2).mean()

def build_affine_theta(trans_row: float, trans_col: float,
                       scale_row: float, scale_col: float,
                       rot_angle: float,
                       H: int, W: int, device: torch.device):
    """
    Build 2x3 affine theta in normalized coords for F.affine_grid.
    Parameters trans_row/trans_col are in PIXELS.
    scale_row: scaling along image rows (y axis).
    scale_col: scaling along image cols (x axis).
    rot_angle: radians (positive = CCW).
    """
    # rotation
    c = torch.cos(torch.tensor(rot_angle, device=device))
    s = torch.sin(torch.tensor(rot_angle, device=device))

    # linear part in physical (x,y) pixel axes: A = R @ S
    # note: x corresponds to columns, y to rows
    A = torch.zeros((2, 2), device=device)
    # S = diag(scale_col, scale_row)
    A[0, 0] = c * scale_col   # a11 (x <- x)
    A[0, 1] = -s * scale_row  # a12 (x <- y)
    A[1, 0] = s * scale_col   # a21 (y <- x)
    A[1, 1] = c * scale_row   # a22 (y <- y)

    # convert pixel translations to normalized coords in [-1,1] (align_corners=True)
    if W > 1:
        tx_norm = 2.0 * trans_col / (W - 1)
    else:
        tx_norm = 0.0
    if H > 1:
        ty_norm = 2.0 * trans_row / (H - 1)
    else:
        ty_norm = 0.0

    theta = torch.zeros((2, 3), device=device)
    theta[:, :2] = A
    theta[0, 2] = tx_norm
    theta[1, 2] = ty_norm
    return theta

def warp_image_with_params(in_img: torch.Tensor,
                           trans_row: float = 0, trans_col: float = 0,
                           scale_row: float = 1, scale_col: float = 1,
                           rot_angle: float = 0):
    """
    Warp in_img using the affine parameters. in_img shape can be (H,W) or (1,H,W) or (C,H,W).
    Returns warped image same shape as input (except channel preserved).
    """
    # normalize shapes
    x = in_img
    if x.dim() == 2:
        x = x.unsqueeze(0).unsqueeze(0)  # 1x1xHxW
        squeeze_out = True
    elif x.dim() == 3:
        # C x H x W -> add batch
        x = x.unsqueeze(0)               # 1xC x H x W
        squeeze_out = True
    elif x.dim() == 4:
        # assume already has batch (not supported)
        squeeze_out = False
    else:
        raise ValueError("in_img must be HxW, CxHxW, or BxCxHxW")

    B, C, H, W = x.shape
    device = x.device

    theta = build_affine_theta(trans_row, trans_col, scale_row, scale_col, rot_angle, H, W, device)
    theta = theta.unsqueeze(0)  # batch dim

    # affine_grid + grid_sample (align_corners=True for consistent normalization)
    grid = F.affine_grid(theta, size=(B, C, H, W), align_corners=True)  # BxHxWx2
    warped = F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    if squeeze_out:
        # remove batch dimension and if originally single-channel, remove channel too
        warped = warped.squeeze(0)
        if in_img.dim() == 2:
            warped = warped.squeeze(0)
    return warped

def register_affine(in_image: torch.Tensor,
                    target_image: torch.Tensor,
                    n_iters: int = 200,
                    lr: float = 1e-1,
                    eps_numdiff: float = 1e-3,
                    verbose: bool = False,
                    device: torch.device = None) -> Tuple[float, float, float, float, float, torch.Tensor]:
    """
    Perform affine registration to maximize cross-correlation (get_cc).
    Returns: (trans_row, trans_col, scale_row, scale_col, rot_angle, warped_image)
    Notes:
      - in_image and target_image should have same H,W. Accepts HxW or 1xHxW / CxHxW.
      - Uses central finite differences for directional derivatives of the loss
        (loss = - get_cc(warped, target)).
    """
    # dispatch device
    if device is None:
        device = in_image.device if isinstance(in_image, torch.Tensor) else torch.device('cpu')

    # prepare tensors on device and float
    inp = in_image.to(device=device, dtype=torch.float32)
    tgt = target_image.to(device=device, dtype=torch.float32)

    # ensure shape compatibility: use single-channel registration (flatten across channels)
    # if multi-channel, we'll average channels when computing CC
    def flatten_for_cc(img):
        # img may be HxW, CxHxW, or 1xHxW
        t = img
        if t.dim() == 2:
            return t.reshape(-1)
        elif t.dim() == 3:
            # channels present -> flatten across channels too
            return t.reshape(-1)
        elif t.dim() == 4:
            return t.reshape(-1)
        else:
            raise ValueError("Unsupported image dims for CC")

    # check shapes
    # unify to at least CxHxW or HxW
    if inp.dim() not in (2, 3):
        raise ValueError("in_image must be HxW or CxHxW")
    if tgt.dim() not in (2, 3):
        raise ValueError("target_image must be HxW or CxHxW")

    # ensure same H,W
    H_in = inp.shape[-2]
    W_in = inp.shape[-1]
    H_t = tgt.shape[-2]
    W_t = tgt.shape[-1]
    if (H_in != H_t) or (W_in != W_t):
        raise ValueError("in_image and target_image must have same H,W")

    # initialize parameters (start with identity)
    trans_row = 0.0
    trans_col = 0.0
    scale_row = 1.0
    scale_col = 1.0
    rot_angle = 0.0  # radians

    params = ['trans_row', 'trans_col', 'scale_row', 'scale_col', 'rot_angle']
    # list-style for convenience
    pvals = torch.tensor([trans_row, trans_col, scale_row, scale_col, rot_angle], device=device, dtype=torch.float32)

    # small helper to compute loss for a given param vector
    def compute_loss_from_pvals(pvec):
        tr = float(pvec[0].item())
        tc = float(pvec[1].item())
        sr = float(pvec[2].item())
        sc = float(pvec[3].item())
        ra = float(pvec[4].item())
        warped = warp_image_with_params(inp, tr, tc, sr, sc, ra)
        # flatten to vectors
        wa = flatten_for_cc(warped)
        tb = flatten_for_cc(tgt)
        cc = get_cc(wa, tb)
        # we want to maximize CC, so loss = -cc
        return -cc, warped

    # main optimization loop
    for it in range(n_iters):
        base_loss, base_warped = compute_loss_from_pvals(pvals)

        # compute numerical gradient for each parameter using central difference
        grads = torch.zeros_like(pvals)
        for i in range(len(pvals)):
            # perturb +/- eps
            p_plus = pvals.clone()
            p_minus = pvals.clone()
            p_plus[i] += eps_numdiff
            p_minus[i] -= eps_numdiff

            loss_plus, _ = compute_loss_from_pvals(p_plus)
            loss_minus, _ = compute_loss_from_pvals(p_minus)
            # central difference
            grad_i = (loss_plus - loss_minus) / (2.0 * eps_numdiff)
            grads[i] = grad_i

        # gradient descent step: p := p - lr * grad
        # use small learning rates for translation in pixels vs angle vs scale;
        # user-specified lr is global; the user can tune. Here we make a simple per-param scaling:
        # scale translations in pixels more conservatively than rotation/scale or we can keep uniform.
        # I'll apply a small nominal scaling factor to make default lr reasonable:
        per_param_scale = torch.tensor([1.0, 1.0, 0.5, 0.5, 0.5], device=device)  # tuneable heuristic
        pvals = pvals - (lr * per_param_scale * grads)

        # optional: clamp scales to reasonable positive range to avoid flips
        pvals[2] = torch.clamp(pvals[2], 0.1, 5.0)  # scale_row
        pvals[3] = torch.clamp(pvals[3], 0.1, 5.0)  # scale_col

        if verbose and (it % max(1, n_iters // 10) == 0 or it == n_iters - 1):
            print(f"iter {it+1}/{n_iters} loss={base_loss.item():.6f} params={pvals.cpu().numpy()}")

    # final warp with optimized params
    final_loss, final_warped = compute_loss_from_pvals(pvals)
    # unpack final params
    trans_row_f = float(pvals[0].item())
    trans_col_f = float(pvals[1].item())
    scale_row_f = float(pvals[2].item())
    scale_col_f = float(pvals[3].item())
    rot_angle_f = float(pvals[4].item())

    # return final parameters and warped image (same shape as input)
    return trans_row_f, trans_col_f, scale_row_f, scale_col_f, rot_angle_f, final_warped
    
def test_register_affine():
    in_img = np.random.rand(100, 100)
    target_img = in_img[:80, 80]
    target_img = mcemtools.masking.crop_or_pad(target_img, (100, 100))
    tr, tc, sr, sc, ra, warped = mcemtools.analysis.register_affine(in_img, target_img,
                                                    n_iters=1200,
                                                    lr=0.5,
                                                    eps_numdiff=1e-2,
                                                    verbose=True,
                                                    device=torch.device('cpu'))
    
    print("Estimated params:", tr, tc, sr, sc, ra)
    
    from lognflow.plt_utils import plt_imshow_subplots, plt
    plt_imshow_subplots([in_img, target_img, warped], titles = ['in_img', 'target_img', 'warped'])
    plt.show()