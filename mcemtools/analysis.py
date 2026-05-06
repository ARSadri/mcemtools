import numpy as np
from .masking import annular_mask, mask2D_to_4D, image_by_windows
from lognflow import printprogress, getLogger, has_len
from skimage.transform import warp_polar
import scipy
from itertools import product
import torch
import mcemtools
import torch.nn.functional as F
from typing import Tuple

def PACBEDs_ch_torch(x, win_shape):
    """
    x: (n_x, n_y, n_ch) on CUDA
    returns: (n_x - n_xw + 1, n_y - n_yw + 1, n_ch)
    """
    n_xw, n_yw = win_shape

    # Move to NCHW
    x = x.permute(2, 0, 1).unsqueeze(0)  # (1, n_ch, n_x, n_y)

    n_ch = x.shape[1]

    # Box filter kernel (one per channel)
    kernel = torch.ones(
        (n_ch, 1, n_xw, n_yw),
        device=x.device,
        dtype=x.dtype
    )

    # Grouped convolution = per-channel window sum
    y = F.conv2d(
        x,
        kernel,
        groups=n_ch
    )

    # Back to (x, y, ch)
    return y.squeeze(0).permute(1, 2, 0)

def images_center_of_mass(images, segments_CoM, eps=1e-12):
    """
    images: (n_images, num_segments)
    segments_CoM: (num_segments, 2)
    returns: (n_images, 2)
    """
    weights_sum = images.sum(dim=1, keepdim=True)     # (n_images, 1)
    com = images @ segments_CoM                       # (n_images, 2)
    return com / (weights_sum + eps)


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
    PACBED = I4D_cpy.sum((0, 1))
    STEM = I4D_cpy.sum((2, 3))
    return STEM, PACBED

def bin_4D(data, real_shape=None, qspace_shape=None, order = 1, preserve_range=True, **kwargs):
    """
    Bin a 4D-STEM dataset.

    Parameters
    ----------
    data : ndarray
        Input array of shape (n_x, n_y, n_r, n_c).
    real_shape : tuple or None
        Desired output shape for scan dimensions (nx_out, ny_out).
    qspace_shape : tuple or None
        Desired output shape for diffraction dimensions (nr_out, nc_out).
    preserve_range : bool
        Keep original value scaling.

    Returns
    -------
    ndarray
        Binned dataset of shape (nx_out, ny_out, nr_out, nc_out).
    """

    #-----------------
    # n_pos_in_bin: int = 1, n_pix_in_bin: int = 1,
    # method_pos: str = 'skip', method_pix: str = 'linear',
    # conv_function = sum_4D, skip = (1, 1), logger = None
    n_x, n_y, n_r, n_c = data.shape

    if 'n_pos_in_bin' in kwargs:
        if real_shape is None:
            real_shape = (n_x // kwargs['n_pos_in_bin'], n_y // kwargs['n_pos_in_bin'])
    if 'n_pix_in_bin' in kwargs:
        if qspace_shape is None:
            qspace_shape = (n_r // kwargs['n_pix_in_bin'], n_c // kwargs['n_pix_in_bin'])
    #-----------------

    from skimage.transform import resize

    if real_shape is None:
        real_shape = (n_x, n_y)
    if qspace_shape is None:
        qspace_shape = (n_r, n_c)

    data_rs = resize(
        data,
        (real_shape[0], real_shape[1], n_r, n_c),
        order=order,
        anti_aliasing=True,
        preserve_range=preserve_range
    )

    data_q = resize(
        data_rs,
        (real_shape[0], real_shape[1], qspace_shape[0], qspace_shape[1]),
        order=order,
        anti_aliasing=True,
        preserve_range=preserve_range
    )

    return data_q.astype(data.dtype)

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

def bin_4D_conv(data4D, 
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

def locate_atoms(
        snr,
        snr_threshold=1.0,
        snr_mask = None,
        min_n_pix = 1,
        peak_win_shape=(2, 2),   # (height, width) of window for refinement
        refinement_iters=3,
        margin = (0, 0),
        merge_dist = 3
    ):
    """
    Find refined analog center-of-mass (COM) coordinates of peaks in an SNR image.

    Parameters
    ----------
    snr : 2D array
        The SNR image of bright field, obtaibed via BrightField_STEM/DarkField_STEM
    threshold : float
        snr_mask = (snr > threshold).
    peak_win_shape : tuple(int, int)
        Full window size (height, width). E.g. (5,5) means a 5×5 region
        centered on the current peak for COM refinement.
    refinement_iters : int
        Number of COM refinement iterations.

    Returns
    -------
    refined_coords : list of (cy, cx) floats
        Refined (analog) COM coordinates for each connected component.
    mask_centers : 2D uint8 array
        Binary mask with 1 at the final rounded peak positions.
    """
    import scipy.ndimage
    import scipy.ndimage as ndi
    from scipy.signal import convolve2d

    snr = np.asarray(snr, dtype=float)
    H, W = snr.shape

    # Threshold mask
    if snr_mask is None:
        snr_mask = mcemtools.remove_islands_by_size(snr > snr_threshold, min_n_pix=min_n_pix)
        
    # Connected components
    labels, num = ndi.label(snr_mask)

    mask_centers = np.zeros_like(labels, dtype=np.uint8)
    refined_coords = []

    # Extract full window sizes
    win_h, win_w = peak_win_shape
    rad_h = win_h // 2
    rad_w = win_w // 2

    # Helper: refine once using weighted centroid in window
    def refine_once(cy, cx):
        iy = int(round(cy))
        ix = int(round(cx))

        # Window bounds
        y0 = max(0, iy - rad_h)
        y1 = min(H, iy + rad_h + 1)
        x0 = max(0, ix - rad_w)
        x1 = min(W, ix + rad_w + 1)

        patch = snr[y0:y1, x0:x1]
        if patch.size == 0:
            return cy, cx

        yy, xx = np.mgrid[y0:y1, x0:x1]
        w = patch

        wsum = w.sum()
        if wsum <= 0:
            return cy, cx

        cy_new = (yy * w).sum() / wsum
        cx_new = (xx * w).sum() / wsum

        return cy_new, cx_new

    # For each component
    for k in range(1, num + 1):
        cy0, cx0 = ndi.center_of_mass(labels == k)
        if np.isnan(cy0):
            continue

        cy, cx = cy0, cx0

        # Refinement iterations
        for _ in range(refinement_iters):
            cy, cx = refine_once(cy, cx)

        # Mark pixel position
        iy = int(round(cy))
        ix = int(round(cx))

        if margin[0] <= iy < H - margin[0] and margin[1] <= ix < W - margin[1]:
            refined_coords.append((cy, cx))
            mask_centers[iy, ix] = 1

    # ------------------------------------------------------------------
    # MERGE COORDS THAT ARE TOO CLOSE (< merge_dist), REPEAT UNTIL STABLE
    # ------------------------------------------------------------------

    coords = np.array(refined_coords, dtype=float)

    changed = True
    while changed and len(coords) > 1:
        changed = False
        N = len(coords)

        # Pairwise distances
        d2 = np.sum((coords[:, None, :] - coords[None, :, :])**2, axis=-1)
        np.fill_diagonal(d2, np.inf)

        # Find all pairs to merge
        pairs = np.argwhere(d2 < merge_dist**2)

        if len(pairs) == 0:
            break

        changed = True

        # Greedy cluster building
        used = set()
        new_coords = []

        for i, j in pairs:
            if i in used or j in used:
                continue
            cluster = {i, j}

            # Grow cluster: anything close to any member
            grow = True
            while grow:
                grow = False
                for k in range(N):
                    if k in cluster:
                        continue
                    if any(np.linalg.norm(coords[k] - coords[m]) < merge_dist for m in cluster):
                        cluster.add(k)
                        grow = True

            used |= cluster
            # Replace with average
            new_coords.append(coords[list(cluster)].mean(axis=0))

        # Add the ones not touched
        for k in range(N):
            if k not in used:
                new_coords.append(coords[k])

        coords = np.array(new_coords)

    # ---------------------------------------------------------
    # REFINEMENT AGAIN AFTER MERGING
    # ---------------------------------------------------------
    refined_coords_final = []
    mask_centers[:] = 0

    for cy, cx in coords:
        for _ in range(refinement_iters):
            cy, cx = refine_once(cy, cx)

        refined_coords_final.append((cy, cx))

        iy = int(round(cy))
        ix = int(round(cx))
        if 0 <= iy < H and 0 <= ix < W:
            mask_centers[iy, ix] = 1

    return refined_coords_final, mask_centers

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
    """ force stem from b to a
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
        
    Example
    -------
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
    """
    # Define target CDF values (avoid exactly 0 and 1 to stay within interpolation range)
    targets = np.linspace(1 / M, 1 - 1 / M, M - 1)
    
    # Interpolate to find x thresholds corresponding to those CDF levels
    thresholds = np.interp(targets, cdf, x_vals)
    
    return targets, thresholds

""" Test for the cdf divisions
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
"""

def get_cc(vec_a: torch.Tensor, vec_b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Cross-correlation between two 1D vectors (zero-mean, unit-std).
    Matches user's definition but stabilized a bit to avoid div-by-zero.
    """
    try: va = vec_a.view(-1).float()
    except: va = vec_a.ravel()
    try: vb = vec_b.view(-1).float()
    except: vb = vec_b.ravel()
    try:
        a_std = va.std(unbiased=False)
        b_std = vb.std(unbiased=False)
        if a_std.item() == 0 or b_std.item() == 0:
            return torch.tensor(0.0, device=va.device)
    except:
        a_std = va.std()
        b_std = vb.std()
        if a_std == 0 or b_std == 0:
            return 0
    vec_1 = (va - va.mean()) / (a_std + eps)
    vec_2 = (vb - vb.mean()) / (b_std + eps)
    return (vec_1 * vec_2).mean()

def affine_transform_scipy(img, shift = (0, 0), angle_deg = 0, scale = (1, 1), final_shape = None, order=1, prefilter=True):
    """
    Apply shift, rotation (degrees), and scaling (2-tuple) in a single affine transform.
    """
    if not has_len(scale): scale = (scale, scale)

    if final_shape is None:
        final_shape = img.shape
    angle = np.deg2rad(angle_deg)

    S = np.diag(scale)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    M = R @ S 

    M_inv = np.linalg.inv(M)

    # Centering: we want to keep final_shape centered after transform
    in_center = 0.5 * np.array(img.shape[::-1])  # (x, y)
    out_center = 0.5 * np.array(final_shape[::-1])
    offset = in_center - M_inv @ out_center - shift[::-1]  # reverse because of (row,col)
    from scipy.ndimage import affine_transform

    transformed = affine_transform(
        img,
        M_inv,
        offset=offset,
        output_shape=final_shape,
        order=order,
        prefilter=prefilter
    )
    return transformed

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
                    trans_row = 0.0,
                    trans_col = 0.0,
                    scale_row = 1.0,
                    scale_col = 1.0,
                    rot_angle = 0.0,
                    verbose: bool = True,
                    loss_func = get_cc,
                    fig_ax_bins = None,
                    log_period = 5,
                    device: torch.device = 'cpu') -> Tuple[float, float, float, float, float, torch.Tensor]:
    """
    Perform affine registration of ``in_image`` to ``target_image`` by maximizing
    a similarity measure (default: cross-correlation via ``get_cc``).  
    Optimization is performed on five affine parameters:

        - ``trans_row`` : translation along rows (Δy)
        - ``trans_col`` : translation along columns (Δx)
        - ``scale_row`` : scaling in the row direction
        - ``scale_col`` : scaling in the column direction
        - ``rot_angle`` : in-plane rotation angle (radians)

    The optimizer uses **central finite-difference numerical gradients** and
    gradient descent (optionally with a per-parameter learning-rate vector).

    Parameters
    ----------
    in_image : torch.Tensor
        Input image to be warped. Must have shape ``HxW`` or ``CxHxW``.
        Multi-channel inputs are flattened across channels for the similarity
        metric.

    target_image : torch.Tensor
        Fixed reference image with the same spatial size as ``in_image``.

    n_iters : int, default=200
        Number of optimization iterations.

    lr : float or sequence of float, default=1e-1
        Learning rate.  
        - If a single float: same LR is used for all parameters.  
        - If a list/tuple of length 5: per-parameter learning rates are used
          in the order
          ``[trans_row, trans_col, scale_row, scale_col, rot_angle]``.

    eps_numdiff : float, default=1e-3
        Step size for central finite-difference numerical derivatives.
        - If a single float: same eps_numdiff is used for all parameters.  
        - If a list/tuple of length 5: per-parameter epsilons are used
          in the order
          ``[trans_row, trans_col, scale_row, scale_col, rot_angle]``.

    trans_row, trans_col : float, default=0.0
        Initial translations in pixels.

    scale_row, scale_col : float, default=1.0
        Initial scale factors along row and column axes.

    rot_angle : float, default=0.0
        Initial rotation angle (radians).

    verbose : bool, default=True
        If True, prints progress and optionally shows a scatter-plot of
        warped-vs-target intensities.

    loss_func : callable, default=get_cc
        Similarity function. Must accept ``(warped, target)`` and return a
        scalar similarity score. The optimizer **maximizes** this value by
        minimizing ``loss = -similarity``.

    fig_ax_bins : int or tuple or None, default=None
        If provided, enables online visualization using ``plt_hist2`` by
        histogramming (warped, target) pixel pairs. Ignored outside notebooks.

    log_period : float, default=5
        Minimum time interval (seconds) between verbose log prints.

    device : torch.device or None, default=None
        Device for computation. If None, uses ``in_image.device`` or CPU.

    Returns
    -------
    trans_row_f : float  
        Optimized row translation.

    trans_col_f : float  
        Optimized column translation.

    scale_row_f : float  
        Optimized vertical scale.

    scale_col_f : float  
        Optimized horizontal scale.

    rot_angle_f : float  
        Optimized rotation angle in radians.

    warped_image : torch.Tensor  
        Final warped version of ``in_image`` with the same shape as input.

    Notes
    -----
    - ``in_image`` and ``target_image`` must have identical spatial dimensions.
    - Works with 2D grayscale or multi-channel images; for similarity evaluation
      all channels are flattened.
    - Numerical gradients use central finite differences:
      ``∂L/∂p ≈ (L(p+ε) - L(p-ε)) / (2ε)``.
    - The function maximizes the similarity metric by minimizing ``-similarity``.
    - ``warp_image_with_params`` must implement the affine transformation
      defined by the five parameters.
    """

    try:
        device_orig = in_image.device
    except:
        device_orig = device

    try:
        if len(lr) == 5:
            lr=torch.tensor(lr).cuda()
    except: assert lr == float(lr), \
        'lr can be a float number or a list of five float numbers'
    try:
        if len(eps_numdiff) == 5:
            eps_numdiff=torch.tensor(eps_numdiff).to(device)
    except: 
        assert eps_numdiff == float(eps_numdiff), \
            'eps_numdiff can be a float number or a list of five float numbers'
        eps_numdiff=torch.tensor([eps_numdiff]*5).to(device)

    # prepare tensors on device and float
    inp = in_image.to(device=device, dtype=torch.float32)
    tgt = target_image.to(device=device, dtype=torch.float32)

    if verbose:
        import time
        if fig_ax_bins is not None:
            from lognflow.plt_utils import plt_hist2, plt
            from lognflow.utils import is_notebook
            _is_notebook = is_notebook()
            if _is_notebook:
                if verbose:
                    print('Visualising the relative pixels values in a notebook')
                from IPython.display import display, clear_output

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

    params = ['trans_row', 'trans_col', 'scale_row', 'scale_col', 'rot_angle']
    # list-style for convenience
    pvals = torch.tensor([trans_row, trans_col, scale_row, scale_col, rot_angle], device=device, dtype=torch.float32)
    print('Columns of the printed outputs are: ')
    print('trans_row, trans_col, scale_row, scale_col, rot_angle')
    # small helper to compute loss for a given param vector
    def compute_loss_from_pvals(pvec):
        tr = float(pvec[0].item())
        tc = float(pvec[1].item())
        sr = float(pvec[2].item())
        sc = float(pvec[3].item())
        ra = float(pvec[4].item())
        warped = warp_image_with_params(inp, tr, tc, sr, sc, ra)
        # flatten to vectors
        
        # loss = ((warped - tgt)**2).max()
        loss = -loss_func(warped, tgt)
        
        # we want to maximize CC, so loss = -cc
        return loss, warped

    # main optimization loop
    if verbose:
        time_time_log_perv = time.time()
    base_loss_perv = 1.0e20
    for it in range(n_iters):
        base_loss, _ = compute_loss_from_pvals(pvals)

        # compute numerical gradient for each parameter using central difference
        grads = torch.zeros_like(pvals)
        for i in range(len(pvals)):
            if eps_numdiff[i] == 0:
                continue
            # perturb +/- eps
            p_plus = pvals.clone()
            p_minus = pvals.clone()
            p_plus[i] += eps_numdiff[i]
            p_minus[i] -= eps_numdiff[i]

            loss_plus, _ = compute_loss_from_pvals(p_plus)
            loss_minus, _ = compute_loss_from_pvals(p_minus)
            # central difference
            grad_i = (loss_plus - loss_minus) / (2.0 * eps_numdiff[i])
            grads[i] = grad_i

        # gradient descent step: p := p - lr * grad
        pvals = pvals - lr * grads

        if verbose:
            time_time_log = time.time()
            if (time_time_log > time_time_log_perv + log_period) or (it == n_iters - 1):
                time_time_log_perv = time_time_log
                if fig_ax_bins is not None:
                    _, final_warped = compute_loss_from_pvals(pvals)
                    if _is_notebook:
                        plt.close()
                        clear_output()
                        fig, ax, x_edges, y_edges = plt_hist2(np.column_stack((final_warped.ravel(), tgt.ravel())), bins = fig_ax_bins, return_bins = True)
                        ax.plot([y_edges.min(), x_edges.min()], [y_edges.max(), x_edges.max()], 'r-.')
                        display(fig)
                print(f"iter {it+1}/{n_iters} loss={base_loss.item():.6f} params={pvals.cpu().numpy()}")
        
        if torch.abs(base_loss - base_loss_perv) < 1e-12:
            break
        base_loss_perv = base_loss

    print(f"iter {it+1}/{n_iters} loss={base_loss.item():.6f} params={pvals.cpu().numpy()}")
    # final warp with optimized params
    _, final_warped = compute_loss_from_pvals(pvals)
    # unpack final params
    trans_row_f = float(pvals[0].item())
    trans_col_f = float(pvals[1].item())
    scale_row_f = float(pvals[2].item())
    scale_col_f = float(pvals[3].item())
    rot_angle_f = float(pvals[4].item())

    # return final parameters and warped image (same shape as input)
    return trans_row_f, trans_col_f, scale_row_f, scale_col_f, rot_angle_f, final_warped.to(device_orig)
    
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