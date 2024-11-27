import numpy as np
from .masking import annular_mask, mask2D_to_4D, image_by_windows
from lognflow import printprogress, lognflow
from skimage.transform import warp_polar
import scipy
from itertools import product
import torch
import mcemtools

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

def spatial_incoherence_4D(data4d, spatInc_params, use_fft = False):
    """
    Apply spatial incoherence using a Gaussian filter on 4D-STEM data.
    
    Parameters:
    -----------
    data4d : ndarray
        4D dataset (n_x, n_y, n_r, n_c) to apply the filtering.
    spatInc_params : dict
        Dictionary of parameters for the Gaussian filter. Should include:
        - 'filter_size': int
        - 's1': float, optional
        - 's2': float, optional
        - 'angle': float, optional
    use_fft: bool, default: False
        if True, we turn both data and filter to fourier space, multiply and inverse.
        Remember that if data is not tiled, you may have artifacts.
    Returns:
    --------
    filtered_data4d : ndarray
        Filtered 4D-STEM data with applied spatial incoherence.
    """
    gaussian_filter = Gaussian_2dkernel(**spatInc_params).astype('float32')
    
    n_x, n_y, n_r, n_c = data4d.shape
    
    if use_fft:
        data4d_ft = scipy.fft.fftn(data4d, axes=(0, 1))
        filter_ft = scipy.fft.fftn(gaussian_filter, s=(n_x, n_y))
        filter_ft_tiled = np.tile(
            filter_ft[:, :, np.newaxis, np.newaxis], (1, 1, n_r, n_c))
        result_ft = data4d_ft * filter_ft_tiled
        filtered_data4d = scipy.fft.ifftn(result_ft, axes=(0, 1)).real
    else:

        gaussian_filter = np.expand_dims(gaussian_filter, -1)
        gaussian_filter = np.expand_dims(gaussian_filter, -1)
        gaussian_filter = np.tile(gaussian_filter, (1, 1, n_r, n_c))
        
        imgbywin = mcemtools.image_by_windows((n_x, n_y), gaussian_filter.shape,
                                              skip = (1, 1), method = 'fixed')
        filtered_data4d = np.zeros(
            (imgbywin.grid_shape[0],imgbywin.grid_shape[1], n_r, n_c),
            dtype = data4d.dtype)
        for grc in imgbywin.grid:
            filtered_data4d[grc[0], grc[1]] = (data4d[
                grc[0]:grc[0] + imgbywin.win_shape[0], 
                grc[1]:grc[1] + imgbywin.win_shape[1]] * gaussian_filter).sum((0, 1))

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

def calc_ccorr(CBED, inputs_to_share: tuple):
    mask_ang, nang, mflag = inputs_to_share
    
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

def calc_symm(CBED, inputs_to_share: tuple):
    mask_ang, nang, mflag = inputs_to_share
    
    polar = warp_polar(CBED)
    kvec = np.tile(np.array([np.arange(polar.shape[0])]).swapaxes(0, 1),
                (1, nang))/(nang/2/np.pi)
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
        polar_fft_ifft = \
            np.real(np.fft.ifft((np.fft.fft(polar,nang,2))**2,nang,2))
    else:          #rotational symmetries
        polar_fft_ifft = \
            np.real(np.fft.ifft(np.abs(np.fft.fft(polar,nang,1))**2,nang,1))
      
    """
        multiply array to account for Jacobian polar r weighting (here kvec). 
        Integrate over radius in the diffraction patter - one could also
        mask the pattern beforehand, as in ACY Liu's correlogram approach.
    """
    corr = np.sum(polar_fft_ifft*kvec)
    
    """
        notice the deliberate omission of fftshift above.     
        factors of nang and 2*pi are for numerical comparison to the Riemann
        sum integrals in the Cartesian case.
        normalise with respect to no symmetry operation.  For accurate
        normalisation, include otherwise redundant polar coordinate 
        conversion and subsequent squaring.
    """
    corr = corr/np.sum(np.sum((polar**2)*kvec[:, :polar.shape[1]]))
    
    return corr


def SymmSTEM(data4D, mask2D = None, nang = 180, mflag = False, 
             verbose = True, use_multiprocessing = False,
             use_autoconvolutiuon = False):
    assert not use_autoconvolutiuon
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

def CoM_detector(det_geo):
    cent_x, cent_y = scipy.ndimage.center_of_mass(det_geo > 0)
    mask_coms = []
    for cnt, label in enumerate(np.unique(det_geo.ravel())[1:]):
        mask_com_x, mask_com_y = scipy.ndimage.center_of_mass(det_geo == label)
        mask_com_x -= cent_x
        mask_com_y -= cent_y
        mask_coms.append([mask_com_x, mask_com_y])
    return np.array(mask_coms)

def CoM_channel_torch(data_per_ch, mask_coms):
    com_x_ch = []
    com_y_ch = []
    for cnt, mask_com in enumerate(mask_coms):
        com_x_ch.append(data_per_ch[:, cnt] * mask_com[0])
        com_y_ch.append(data_per_ch[:, cnt] * mask_com[1])
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
        mass = (data4D * mask4D).sum(3).sum(2).astype('float')
        CoMx = (data4D * row_grid_cube * mask4D).sum(3).sum(2).astype('float')
        CoMy = (data4D * clm_grid_cube * mask4D).sum(3).sum(2).astype('float')
    else:
        mass = data4D.sum(3).sum(2).astype('float')
        CoMx = (data4D * row_grid_cube).sum(3).sum(2).astype('float')
        CoMy = (data4D * clm_grid_cube).sum(3).sum(2).astype('float')
        
    CoMx[mass!=0] = CoMx[mass!=0] / mass[mass!=0]
    CoMy[mass!=0] = CoMy[mass!=0] / mass[mass!=0]

    if normalize:
        CoMx -= CoMx.mean()
        CoMy -= CoMy.mean()

    return CoMx, CoMy

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

def locate_atoms(stem, min_distance = 3, min_distance_init = 1,
                 maxfilter_size = 3, reject_too_close = False,
                 rgflib_fitBackground_kwargs = None, logger = None):
    
    n_x, n_y = stem.shape
    
    nSTEM = stem.max() -stem.copy()
    
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
    
    inds = []
    if(reject_too_close):
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
        StemImage,xlen,ylen,alpha,Knought,npixout,npiyout):
    """
        StemImageNyquistInterpolation Nyquist interpolates a STEM image 
        using Fourier methods. STEMImage has real space dimensions ylen 
        and xlen in Angstrom - Note use of spatial coordinates
        
        alpha = probe-forming aperture semiangle in mrad 
        Knought = vacuum wavevector (in inverse Angstrom)
        
        \\lambda_0 = \frac{h}{\sqrt{2 m_e e V \left( 1 + \frac{e V}{2 m_e c^2} \right)}}
        K_0 = \frac{2pi}{\\lambda_0}
        
        npixout,npiyout give number of pixels in x and y for output
    """
   
    [npiy, npix] = np.shape(StemImage)   # Note use of spatial coordinates
    qalpha = Knought * alpha * 1.0e-3    # Probe cutoff (in inverse Angstrom)
    qband = 2.0 * qalpha    # STEM image bandwith limit (in inverse Angstrom)
    qnyq = 2.0 * qband    # Nyquist spatial frequency (in inverse Angstrom)
    # kxscale = 1/xlen
    # kyscale = 1/ylen
    # xscale = xlen/npix
    # yscale = ylen/npiy

    npixmin = np.ceil(xlen * qnyq)
    npiymin = np.ceil(ylen * qnyq)
   
    if npix < npixmin or npiy < npiymin:
        print('Input STEM image is insufficiently '
              'sampled for Nyquist interpolation')
       
    # Ex-D'Alfonso implementation
    ctemp2 = np.vectorize(complex)(StemImage)
    ctemp2 = np.fft.fft2(ctemp2)
    # ctemp2 = ctemp2 * sqrt(npiy*npix)
       
    Npos_y = round(np.floor(npiy/2))
    Npos_x = round(np.floor(npix/2))
     
    shifty = npiy - Npos_y - 1
    shiftx = npix - Npos_x - 1

    ctemp2 = np.roll(np.roll(ctemp2,int(shifty),axis=0), int(shiftx), axis=1)

    ctemp = np.vectorize(complex)(np.zeros((npiyout,npixout)))
    ctemp[0:npiy,0:npix] = ctemp2

    ctemp = np.roll(np.roll(ctemp,-int(shifty),axis=0),-int(shiftx),axis=1)
   
    StemImageInterpolated = np.real(np.fft.ifft2(ctemp))
   
    StemImageInterpolated = StemImageInterpolated * (npixout * npiyout) / (npix * npiy)
   
    return StemImageInterpolated

def force_stem_4d(a4d, b4d):
    """ force stem
        force the stem image of 4d dataset a to be the stem image of 4d dataset b.
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

def compute_pixel_histograms(images, bins=np.arange(10)):
    """
    Compute histograms for each pixel position across a stack of images.
    
    Parameters:
    - images: np.ndarray of shape (N, M, M), where N is the number of images and M x M is the image size.
    - bins: np.ndarray, the bin edges to use for histogram calculation.
    
    Returns:
    - histograms: np.ndarray of shape (len(bins)-1, M, M), where each slice corresponds to a bin count per pixel.
    """
    num_bins = len(bins) - 1
    N, M, _ = images.shape
    
    histograms = np.zeros((num_bins, M, M), dtype=int)
    
    binned = np.digitize(images, bins=bins) - 1

    for b in range(num_bins):
        histograms[b] = np.sum(binned == b, axis=0)

    return histograms