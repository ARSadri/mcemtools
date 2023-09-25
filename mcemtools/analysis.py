import numpy as np
from .masking import annular_mask, mask2D_to_4D, image_by_windows
from lognflow import printprogress, lognflow
from skimage.transform import warp_polar

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
                data4D[x_cnt, y_cnt][weights4D[x_cnt, y_cnt] > 0] = cbed
            else:
                data4D[x_cnt, y_cnt] = cbed
    return data4D

def calc_ccorr(inputs_to_iter_sliced: tuple, inputs_to_share: tuple):
    CBED = inputs_to_iter_sliced[0]
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

def SymmSTEM(data4D, mask2D = None, nang = 180, mflag = False, 
             verbose = True, use_multiprocessing = True):
    
    n_x, n_y, n_r, n_c = data4D.shape
    
    if mask2D is not None:
        assert mask2D.shape == (n_r, n_c),\
            'mask2D should have the same shape as'\
            ' (data4D.shape[2], data4D.shape[3])'
        mask_ang = warp_polar(mask2D.copy())
    else:
        mask_ang = warp_polar(np.ones((n_r, n_c)))
        
    inputs_to_share = (mask_ang, nang, mflag, )
    if use_multiprocessing:
        inputs_to_iter = (data4D.reshape((n_x*n_y, n_r, n_c)), )
        from lognflow import multiprocessor
        corr_ang_auto = multiprocessor(
            calc_ccorr, 
            inputs_to_iter = inputs_to_iter,
            inputs_to_share = inputs_to_share,
            outputs = np.zeros((n_x*n_y, nang)),
            verbose = verbose)
        corr_ang_auto = corr_ang_auto.reshape(
            (n_x, n_y, corr_ang_auto.shape[1]))
    
    else:
        corr_ang_auto = np.zeros((n_x,n_y,nang))
        if(verbose):
            pBar = printprogress(
                n_x * n_y, title = f'Symmetry STEM for {n_x * n_y} patterns')
        for i in range(n_x):
            for j in range(n_y):
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
        I4D_cpy *= weight4D
    PACBED = I4D_cpy.sum(1).sum(0).squeeze()
    totI = I4D_cpy.sum(3).sum(2).squeeze()
    return totI, PACBED

def conv_4D(data4D, winXY, conv_function = sum_4D, skip = (1, 1)):
    """
        :param conv_function:
            a function that returns a tuple, we will use the second element:
            _, stat = conv_function(data4D)
            This function should return a 2D array at second position in the 
            tuple. For example sum_4D returns sum((0,1)) of the 4D array. 
    """
    imgbywin = image_by_windows(
                data4D.shape, winXY, skip = skip)
    views = imgbywin.image2views(data4D)
    for cnt, view in enumerate(views):
        _, stat = conv_function(view)
        views[cnt] = np.tile(np.array([np.array([stat])]), 
                             (view.shape[0], view.shape[1], 1, 1)).copy()
    data4D = imgbywin.views2image(views)
    return data4D

def bin_4D(data4D, 
          n_pos_in_bin: int = 1, n_pix_in_bin: int = 1,
          method_pos: str = 'skip', method_pix: str = 'linear',
          conv_function = sum_4D, skip = (1, 1)):
    data4D = data4D.copy()
    if(n_pos_in_bin > 1):
        if(method_pos == 'skip'):
            data4D = data4D[::n_pos_in_bin, ::n_pos_in_bin]
        if(method_pos == 'linear'):
            data4D = conv_4D(
                data4D, (n_pos_in_bin, n_pos_in_bin), conv_function,
                skip = skip)
            data4D = data4D[::n_pos_in_bin, ::n_pos_in_bin, :, :]
    if(n_pix_in_bin > 1):
        if(method_pix == 'skip'):
            data4D = data4D[:, :, ::n_pix_in_bin, ::n_pix_in_bin]
        if(method_pix == 'linear'):
            data4D = data4D.swapaxes(
                1,2).swapaxes(0,1).swapaxes(2,3).swapaxes(1,2)
            data4D = conv_4D(
                data4D, (n_pix_in_bin, n_pix_in_bin), conv_function,
                skip = skip)
            data4D = data4D.swapaxes(
                1,2).swapaxes(0,1).swapaxes(2,3).swapaxes(1,2)
            data4D = data4D[:, :, ::n_pix_in_bin, ::n_pix_in_bin]
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
    
    clm_grid, row_grid = np.meshgrid(np.arange(n_c), np.arange(n_r))
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
    corr_mat  = (data4D_a * data4D_b).sum(3).sum(2)
    
    if mask4D is not None:
        mask_STEM = mask4D.sum(3).sum(2)
        corr_mat[mask_STEM>0] /= mask_STEM[mask_STEM>0]
    return corr_mat

def locate_atoms(data4D, mask4D = None, min_distance = 3, filter_size = 3,
                 reject_too_close = False):
    
    totI, PACBED = sum_4D(data4D, mask4D)
    
    from skimage.feature import peak_local_max
    import scipy.ndimage
    _, _, n_r, n_c = data4D.shape
    image_max = scipy.ndimage.maximum_filter(
        -totI, size=filter_size, mode='constant')
    coordinates = peak_local_max(-totI, min_distance=min_distance)
    if(reject_too_close):
        from RobustGaussianFittingLibrary import fitValue
        dist2 = scipy.spatial.distance.cdist(coordinates, coordinates)
        dist2 = dist2 + np.diag(np.inf + np.zeros(coordinates.shape[0]))
        mP = fitValue(dist2.min(1))
        dist2_threshold = mP[0] - mP[1]
        dist2_threshold = np.minimum(dist2_threshold, dist2.min(1).mean())
        
        inds = np.where(   (dist2_threshold < coordinates[:, 0])
                         & (coordinates[:, 0] < n_r - dist2_threshold)
                         & (dist2_threshold < coordinates[:, 1])
                         & (coordinates[:, 1] < n_c - dist2_threshold)  )[0]
        
        coordinates = coordinates[inds]
    return coordinates

def stem_image_nyquist_interpolation(
        StemImage,xlen,ylen,alpha,Knought,npixout,npiyout):
    """
        StemImageNyquistInterpolation Nyquist interpolates a STEM image 
        using Fourier methods. STEMImage has real space dimensions ylen 
        and xlen in Angstrom - Note use of spatial coordinates
        
        alpha = probe-forming aperture semiangle in mrad 
        Knought = vacuum wavevector (in inverse Angstrom)
        
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