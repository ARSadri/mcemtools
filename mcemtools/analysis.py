import numpy as np
from .masking import annular_mask, mask2D_to_4D, image_by_windows
from lognflow import printprogress
from skimage.transform import warp_polar

def normalize_4D(data4D, weights4D = None):
    """
        Note::
            make sure you have set weights4D[data4D == 0] = 0 when dealing with
            Poisson.
    """
    data4D = data4D.copy()
    n_x, n_y, n_r, n_c = data4D.shape
    data4D = data4D.reshape(n_x, n_y, n_r * n_c)
    data4D = data4D.reshape(n_x * n_y, n_r * n_c)

    if weights4D is not None:
        weights4D = weights4D.copy()
        weights4D = weights4D.reshape(n_x, n_y, n_r * n_c)
        weights4D = weights4D.reshape(n_x * n_y, n_r * n_c)
        dset_mean = (data4D*weights4D).sum(1)
        weights_sum = weights4D.sum(1)
        dset_mean[weights_sum > 0] /= weights_sum[weights_sum>0]
        data4D -= np.tile(np.array([dset_mean]).swapaxes(0,1), (1, n_r * n_c))
        dset_std = (data4D ** 2).sum(1)
        dset_std[weights_sum > 0] /= weights_sum[weights_sum>0]
        dset_std[weights_sum == 0] = 0
    else:
        dset_mean = data4D.mean(1)
        data4D -= np.tile(np.array([dset_mean]).swapaxes(0,1), (1, n_r * n_c))
        dset_std = (data4D ** 2).mean(1)
    dset_std = dset_std**0.5
    dset_std_tile = np.tile(np.array([dset_std]).swapaxes(0,1), (1, n_r * n_c))
    data4D[dset_std_tile>0] /= dset_std_tile[dset_std_tile>0]
    
    data4D = data4D.reshape(n_x, n_y, n_r* n_c)
    data4D = data4D.reshape(n_x, n_y, n_r, n_c)
    
    return data4D

def locate_atoms(data4D, min_distance = 3, filter_size = 3,
                 reject_too_close = False):
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

def pyMSSE(fitting_errors, MSSE_LAMBDA = 3, k = 12) -> tuple:
    res_sq = fitting_errors**2
    res_sq_sortinds = np.argsort(res_sq)
    res_sq_sorted = res_sq[res_sq_sortinds]
    res_sq_cumsum = np.cumsum(res_sq_sorted)
    cumsums = res_sq_cumsum[:-1]/np.arange(1, res_sq_cumsum.shape[0])
    cumsums[cumsums==0] = cumsums[cumsums>0].min()
    adjacencies = (res_sq_sorted[1:]/ cumsums)**0.5
    adjacencies[:k] = 0
    inds = np.where(adjacencies > MSSE_LAMBDA)[0]
    est_done = False
    if(inds.shape[0]>0):
        if inds[0] > 0 :
            n_inliers = inds[0] - 1
            est_std = cumsums[n_inliers] ** 0.5
            est_done = True
    if (not est_done):
        est_std = cumsums[-1] ** 0.5
        n_inliers = fitting_errors.shape[0]
    return (est_std, n_inliers, adjacencies, res_sq_sortinds)

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

def conv_4D(data4D, winXY, conv_function = sum_4D):
    """
        :param conv_function:
            a function that returns a tuple, we will use the second element:
            _, stat = conv_function(data4D)
            This function should return a 2D array at second position in the 
            tuple. For example sum_4D returns sum((0,1)) of the 4D array. 
    """
    imgbywin = image_by_windows(
                data4D.shape, winXY)
    views = imgbywin.image2views(data4D)
    _views = views.copy()
    for cnt, view in enumerate(views):
        _, stat = conv_function(view)
        stat = np.tile(np.array([np.array([stat])]), 
                       (view.shape[0], view.shape[1], 1, 1))
        _views[cnt] = stat.copy()
    data4D = imgbywin.views2image(_views)
    return data4D

def bin_4D(data4D, 
          n_pos_in_bin: int = 1, n_pix_in_bin: int = 1,
          method_pos: str = 'skip', method_pix: str = 'linear',
          conv_function = sum_4D):
    data4D = data4D.copy()
    if(n_pos_in_bin > 1):
        if(method_pos == 'skip'):
            data4D = data4D[::n_pos_in_bin, ::n_pos_in_bin]
        if(method_pos == 'linear'):
            data4D = data4D.swapaxes(
                1,2).swapaxes(0,1).swapaxes(2,3).swapaxes(1,2)
            data4D = conv_4D(data4D, (n_pos_in_bin, n_pos_in_bin),
                             conv_function)
            data4D = data4D[:, :, ::n_pos_in_bin, ::n_pos_in_bin]
            data4D = data4D.swapaxes(
                1,2).swapaxes(0,1).swapaxes(2,3).swapaxes(1,2)
    if(n_pix_in_bin > 1):
        if(method_pix == 'skip'):
            data4D = data4D[:, :, ::n_pix_in_bin, ::n_pix_in_bin]
        if(method_pix == 'linear'):
            data4D = conv_4D(data4D, (n_pix_in_bin, n_pix_in_bin),
                             conv_function)
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

def SymmSTEM(data4D, mask4D = None, nang = 180, mflag = 0, verbose = True):
    
    n_x, n_y, n_r, n_c = data4D.shape
    
    if mask4D is not None:
        assert mask4D.shape == data4D.shape,\
            'mask4D should have the same shape as data4D'
    
    corr_ang_auto = np.zeros((n_x,n_y,nang))
    
    data4D = normalize_4D(data4D, mask4D)
    n_unmasked = 1
    
    if(verbose):
        pBar = printprogress(
            n_x * n_y, title = f'Symmetry STEM for {n_x * n_y} patterns')
    for i in range(n_x):
        for j in range(n_y):
            if mask4D is not None:
                vec_a = warp_polar(data4D[i, j] * mask4D[i, j]).copy()
                n_unmasked = mask4D[i, j].sum()
            else:
                vec_a = warp_polar(data4D[i, j].copy())
            rot = vec_a.copy()
            for _ang in range(nang):
                corr_ang_auto[i,j, _ang] = (rot * vec_a).sum() / n_unmasked
                rot = np.roll(rot, 1, axis=0)
            if(verbose):
                pBar()
    corr_ang_auto[corr_ang_auto > 1] = 1
    corr_ang_auto[corr_ang_auto < -1] = -1
    return corr_ang_auto