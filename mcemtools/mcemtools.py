import numpy as np

def CoM4D(datacube : np.ndarray, 
          mask : np.ndarray = None, 
          normalize : bool=True):
    """ modified from py4DSTEM
    I wish they had written it like this
    Calculates two images - center of mass x and y - from a 4D-STEM datacube.

    The centers of mass are returned in units of pixels and in the Qx/Qy detector
    coordinate system.

    Args:
    ^^^^^^^
        :param datacube: np.ndarray 
            the 4D-STEM data of shape (R_Nx, R_Ny, Q_Nx, Q_Ny)
        :param mask: np.ndarray
            a 2D array, optionally, calculate the CoM only in the areas 
            where mask==True
        :param normalize: bool
            if true, subtract off the mean of the CoM images
    Returns:
    ^^^^^^^
        :returns: (2-tuple of 2d arrays), the center of mass coordinates, (x,y)
        :rtype: np.ndarray
    """
    R_Nx, R_Ny, Q_Nx, Q_Ny = datacube.shape

    if mask is None:
        mask = np.ones((Q_Nx, Q_Ny))
    else:
        assert len(mask.shape) == 2, 'mask should be 2d'
        assert mask.shape[0] == Q_Nx, 'mask should have same shape as patterns'
        assert mask.shape[1] == Q_Ny, 'mask should have same shape as patterns'
    
    qy, qx = np.meshgrid(np.arange(Q_Ny), np.arange(Q_Nx))
    qx_cube   = np.tile(qx,   (R_Nx, R_Ny, 1, 1))
    qy_cube   = np.tile(qy,   (R_Nx, R_Ny, 1, 1))
    mask_cube = np.tile(mask, (R_Nx, R_Ny, 1, 1))
    mass = (datacube * mask_cube).sum(3).sum(2).astype('float')
    CoMx = (datacube * qx_cube * mask_cube).sum(3).sum(2).astype('float')
    CoMy = (datacube * qy_cube * mask_cube).sum(3).sum(2).astype('float')
    CoMx[mass!=0] = CoMx[mass!=0] / mass[mass!=0]
    CoMy[mass!=0] = CoMy[mass!=0] / mass[mass!=0]

    if normalize:
        CoMx -= CoMx.mean()
        CoMy -= CoMy.mean()

    return CoMx, CoMy

def annular4D(data4D, radius = None, in_radius = None, centre = None):
    """ Annular virtual detector
            Given a 4D dataset, n_x x n_y x n_r x n_c.
            the output is the marginalized images over the n_x, n_y or n_r,n_c
        
        :param data4D:
            data in 4 dimension real_x x real_y x k_r x k_c
        :param radius:
            inside radius from zero to inf, default: 0
        :param in_radius:
            outside radius from zero to inf,default : inf
        :param centre:
            a tuple for the centre of the circular mask
            
    """
    mask_ = annular_mask(
        (data4D.shape[2], data4D.shape[3]), 
        center = centre, radius = radius, in_radius = in_radius)
    I4D_cpy = data4D.copy()
    I4D_cpy[:, :, mask_ == 0] = 0
    PACBED_ = I4D_cpy.sum(1).sum(0).squeeze()
    totI_ = I4D_cpy.sum(3).sum(2).squeeze()
    return totI_, PACBED_

def annular_CoM4D(data4D, in_radius = None, out_radius = None, centre = None):
    mask_ = annular_mask(
        (data4D.shape[2], data4D.shape[3]), 
        center = centre, radius = radius, in_radius = in_radius)
    I4D_CoMx, I4D_CoMy = CoM4D(data4D, mask_)
    return I4D_CoMx, I4D_CoMy

def mask2D_to_4D(mask2D, data4D_shape):
    n_x, n_y, n_r, n_c = data4D_shape
    assert len(mask.shape) == 2, 'mask should be 2d'
    assert mask2D.shape[0] == n_r, 'mask should have same shape as patterns'
    assert mask2D.shape[1] == n_c, 'mask should have same shape as patterns'
        
    _mask4D = np.array([np.array([mask2D.copy()])])
    _mask4D = np.tile(_mask4D, (n_x, n_y, 1, 1))
    return _mask4D

def normalize_4DSTEM(data4D, mask2D):
    data4D = data4D.copy()
    n_x, n_y, n_r, n_c = data4D.shape
    
    mask4D = mask2D_to_4D(mask2D, data4D.shape)
    mask4D[data4D == 0] = 0
    mask4D = mask4D.reshape(n_x, n_y, n_r * n_c)
    mask4D = mask4D.reshape(n_x * n_y, n_r * n_c)

    data4D = data4D.reshape(n_x, n_y, n_r * n_c)
    data4D = data4D.reshape(n_x * n_y, n_r * n_c)
    
    dset_mean = (data4D*mask4D).sum(1)
    dset_mask_sum_1 = mask4D.sum(1)
    dset_mean[dset_mask_sum_1 > 0] /= dset_mask_sum_1[dset_mask_sum_1>0]
    data4D -= np.tile(np.array([dset_mean]).swapaxes(0,1), (1, n_r * n_c))
    dset_std = (data4D ** 2).sum(1)
    dset_std[dset_mask_sum_1 > 0] /= dset_mask_sum_1[dset_mask_sum_1>0]
    dset_std = dset_std**0.5
    dset_std_tile = np.tile(np.array([dset_std]).swapaxes(0,1), (1, n_r * n_c))
    data4D[dset_std_tile>0] /= dset_std_tile[dset_std_tile>0]
    return data4D

def cross_corr(data4D_a, data4D_b, mask2D):
    assert data4D_a.shape == data4D_b.shape
    n_x, n_y, _, _ = data4D_a.shape
    data4D_a = normalize_4DSTEM(data4D_a.copy(), mask2D)
    data4D_b = normalize_4DSTEM(data4D_b.copy(), mask2D)
    
    mask4D = mask2D_to_4D(mask2D, data4D.shape)
    mask4D[data4D_a == 0] = 0
    mask4D[data4D_b == 0] = 0
    mask4D = mask4D.reshape(n_x, n_y, n_r * n_c)
    mask4D = mask4D.reshape(n_x * n_y, n_r * n_c)
    dset_mask_sum_1 = mask4D.sum(1)
    
    corr_mat  = (data4D_a * data4D_b).sum(1)
    corr_mat[dset_mask_sum_1>0] /= dset_mask_sum_1[dset_mask_sum_1>0]
    corr_mat = corr_mat.reshape(n_x, n_y)
    return corr_mat

def SymmSTEM(data4D, mask2D, nang = 180, mflag = 0):
    
    from skimage.transform import warp_polar
    
    realx,realy, _, _ = data4D.shape
    corr_ang_auto = np.zeros((realx,realy,nang))
    
    data4D = normalize_4DSTEM(data4D, mask2D)
       
    pBar = printprogress(realy * realx)
    for i in range(realx):
        for j in range(realy):
            vec_a = warp_polar(data4D[i,j] * mask2D).copy()
            rot = vec_a.copy()
            for _ang in range(nang):
                corr_ang_auto[i,j, _ang] = (rot* vec_a).mean()
                rot = np.roll(rot, 1, axis=0)
            pBar()
    return corr_ang_auto

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