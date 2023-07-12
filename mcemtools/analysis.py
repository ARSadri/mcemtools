import numpy as np

def pyMSSE(res, MSSE_LAMBDA = 3, k = 12) -> tuple:
    res_sq_sorted = np.sort(res**2)
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
        n_inliers = res.shape[0]
    return (est_std, n_inliers, adjacencies, inds)


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
            the 4D-STEM data of shape (n_x, n_y, n_r, n_c)
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
    n_x, n_y, n_r, n_c = datacube.shape

    if mask is None:
        mask = np.ones((n_r, n_c))
    else:
        assert len(mask.shape) == 2, 'mask should be 2d'
        assert mask.shape[0] == n_r, 'mask should have same shape as patterns'
        assert mask.shape[1] == n_c, 'mask should have same shape as patterns'
    
    qy, qx = np.meshgrid(np.arange(n_c), np.arange(n_r))
    qx_cube   = np.tile(qx,   (n_x, n_y, 1, 1))
    qy_cube   = np.tile(qy,   (n_x, n_y, 1, 1))
    mask_cube = np.tile(mask, (n_x, n_y, 1, 1))
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

def annular_CoM4D(data4D, radius = None, in_radius = None, centre = None):
    mask_ = annular_mask(
        (data4D.shape[2], data4D.shape[3]), 
        center = centre, radius = radius, in_radius = in_radius)
    I4D_CoMx, I4D_CoMy = CoM4D(data4D, mask_)
    return I4D_CoMx, I4D_CoMy

def cross_corr(data4D_a, data4D_b, mask2D):
    assert data4D_a.shape == data4D_b.shape
    n_x, n_y, n_r, n_c = data4D_a.shape

    mask4D = mask2D_to_4D(mask2D, data4D_a.shape)
    mask4D[data4D_a == 0] = 0
    mask4D[data4D_b == 0] = 0
    mask4D = mask4D.reshape(n_x, n_y, n_r * n_c)
    mask4D = mask4D.reshape(n_x * n_y, n_r * n_c)
    dset_mask_sum_1 = mask4D.sum(1)

    data4D_a = normalize4D(data4D_a.copy(), mask2D)
    data4D_b = normalize4D(data4D_b.copy(), mask2D)
    
    corr_mat  = (data4D_a * data4D_b).sum(1)
    corr_mat[dset_mask_sum_1>0] /= dset_mask_sum_1[dset_mask_sum_1>0]
    corr_mat = corr_mat.reshape(n_x, n_y)
    return corr_mat

def SymmSTEM(data4D, mask2D = None, nang = 180, mflag = 0,
             verbose = True):
    
    from skimage.transform import warp_polar
    
    n_x, n_y, n_r, n_c = data4D.shape
    
    if mask2D is None:
        mask2D = np.ones((n_r, n_c), dtype='int8')
    
    corr_ang_auto = np.zeros((n_x,n_y,nang))
    
    data4D = normalize4D(data4D, mask2D)
    
    if(verbose):
        pBar = printprogress(n_x * n_y)
    for i in range(n_x):
        for j in range(n_y):
            vec_a = warp_polar(data4D[i,j] * mask2D).copy()
            rot = vec_a.copy()
            for _ang in range(nang):
                corr_ang_auto[i,j, _ang] = (rot* vec_a).mean()
                rot = np.roll(rot, 1, axis=0)
            if(verbose):
                pBar()
    return corr_ang_auto