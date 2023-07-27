import numpy as np
from .masking import annular_mask, mask2D_to_4D
from .transforms import normalize_4D
from lognflow import printprogress
from skimage.transform import warp_polar

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
    
def sum_4D(data4D, mask4D = None):
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
    
    I4D_cpy = data4D.copy()
    I4D_cpy[mask4D == 0] = 0
    PACBED = I4D_cpy.sum(1).sum(0).squeeze()
    totI = I4D_cpy.sum(3).sum(2).squeeze()
    return totI, PACBED

def centre_of_mass_4D(data4D, mask4D = None, normalize = True):
    """ modified from py4DSTEM
    I wish they had written it as follows
    Calculates two images - centre of mass x and y - from a 4D data4D.

    Args:
    ^^^^^^^
        :param data4D: np.ndarray 
            the 4D-STEM data of shape (n_x, n_y, n_r, n_c)
        :param mask4D: np.ndarray
            a 4D array, optionally, calculate the CoM only in the areas 
            where mask==True
        :param normalize: bool
            if true, subtract off the mean of the CoM images
    Returns:
    ^^^^^^^
        :returns: (2-tuple of 2d arrays), the centre of mass coordinates, (x,y)
        :rtype: np.ndarray
    """
    n_x, n_y, n_r, n_c = data4D.shape

    if mask4D is not None:
        assert mask4D.shape == data4D.shape,\
            'mask4D should have the same shape as data4D'
    
    clm_grid, row_grid = np.meshgrid(np.arange(n_c), np.arange(n_r))
    row_grid_cube   = np.tile(row_grid,   (n_x, n_y, 1, 1))
    clm_grid_cube   = np.tile(clm_grid,   (n_x, n_y, 1, 1))
    
    if mask4D is not None:
        mass = (data4D * mask_cube).sum(3).sum(2).astype('float')
        CoMx = (data4D * row_grid_cube * mask_cube).sum(3).sum(2).astype('float')
        CoMy = (data4D * clm_grid_cube * mask_cube).sum(3).sum(2).astype('float')
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
    return corr_ang_auto