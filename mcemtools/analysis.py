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
