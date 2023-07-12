import numpy as np
from .masking import annular_mask, mask2D_to_4D
from lognflow import printprogress

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

def fig2data(fig):
    """ from https://www.icare.univ-lille.fr/how-to-
                    convert-a-matplotlib-figure-to-a-numpy-array-or-a-pil-image/
    """
    fig.canvas.draw()
    w,h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.ubyte)
    buf.shape = (w, h, 4)
    buf = np.roll (buf, 3, axis = 2)
    return buf

def numbers_as_images(dataset_shape, fontsize, verbose = False):
    """ Numbers4D
    This scripts generates a 4D dataset of images with shape
    (n_x, n_y, n_r, n_c) where in each image the value "x, y" is written as a text
    that fills the image. As such later when working with such a dataset you can
    look at the image and know which index it had before you using it.
    
    Follow this receip to make good images:
    
    1- set n_x, n_y to 10, Set the desired n_r and width. 
    2- try fontsize that is the largest
    3- Increase n_x and n_y to desired siez.
    
    You can provide a logs_root, log_dir or simply select a directory to save the
    output 4D array.
    
    """
    n_x, n_y, n_r, n_c = dataset_shape
    dataset = np.zeros((n_x, n_y, n_r, n_c))    
    txt_width = int(np.log(np.maximum(n_x, n_y))
                    /np.log(np.maximum(n_x, n_y))) + 1
    number_text_base = '{ind_x:0{width}}, {ind_y:0{width}}'
    if(verbose):
        pBar = printprogress(n_x * n_y)
    for ind_x in range(n_x):
        for ind_y in range(n_y):
            mat = np.ones((n_r, n_c))
            number_text = number_text_base.format(ind_x = ind_x, 
                                                  ind_y = ind_y,
                                                  width = txt_width)
            fig = plt.figure(figsize = (1, 1))
            ax = fig.add_subplot(111)
            ax.imshow(mat, cmap = 'gray', vmin = 0, vmax = 1)
            ax.text(mat.shape[0]//2 - fontsize, mat.shape[1]//2 ,
                    number_text, fontsize = fontsize)
            ax.axis('off')
            buf = fig2data (fig)
            plt.close()
            buf2 = buf[::buf.shape[0]//n_r, ::buf.shape[1]//n_c, :3].mean(2)
            buf2 = buf2[:n_r, :n_c]
            dataset[ind_x, ind_y] = buf2.copy()
            if(verbose):
                pBar()
    return dataset