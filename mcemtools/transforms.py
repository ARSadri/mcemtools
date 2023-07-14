import numpy as np
import scipy

def bin_4D(data4D, 
          n_pos_in_bin: int = 1, n_pix_in_bin: int = 1,
          method_pos: str = 'skip', method_pix: str = 'linear'):
    
    data4D = data4D.copy()
    if(n_pos_in_bin > 1):
        if(method_pos == 'skip'):
            data4D = data4D[::n_pos_in_bin, ::n_pos_in_bin]
        if(method_pos == 'linear'):
            data4D = data4D.swapaxes(
                1,2).swapaxes(0,1).swapaxes(2,3).swapaxes(1,2)
            kern = np.full((n_pos_in_bin, n_pos_in_bin), fill_value = 1)
            for rcnt in range(data4D.shape[0]):
                for ccnt in range(data4D.shape[1]):
                    data4D[rcnt, ccnt] = \
                        scipy.ndimage.convolve(data4D[rcnt, ccnt], kern)
            data4D = data4D[:, :, ::n_pos_in_bin, ::n_pos_in_bin]
            data4D = data4D.swapaxes(
                1,2).swapaxes(0,1).swapaxes(2,3).swapaxes(1,2)
    if(n_pix_in_bin > 1):
        if(method_pix == 'skip'):
            data4D = data4D[:, :, ::n_pix_in_bin, ::n_pix_in_bin]
        if(method_pix == 'linear'):
            kern = np.full((n_pix_in_bin, n_pix_in_bin), fill_value = 1)
            for xcnt in range(data4D.shape[0]):
                for ycnt in range(data4D.shape[1]):
                    data4D[xcnt, ycnt] = \
                        scipy.ndimage.convolve(data4D[xcnt, ycnt], kern)
            data4D = data4D[:, :, ::n_pix_in_bin, ::n_pix_in_bin]
    return data4D

def get_polar_coords(image_shape, centre, polar_shape):
    n_angles, n_rads = polar_shape
    n_rows, n_clms = image_shape
    if (centre is None):
        centre = (n_rows//2, n_clms//2)
    cc, rr = np.meshgrid(np.arange(n_clms), np.arange(n_rows))

    angles = np.arctan2((rr - centre[0]), (cc - centre[1])) 
    angles_min_dist = np.diff(np.sort(angles.ravel()))
    angles_min_dist = angles_min_dist[angles_min_dist>0].min()

    anglesq = np.arctan2((rr - centre[0]), -(cc - centre[1])) 
    anglesq_min_dist = np.diff(np.sort(anglesq.ravel()))
    anglesq_min_dist = anglesq_min_dist[anglesq_min_dist>0].min()
    
    rads   = ((rr - centre[0])**2 + (cc - centre[1])**2)**0.5
    rads_min_dist = np.diff(np.sort(rads.ravel()))
    rads_min_dist = rads_min_dist[rads_min_dist>0].min()
    
    angles_pix_in_polar = angles - angles.min()
    angles_pix_in_polar = (angles_pix_in_polar / angles_pix_in_polar.max() 
                           * n_angles).astype('int')
    anglesq_pix_in_polar = anglesq - anglesq.min()
    anglesq_pix_in_polar = (anglesq_pix_in_polar / anglesq_pix_in_polar.max() 
                           * n_angles).astype('int')
                                                  
    rads_pix_in_polar = (rads / rads.max() * n_rads).astype('int')
    
    angles_pix_in_polar = angles_pix_in_polar.ravel()
    anglesq_pix_in_polar = anglesq_pix_in_polar.ravel()
    rads_pix_in_polar = rads_pix_in_polar.ravel()
    rr = rr.ravel()
    cc = cc.ravel()
    return (angles_pix_in_polar, anglesq_pix_in_polar, 
            rads_pix_in_polar, rr, cc)

def polar2image(data, image_shape, dataq = None, centre = None,
                get_polar_coords_output = None):
    """ 
        :param dataq:
            To those who ignore loss of information at the angle 0, you have to
            make two polar images out of a cartesian image, one beginning from 
            angle 0 and the other from another angle far from zero, better be 
            180. Then you have to process both images, and then give it back to
            this function to make the original cartesian image. 
            Use dataq as the output of image2polar then give its processed 
            version to this function as dataq...., now, see? you hadn't paid
            attention...am I right? It is very importnt, isn't it? ... 
            Yes! it is importnat....Hey!, I said it is important.
    """
    n_rows, n_clms = image_shape
    if dataq is None:
        dataq = data
    else:
        assert dataq.shape == data.shape,\
            'dataq should have the same type, shape and dtype as data'

    data_shape = data.shape
    data_shape_rest = data_shape[2:]

    if get_polar_coords_output is None:
        n_angles = data_shape[0] - 1
        n_rads = data_shape[1] - 1
        if (centre is None):
            centre = (n_rows//2, n_clms//2)
        angles_pix_in_polar, anglesq_pix_in_polar, rads_pix_in_polar, rr, cc = \
            get_polar_coords(image_shape, centre, (n_angles, n_rads))
    else:
        angles_pix_in_polar, anglesq_pix_in_polar, rads_pix_in_polar, rr, cc = \
            get_polar_coords_output
            
    image = np.zeros(
        (n_rows, n_clms) + data_shape_rest, dtype = data.dtype)
    mask = image.astype('int').copy()
    for a, aq, b, c, d in zip(angles_pix_in_polar.ravel(),
                              anglesq_pix_in_polar.ravel(),
                              rads_pix_in_polar.ravel(),
                              rr.ravel(), 
                              cc.ravel()):
        image[c,d] += data[a,b]
        mask[c,d] += 1
        image[c,d] += dataq[aq,b]
        mask[c,d] += 1
    image[mask>0] /= mask[mask>0]
    
    return (image, mask)

def image2polar(data,
               n_angles = 360,
               n_rads = None,
               centre = None,
               get_polar_coords_output = None):
    """ image to polar transform
    
        :param get_polar_coords_output:
            there is a function up there called get_polar_coords. It produces
            the polar coordinates. One can call that function first to
            generate coordinates, then pass the coordinates to these
            two funcitons (image2polar and polar2image) any number of times.
            If user does not call this function abefore hand and does not 
            provide it to image2polar or polar2image, the functions will 
            call it. get_polar_coords is a fast function... No OOP here.
    """

    data_shape = data.shape
    n_rows = data_shape[0]
    n_clms = data_shape[1]
    data_shape_rest = data_shape[2:]
    
    if get_polar_coords_output is None:
        if(n_rads is None):
            n_rads = int(np.ceil(((n_rows/2)**2 + (n_clms/2)**2)**0.5))
        if (centre is None):
            centre = (n_rows//2, n_clms//2)
        angles_pix_in_polar, anglesq_pix_in_polar, rads_pix_in_polar, rr, cc = \
            get_polar_coords((n_rows, n_clms), centre, (n_angles, n_rads))
    else:
        angles_pix_in_polar, anglesq_pix_in_polar, rads_pix_in_polar, rr, cc = \
            get_polar_coords_output
    
    polar_image = np.zeros(
        (angles_pix_in_polar.max() + 1, 
         rads_pix_in_polar.max() + 1) + data_shape_rest, dtype = data.dtype)
    polar_imageq = polar_image.copy()
    polar_mask = polar_image.astype('int').copy()
    polar_maskq = polar_mask.copy()
    for a, aq, b, c,d in zip(angles_pix_in_polar,
                             anglesq_pix_in_polar,
                             rads_pix_in_polar,
                             rr, 
                             cc):
        polar_image[a,b] += data[c,d]
        polar_imageq[aq,b] += data[c,d]
        polar_mask[a,b] += 1
        polar_maskq[aq,b] += 1
    polar_image[polar_mask>0] /= polar_mask[polar_mask>0]
    polar_imageq[polar_maskq>0] /= polar_maskq[polar_maskq>0]
    
    return (polar_image, polar_imageq, polar_mask, polar_maskq)

def normalize_4D(data4D, mask4D = None):
    """
        Note::
            make sure you have set mask4D[data4D == 0] = 0 when dealing with
            Poisson.
    """
    data4D = data4D.copy()
    if mask4D is None:
        mask4D = np.ones(data4D.shape)
    else:
        mask4D = mask4D.copy()
    n_x, n_y, n_r, n_c = data4D.shape
    if mask4D is not None:
        mask4D = mask4D.reshape(n_x, n_y, n_r * n_c)
        mask4D = mask4D.reshape(n_x * n_y, n_r * n_c)

    data4D = data4D.reshape(n_x, n_y, n_r * n_c)
    data4D = data4D.reshape(n_x * n_y, n_r * n_c)
    
    if mask4D is not None:
        dset_mean = (data4D*mask4D).sum(1)
        dset_mask_sum_1 = mask4D.sum(1)
        dset_mean[dset_mask_sum_1 > 0] /= dset_mask_sum_1[dset_mask_sum_1>0]
        data4D -= np.tile(np.array([dset_mean]).swapaxes(0,1), (1, n_r * n_c))
        dset_std = (data4D ** 2).sum(1)
        dset_std[dset_mask_sum_1 > 0] /= dset_mask_sum_1[dset_mask_sum_1>0]
    else:
        dset_mean = (data4D).mean(1)
        data4D -= np.tile(np.array([dset_mean]).swapaxes(0,1), (1, n_r * n_c))
        dset_std = (data4D ** 2).mean(1)
    dset_std = dset_std**0.5
    dset_std_tile = np.tile(np.array([dset_std]).swapaxes(0,1), (1, n_r * n_c))
    data4D[dset_std_tile>0] /= dset_std_tile[dset_std_tile>0]
    
    data4D = data4D.reshape(n_x, n_y, n_r* n_c)
    data4D = data4D.reshape(n_x, n_y, n_r, n_c)
    
    return data4D