import numpy as np
import scipy

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

class polar_transform:
    def __init__(self, image_shape, centre, polar_shape):
        self.image_shape = image_shape
        self.polar_shape = polar_shape
        self.centre = centre
        self.get_polar_coords_output = \
            get_polar_coords(self.image_shape, self.centre, self.polar_shape)
    def image2polar(self, data):
        return image2polar(data, self.polar_shape[0], self.polar_shape[1],
                           self.centre, self.get_polar_coords_output)
    def polar2image(self, data, dataq = None):
        return polar2image(data, self.image_shape, dataq, self.centre,
                           self.get_polar_coords_output)

def data4D_to_multichannel(data4D):
    """data4D to multichannel
        Given the input numpy array of shape n_x x n_y x n_r x n_c the output
        would simply be (n_r+2)*n_x x (n_c+2)*n_y.    
    """
    n_x, n_y, n_r, n_c = data4D.shape
    new_n_r = (n_r + 2) * n_x
    new_n_c = (n_c + 2) * n_y
    canv = np.zeros((new_n_r, new_n_c), dtype=data4D.dtype)
    for xcnt in range(n_x):
        for ycnt in range(n_y):
            canv[xcnt*n_r + 1 : (xcnt + 1)*n_r - 1, 
                 ycnt*n_c + 1 : (ycnt + 1)*n_c - 1] = data4D[xcnt, ycnt]
    return canv
    