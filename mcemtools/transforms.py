import numpy as np

def get_polar_coords(n_rows, n_clms, centre):
        cc, rr = np.meshgrid(np.arange(n_rows), np.arange(n_clms))
    
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
                               * n_distinct_angles).astype('int')
        anglesq_pix_in_polar = anglesq - anglesq.min()
        anglesq_pix_in_polar = (anglesq_pix_in_polar / anglesq_pix_in_polar.max() 
                               * n_distinct_angles).astype('int')
                                                      
        rads_pix_in_polar = (rads / rads.max() * n_distinct_rads).astype('int')
        
        angles_pix_in_polar = angles_pix_in_polar.ravel()
        anglesq_pix_in_polar = anglesq_pix_in_polar.ravel()
        rads_pix_in_polar = rads_pix_in_polar.ravel()
        rr = rr.ravel()
        cc = cc.ravel()
        return (angles_pix_in_polar, anglesq_pix_in_polar, 
                rads_pix_in_polar, rr, cc)

def polar2image(data, n_rows, n_clms, dataq = None, centre = None,
                get_polar_coords_output = None):
    """ 
        :param dataq:
            To those who ignore loss of information at the angle 0, you have to
            make two polar images out of a cartesian image, one beginning from 
            angle 0 and the other from another angle far from zero, better be 
            180. Then you have to process both images, and then give it back to
            this function. Use dataq for the second one....see? you didn't pay
            attention... Yes! it importnat....Hey!, I said it is important.
    """

    if dataq is None:
        dataq = data
    else:
        assert dataq.shape == data.shape,\
            'dataq should have the same type, shape and dtype as data'

    data_shape = data.shape
    n_distinct_angles = data_shape[0] - 1
    n_distinct_rads = data_shape[1] - 1
    data_shape_rest = data_shape[2:]
    if (centre is None):
        centre = (n_rows//2, n_clms//2)
    
    if get_polar_coords_output is None:
        angles_pix_in_polar, anglesq_pix_in_polar, rads_pix_in_polar, rr, cc = \
            get_polar_coords(n_rows, n_clms, centre)
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
               n_distinct_angles = 360,
               n_distinct_rads = None,
               centre = None,
               get_polar_coords_output = None):
    """ image to polar transform
    
        :param get_polar_coords_output:
            there is a function up there called get_polar_coords. It produces
            the polar coordinates. If user likes to call the function first
            generate coordinates, then user can pass these coordinates to these
            two funcitons any number of times. If user does not call thi
            function at first and does not provide it to image2polar and to
            polar2image, the functions will call it. It is fast, but you can
            clearly see how I am avoiding the absurdity of using OOP
            for this task.    
    """

    data_shape = data.shape
    n_rows = data_shape[0]
    n_clms = data_shape[1]
    data_shape_rest = data_shape[2:]
    
    if(n_distinct_rads is None):
        n_distinct_rads = int(np.ceil(((n_rows/2)**2 + (n_clms/2)**2)**0.5))
    if (centre is None):
        centre = (n_rows//2, n_clms//2)
    
    if get_polar_coords_output is None:
        angles_pix_in_polar, anglesq_pix_in_polar, rads_pix_in_polar, rr, cc = \
            get_polar_coords(n_rows, n_clms, centre)
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

if __name__ == '__main__':
    
    import matplotlib.pyplot as plt

    data = plt.imread('c_CET-C6.png')
    print(data.shape)
    n_distinct_angles = 720
    n_distinct_rads = 720
    
    polar_image, polar_imageq, polar_mask, _ = image2polar(
        data, n_distinct_angles, n_distinct_rads)
    print(f'polar_image.shape: {polar_image.shape}')
    
    recon_image, recon_mask = polar2image(polar_image, 
                                          data.shape[0], 
                                          data.shape[1],
                                          polar_imageq)
    
    plt.figure(), plt.imshow(data, cmap = 'gray'), plt.colorbar()
    plt.figure(), plt.imshow(polar_image)
    plt.figure(), plt.imshow(polar_imageq)
    plt.figure(), plt.imshow(recon_image, cmap = 'gray'), plt.colorbar()
    plt.show()