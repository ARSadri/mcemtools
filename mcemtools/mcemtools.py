import numpy as np
import pathlib
import matplotlib.pyplot as plt
from lognflow import printprogress
from .analysis import sum_4D

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

def pltfig_to_numpy(fig):
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
    This function generates a 4D dataset of images with shape
    (n_x, n_y, n_r, n_c) where in each image the value "x, y" is written as a text
    that fills the image. As such, later when working with such a dataset you can
    look at the image and know which index it had before you use it.
    
    Follow this recipe to make good images:
    
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
            buf = pltfig_to_numpy(fig)
            plt.close()
            buf2 = buf[::buf.shape[0]//n_r, ::buf.shape[1]//n_c, :3].mean(2)
            buf2 = buf2[:n_r, :n_c]
            dataset[ind_x, ind_y] = buf2.copy()
            if(verbose):
                pBar()
    return dataset

import re,os,numpy as np

def open_muSTEM_binary(filename):
    '''opens binary with name filename outputted from the muSTEM software
        This peice of code is modified from muSTEM repo.
    '''
    filename = pathlib.Path(filename)
    assert filename.is_file(), f'{filename.absolute()} does not exist'
    m = re.search('([0-9]+)x([0-9]+)',filename)
    if m:
        y = int(m.group(2))
        x = int(m.group(1))
    #Get file size and intuit datatype
    size =  os.path.getsize(filename)
    if (size/(y*x) == 4):
        d_type = '>f4'
    elif(size/(y*x) == 8):
        d_type = '>f8'
    #Read data and reshape as required.
    return np.reshape(np.fromfile(filename, dtype = d_type),(y,x))

class viewer_4D:
    def __init__(self, data4D):
        import napari
        self.data4D = data4D
        self.data4D_shape = self.data4D.shape
        self.data4D_shape_list = np.array(self.data4D_shape)
        self.viewers_list = [napari.Viewer(), napari.Viewer()]
        self.viewers_list[0].add_image(self.data4D.sum(1).sum(0).squeeze())
        self.viewers_list[1].add_image(self.data4D.sum(3).sum(2).squeeze())

        # self.viewers_list[0].bind_key(key = 'n', func = self.get_shape_info0)
        # self.viewers_list[1].bind_key(key = 'n', func = self.get_shape_info1)
        
        self.viewers_list[0].mouse_drag_callbacks.append(self.get_shape_info0)
        self.viewers_list[1].mouse_drag_callbacks.append(self.get_shape_info1)
        
        napari.run()
        
    def get_mask(self, shape_layer, viewer_axis):
        from skimage.draw import polygon2mask
        mask4D = np.ones(self.data4D_shape, dtype='int8')

        data4D_shape_select = tuple(
            self.data4D_shape_list[np.array(viewer_axis)])
        mask2D = shape_layer.to_labels(data4D_shape_select) > 0
        if mask2D.sum() > 0:
            for shape_cnt in range(len(shape_layer.data)):
                if shape_layer.shape_type[shape_cnt] == 'path':
                    pt_data = shape_layer.data[shape_cnt]
                    mask2D += polygon2mask(data4D_shape_select ,pt_data)
            if(viewer_axis[0] == 0):
                mask4D[mask2D==0, :, :] = 0
            if(viewer_axis[0] == 2):
                mask4D[:, :, mask2D==0] = 0
        return mask4D
            
    def get_shape_info0(self, viewer, event):
        dragged = False
        yield
        while event.type == 'mouse_move':
            # print(event.position)
            dragged = True
            yield
        if dragged:
            # print('drag end')
            try:
                mask4D = self.get_mask(viewer.layers[1], (2, 3))
            except:
                mask4D = np.ones(self.data4D.shape,dtype='int8')
            totI, _ = sum_4D(self.data4D, mask4D)
            self.viewers_list[1].layers[0].data = totI
            print('STEM updated')
            
        
    def get_shape_info1(self, viewer, event):
        dragged = False
        yield
        while event.type == 'mouse_move':
            # print(event.position)
            dragged = True
            yield
        if dragged:
            # print('drag end')
            try:
                mask4D = self.get_mask(viewer.layers[1], (0, 1))
            except:
                mask4D = np.ones(self.data4D.shape,dtype='int8')
            _, PACBED = sum_4D(self.data4D, mask4D)
            self.viewers_list[0].layers[0].data = PACBED
            print('PACBED updated')
            
            
            