import numpy as np
import pathlib
import matplotlib.pyplot as plt
from lognflow import printprogress, select_directory, lognflow
from .analysis import sum_4D, swirl_and_sum
from .masking import mask2D_to_4D

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
    def __init__(self, data4D, 
                 statistics_4D = sum_4D, logger = print):
        import napari
        self.data4D = data4D
        self.statistics_4D = statistics_4D
        self.logger = logger

        self.data4D_shape = self.data4D.shape
        self.data4D_shape_list = np.array(self.data4D_shape)
        self.viewers_list = [napari.Viewer(), napari.Viewer()]
        STEM_img, PACBED = self.statistics_4D(self.data4D)
        self.viewers_list[0].add_image(PACBED)
        self.viewers_list[1].add_image(STEM_img)

        self.viewers_list[0].bind_key(
            key = 'i', func = self.print_shape_info)
        self.viewers_list[1].bind_key(
            key = 'i', func = self.print_shape_info)
        
        self.viewers_list[0].bind_key(
            key = 'm', func = self.show_mask)
        self.viewers_list[1].bind_key(
            key = 'm', func = self.show_mask)
        
        self.viewers_list[0].bind_key(
            key = 'F5', func = self.update_by_masked_sum_4D)
        self.viewers_list[1].bind_key(
            key = 'F5', func = self.update_by_masked_sum_4D)
        self.viewers_list[0].mouse_drag_callbacks.append(self.mouse_drag_event)
        self.viewers_list[1].mouse_drag_callbacks.append(self.mouse_drag_event)
        
        self.mask2D_list = []
        self.mask2D_list.append(np.ones(
            (self.data4D_shape[2], self.data4D_shape[3]), dtype='int8'))
        self.mask2D_list.append(np.ones(
            (self.data4D_shape[0], self.data4D_shape[1]), dtype='int8'))
        
        napari.run()
    
    def show_mask(self, viewer):
        self.update_by_masked_sum_4D(viewer)
        viewer_index = self.viewers_list.index(viewer)
        plt.figure(), plt.imshow(self.mask2D_list[viewer_index])
        plt.title(f'mask for viewer {viewer_index}'), plt.show()
    
    def get_mask2D(self, shape_layer, mask_shape):
        from skimage.draw import polygon2mask
        from scipy.ndimage import binary_dilation, binary_fill_holes
        mask2D = np.zeros(mask_shape, dtype='int8')
        
        label2D = shape_layer.to_labels(mask_shape)
        if label2D.sum() > 0:
            for shape_cnt in range(len(shape_layer.data)):
                _mask2D = np.ones(mask_shape, dtype='int8')
                sh_width = int(shape_layer.edge_width[shape_cnt])
                sh_type = shape_layer.shape_type[shape_cnt]
                if sh_type == 'path':
                    if (sh_width < 2):
                        pt_data = shape_layer.data[shape_cnt]
                        _mask2D = polygon2mask(mask_shape ,pt_data)
                    else:
                        _mask2D = label2D.copy()
                        _mask2D[_mask2D != shape_cnt + 1] = 0
                else:
                    _mask2D = label2D.copy()
                    _mask2D[_mask2D != shape_cnt + 1] = 0
                if sh_width == 2:
                    _mask2D_swirl_sum = swirl_and_sum(_mask2D)
                    _mask2D_swirl_sum[_mask2D_swirl_sum >= 7] = 0
                    _mask2D_swirl_sum[_mask2D_swirl_sum>0] = 1
                    _mask2D = _mask2D_swirl_sum.copy()
                elif sh_width > 2:
                    _mask2D_swirl_sum = swirl_and_sum(_mask2D)
                    _mask2D_swirl_sum[_mask2D_swirl_sum >= 4] = 0
                    _mask2D_swirl_sum[_mask2D_swirl_sum>0] = 1
                    if sh_width == 3:
                        switers = 1
                    else:
                        switers = sh_width // 2
                    if sh_width / 2.0 != sh_width // 2:
                        _mask2D = binary_dilation(
                            _mask2D_swirl_sum, iterations = switers)
                    else:
                        _mask2D = binary_dilation(
                            _mask2D_swirl_sum, iterations = switers - 1)
                        _mask2D_filled_out = binary_fill_holes(_mask2D)
                        _mask2D_filled_out[_mask2D == 1] = 0
                        _mask2D_filled_out = 1 - _mask2D_filled_out
                        _erroded = binary_dilation(_mask2D_filled_out)
                        _erroded[_mask2D_filled_out == 1] = 0
                        _mask2D[_erroded == 1] = 1
                mask2D[_mask2D > 0] = 1
        else:
            mask2D += 1
        return mask2D
                
    def update_by_masked_sum_4D(self, viewer, *args, **kwargs):
        viewer_index = self.viewers_list.index(viewer)
        if(len(viewer.layers) > 1):
            data4D_shape_select = viewer.layers[0].data.shape
            mask2D = self.get_mask2D(viewer.layers[1], data4D_shape_select)
            if( (self.mask2D_list[viewer_index] != mask2D).sum()>0):
                self.mask2D_list.__setitem__(viewer_index, mask2D.copy())
                mask4D = np.zeros(self.data4D_shape, dtype='int8')
                if(viewer_index == 0):
                    mask4D[:, :, mask2D==1] = 1
                    STEM_img, PACBED = self.statistics_4D(self.data4D, mask4D)
                    self.viewers_list[1].layers[0].data = STEM_img
                    self.logger('STEM image updated')
                if(viewer_index == 1):
                    mask4D[mask2D==1, :, :] = 1
                    STEM_img, PACBED = self.statistics_4D(self.data4D, mask4D)
                    self.viewers_list[0].layers[0].data = PACBED
                    self.logger('PACBED image updated')

    def mouse_drag_event(self, viewer, event):
        dragged = False
        yield
        while event.type == 'mouse_move':
            dragged = True
            yield
        if dragged:
            self.update_by_masked_sum_4D(viewer)
            
    def print_shape_info(self, viewer):
        self.logger(f'shape_type:{viewer.layers[1].shape_type}')
        self.logger(f'data:{viewer.layers[1].data}')
        self.logger(f'edge_width:{viewer.layers[1].edge_width}')