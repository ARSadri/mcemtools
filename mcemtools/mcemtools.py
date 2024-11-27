import numpy as np
import pathlib
import matplotlib.pyplot as plt
from lognflow import printprogress, select_directory, lognflow, plt_imshow
from .analysis import sum_4D, swirl_and_sum
from .masking import mask2D_to_4D
from .transforms import data4D_to_frame
import time
import mcemtools

def dummy_function(*args, **kwargs): ...

def nvidia_smi_line(important_line = 9):
    memory_free_info = sp.check_output('nvidia-smi').decode('ascii').split('\n')
    return memory_free_info[important_line]

class viewer_4D:
    def __init__(self, data4D, 
                 statistics_4D = sum_4D,
                 logger = print,
                 sleep_between_moves = 0.5,
                 min_shape_edge_width = 2.0,
                 title = 'viewer_4D'):
        import napari
        try:
            self.data4D = np.load(data4D)
        except:
            self.data4D = data4D
        self.statistics_4D = statistics_4D
        self.logger = logger
        self.sleep_between_moves = sleep_between_moves
        self.min_shape_edge_width = min_shape_edge_width
        
        self.data4D_shape = self.data4D.shape
        self.data4D_shape_list = np.array(self.data4D_shape)
        self.viewers_list = [napari.Viewer(title = title + '_PACBED'), 
                             napari.Viewer(title = title + '_STEM')]
        STEM_img, PACBED = self.statistics_4D(self.data4D)
        self.viewers_list[0].add_image(PACBED)
        self.viewers_list[1].add_image(STEM_img)
        
        for viewer in self.viewers_list:
            viewer.bind_key(key = 'Up'   , func = self.move_up)
            viewer.bind_key(key = 'Down' , func = self.move_down)
            viewer.bind_key(key = 'Left' , func = self.move_left)
            viewer.bind_key(key = 'Right', func = self.move_right)
            viewer.bind_key(key = 'i'    , func = self.print_shape_info)
            viewer.bind_key(key = 'm'    , func = self.show_mask)
            viewer.bind_key(key = 'F5'   , func = self.update_by_masked_sum_4D)

        self.viewers_list[1].bind_key(key = 'g', func = self.show_frame4D)
        self.viewers_list[0].bind_key(key = 'c', func = self.show_CoM)
        
        self.viewers_list[0].mouse_drag_callbacks.append(self.mouse_drag_event)
        self.viewers_list[1].mouse_drag_callbacks.append(self.mouse_drag_event)
        
        self.mask2D_list = []
        self.mask2D_list.append(np.ones(
            (self.data4D_shape[2], self.data4D_shape[3]), dtype='int8'))
        self.mask2D_list.append(np.ones(
            (self.data4D_shape[0], self.data4D_shape[1]), dtype='int8'))
        self.mask2D_neg_list = []
        self.mask2D_neg_list.append(np.zeros(
            (self.data4D_shape[2], self.data4D_shape[3]), dtype='int8'))
        self.mask2D_neg_list.append(np.zeros(
            (self.data4D_shape[0], self.data4D_shape[1]), dtype='int8'))
        self.move_perv_time_time = 0
        napari.run()
    
    def show_mask(self, viewer):
        self.update_by_masked_sum_4D(viewer)
        viewer_index = self.viewers_list.index(viewer)
        self.logger(f'showing mask for viewer {viewer_index}')
        mask2D = self.mask2D_list[viewer_index] - self.mask2D_neg_list[viewer_index]
        try:
            fpath = self.logger.log_single(
                'mask', mask2D,time_tag = False)
            print(f'mask saved to: {fpath}')
        except:
            pass
        plt.figure()
        plt.imshow(mask2D)
        plt.title(f'mask for viewer {viewer_index}')
        plt.show()
    
    def get_mask2D(self, shape_layer, mask_shape):
        from skimage.draw import polygon2mask
        from scipy.ndimage import binary_dilation, binary_fill_holes
        
        label2D = shape_layer.to_labels(mask_shape)
        if label2D.sum() > 0:
            n_shapes = len(shape_layer.data)
            mask2D = np.zeros((n_shapes, ) + mask_shape, dtype='int8')
            for shape_cnt in range(n_shapes):
                _mask2D = np.ones(mask_shape, dtype='int8')
                sh_width = int(shape_layer.edge_width[shape_cnt])
                sh_type = shape_layer.shape_type[shape_cnt]
                if ((sh_type == 'path') | (sh_type == 'line')):
                    if (sh_width < 2):
                        pt_data = shape_layer.data[shape_cnt]
                        _mask2D = polygon2mask(mask_shape ,pt_data)
                    else:
                        _mask2D = label2D.copy()
                        _mask2D[_mask2D != shape_cnt + 1] = 0                   #<-- OVERLAPING MASKS DISAPPEARING ISSUE, Maybe put all masks in a 3D array and get the max
                else:
                    _mask2D = label2D.copy()
                    _mask2D[_mask2D != shape_cnt + 1] = 0                       #<-- OVERLAPING MASKS DISAPPEARING ISSUE, Maybe put all masks in a 3D array and get the max
                    
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
                    if sh_width / self.min_shape_edge_width != sh_width // 2:
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
                mask2D[shape_cnt, _mask2D > 0] = 1
            mask2D = mask2D.max(0).squeeze().astype('int8')
        else:
            mask2D = np.ones(mask_shape, dtype='int8')
        return mask2D
                
    def update_by_masked_sum_4D(self, viewer, *args, **kwargs):
        viewer_index = self.viewers_list.index(viewer)
        
        data4D_shape_select = viewer.layers[0].data.shape
        mask2D = np.ones(data4D_shape_select, dtype='int8')
        if(len(viewer.layers) > 1):
            mask2D = self.get_mask2D(viewer.layers[1], data4D_shape_select)
        mask2D_neg = np.zeros(data4D_shape_select, dtype='int8')
        if(len(viewer.layers) > 2):
            mask2D_neg = self.get_mask2D(viewer.layers[2], data4D_shape_select)
        
        if( ((self.mask2D_list[viewer_index] != mask2D).sum()>0) |
            ( (self.mask2D_neg_list[viewer_index] != mask2D_neg).sum()>0) ):
            self.mask2D_list.__setitem__(viewer_index, mask2D.copy())
            self.mask2D_neg_list.__setitem__(viewer_index, mask2D_neg.copy())
            if(viewer_index == 0):
                if (mask2D==1).sum() == 1:
                    ind_r, ind_c = np.where(mask2D==1)
                    STEM_img = self.data4D[..., ind_r, ind_c].squeeze().copy()
                else:
                    mask4D = np.zeros(self.data4D_shape, dtype='int8')
                    mask4D[:, :, mask2D==1] = 1
                    STEM_img, _ = self.statistics_4D(self.data4D, mask4D)
                if(mask2D_neg.sum() > 0):
                    if (mask2D_neg==1).sum() == 1:
                        ind_r, ind_c = np.where(mask2D_neg==1)
                        STEM_img -= self.data4D[..., ind_r, ind_c].squeeze().copy()
                    else:
                        mask4D = np.zeros(self.data4D_shape, dtype='int8')
                        mask4D[:, :, mask2D_neg==1] = 1
                        _STEM_img, _ = self.statistics_4D(self.data4D, mask4D)
                        STEM_img -= _STEM_img/mask2D_neg.sum()*(
                            mask2D==1).sum()
                
                self.viewers_list[1].layers[0].data = STEM_img
                self.logger('STEM image updated')
            if(viewer_index == 1):
                if (mask2D==1).sum() == 1:
                    ind_x, ind_y = np.where(mask2D==1)
                    PACBED = self.data4D[ind_x, ind_y].copy()
                else:
                    mask4D = np.zeros(self.data4D_shape, dtype='int8')
                    mask4D[mask2D==1, :, :] = 1
                    _, PACBED = self.statistics_4D(self.data4D, mask4D)
                    
                if(mask2D_neg.sum() > 0):
                    print('-+'*40)
                    if (mask2D_neg==1).sum() == 1:
                        ind_r, ind_c = np.where(mask2D_neg==1)
                        PACBED -= self.data4D[ind_r, ind_c].squeeze().copy()
                    else:
                        mask4D = np.zeros(self.data4D_shape, dtype='int8')
                        mask4D[mask2D_neg==1] = 1
                        _, _PACBED = self.statistics_4D(self.data4D, mask4D)
                        self.logger(f'PACBED:{PACBED.mean()}')
                        self.logger(f'PACBED:{_PACBED.mean()}')
                        self.logger(f'PACBED:{mask2D.sum()/mask2D_neg.sum()}')
                        PACBED -= _PACBED / mask2D_neg.sum()* mask2D.sum()
                        
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
    
    def show_CoM(self, viewer):
        print('c pressed', flush = True)
        viewer_index = self.viewers_list.index(viewer)
        self.logger(f'showing CoM for the mask in viewer {viewer_index}')
        data4D_shape_select = viewer.layers[0].data.shape
        mask2D = np.ones(data4D_shape_select, dtype='int8')
        if(len(viewer.layers) > 1):
            mask2D = self.get_mask2D(viewer.layers[1], data4D_shape_select)
        
        self.logger(f'mask2D.sum() " {mask2D.sum()}')
        if(mask2D.sum() > 0):
            mask4D = mask2D_to_4D(mask2D, self.data4D.shape)
            CoM_x, CoM_y = mcemtools.centre_of_mass_4D(self.data4D.copy(), mask4D)
            try:
                self.logger.log_imshow('CoM_x', CoM_x)
                self.logger.log_single('CoM_x', CoM_x)
                self.logger.log_imshow('CoM_y', CoM_y)
                self.logger.log_single('CoM_y', CoM_y)
            except:
                pass
            plt_imshow(CoM_x + 1j * CoM_y, cmap = 'complex',
                       title = f'centre_of_mass_4D for viewer {viewer_index}')
            plt.show()
        else:
            self.logger('No mask is selected, and I am surely not turning ' + \
                        'the whole datset into a frame')
    
    def show_frame4D(self, viewer):
        print('f pressed', flush = True)
        viewer_index = self.viewers_list.index(viewer)
        self.logger(f'showing frame for viewer {viewer_index}')
        data4D_shape_select = viewer.layers[0].data.shape
        mask2D = np.ones(data4D_shape_select, dtype='int8')
        if(len(viewer.layers) > 1):
            mask2D = self.get_mask2D(viewer.layers[1], data4D_shape_select)
        
        self.logger(f'mask2D.sum() " {mask2D.sum()}')
        if(mask2D.sum() > 0):
            _data4D = self.data4D.copy()
            if(viewer_index == 0):
                _data4D[:, :, mask2D == 0] = 0
            if(viewer_index == 1):
                _data4D[mask2D == 0] = 0
            indi, indj = np.where(mask2D == 1)
            _data4D = _data4D[indi.min():indi.max(), indj.min():indj.max()]
            framed4D = data4D_to_frame(_data4D)
            try:
                self.logger.log_single(
                    'mask_for_frame', self.mask2D_list[viewer_index])
                self.logger.log_single(
                    'framed4D', framed4D)
            except:
                pass
            plt.figure()
            plt.imshow(framed4D)
            plt.title(f'framed4D for viewer {viewer_index}')
            plt.show()
        else:
            self.logger('No mask is selected, and I am surely not turning ' + \
                        'the whole datset into a frame')
        
    def move(self, viewer, axis, sign):
        viewer_index = self.viewers_list.index(viewer)
        if(len(viewer.layers) <= 1):
            return
        layer = viewer.layers[1]
        try:
            layer_1 = int(str(viewer.layers.selection).split('[')[1].split(']')[0])
            if layer_1 == 1:
                layer = viewer.layers[2]
        except:
            pass
        time_since_last_move = time.time() - self.move_perv_time_time
        n_shapes = len(layer.data)
        selected_shape_cnt_list = list(layer.selected_data)
        if selected_shape_cnt_list:
            cdata = []
            for shape_cnt in range(n_shapes):
                _cdata = layer.data[int(shape_cnt)]
                if shape_cnt in selected_shape_cnt_list:
                    _cdata[:, axis] += sign
                cdata.append(_cdata)
            layer.data = cdata
            selected_data = set()
            for shape_cnt in selected_shape_cnt_list:
                selected_data.add(shape_cnt)
            layer.selected_data = selected_data
        sleep_between_moves = self.sleep_between_moves
        if (self.mask2D_list[viewer_index]==1).sum() == 1:
            sleep_between_moves = 0
        if time_since_last_move > sleep_between_moves:
            self.update_by_masked_sum_4D(viewer)
        self.move_perv_time_time = time.time()

    def move_up(self, viewer):
        self.move(viewer, axis = 0, sign = -1)
    def move_down(self, viewer):
        self.move(viewer, axis = 0, sign = 1)
    def move_left(self, viewer):
        self.move(viewer, axis = 1, sign = -1)
    def move_right(self, viewer):
        self.move(viewer, axis = 1, sign = 1)