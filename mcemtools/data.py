import re
import os
import numpy as np
from .masking import image_by_windows
import mcemtools
from lognflow import printprogress
import scipy
from itertools import product

def channel_to_image(image_by_ch, detector_response):
    n_ch, _, _ = detector_response.shape
    n_ch_ = len(image_by_ch)
    assert n_ch_ == n_ch, 'image_by_ch should have same number of channels as detector_response'
    img = (image_by_ch[:, None, None] * detector_response).sum(0)
    return img

def load_dm4(filename):
    from hyperspy.api import load as hyperspy_api_load
    return hyperspy_api_load(filename)

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

def check_raw_filename(fpath, scan_shape = None):
    import re
    from pathlib import Path
    fpath = Path(fpath)

    pattern = r"^scan_x(\d+)_y(\d+)\.raw$"
    
    match = re.match(pattern, fpath.name)
    if match:
        nx, ny = map(int, match.groups())
        if scan_shape is not None:
            if (ny, nx) != scan_shape:
                print(f"load_raw: Be careful! It seems the given scan_shape {scan_shape} "
                    f"does not match the filename ({ny}, {nx}).")
        return ny, nx

def load_raw(fpath, scan_shape: tuple[int, int] = None, 
             detector_shape: tuple[int, int] = (128, 128), 
             dtype = np.float32, footer_length = 256):
    
    scan_shape_ = check_raw_filename(fpath, scan_shape)
    if scan_shape is None:
        scan_shape = scan_shape_
    dt = np.dtype([("data", dtype, detector_shape),("footer", dtype, footer_length)])
    dataset = np.fromfile(fpath, dtype = dt)["data"].reshape(scan_shape + detector_shape)
    return dataset

class mesh_maker_2D:
    def __init__(self, input_image, ground_truth = None):
        
        if ground_truth is None:
            ground_truth = input_image.copy()
        
        self.input_image_shape = input_image.shape
        self.n_pts = self.input_image_shape[0] * self.input_image_shape[1]
        xx, yy = np.meshgrid(np.arange(self.input_image_shape[1], dtype='int'),
                             np.arange(self.input_image_shape[0], dtype='int'))
        xx = xx.ravel()
        xx = xx - xx.mean()
        xx = xx / xx.std()
        yy = yy.ravel()
        yy = yy - yy.mean()
        yy = yy / yy.std()
        
        self.X_in = np.array([xx, yy]).T.astype('float32')
        
        self.Y_lable = input_image.copy().ravel().astype('float32')
        self.Y_lable_mean = self.Y_lable.mean()
        self.Y_lable -= self.Y_lable_mean
        self.Y_lable /= self.Y_lable.std()
        self.Y_lable = np.array([self.Y_lable]).swapaxes(0,1)
        
        self.GNDTruth = ground_truth.copy().ravel().astype('float32')
        self.GNDTruth_mean = self.GNDTruth.mean()
        self.GNDTruth -= self.GNDTruth_mean
            
    def dist2Truth(self, pred, ind):
        return np.fabs(pred - self.GNDTruth[ind])
    
    def dist2label(self, pred, ind):
        return np.fabs(pred - self.Y_lable[ind])
    
    def reconstruct(self, outimg_viewed):
        return outimg_viewed.reshape(*self.input_image_shape)

    def __call__(self, inds):
        try:
            _ = inds.shape[0]
        except:
            inds = np.array([inds])
        return(self.X_in[inds], self.Y_lable[inds])

def mask_random_pixels(inimg_set, mask_rate, random_function = np.zeros):
    """
    """
    inimg_set = inimg_set.copy()
    for imgcnt, inimg in enumerate(inimg_set):
        inimg = inimg.squeeze()
        nprange = np.arange(inimg.size).astype('int')
        np.random.shuffle(nprange)
        mask_vec = np.ones(nprange.shape)
        mask_vec[nprange[int(mask_rate*inimg.size):]] = 0
        mask = mask_vec.reshape(*inimg.shape)
        inimg[mask == 0] = random_function((mask == 0).sum())
        inimg_set[imgcnt] = inimg.copy()
    return inimg_set
    
class data_maker_2D:
    def __init__(self, inimg, groundtruth, win_shape,
                 skip = (1,1), mask_rate = 0.5):
        assert inimg.shape == groundtruth.shape
        
        self.inimg_mean = inimg.mean()
        
        self.mask_rate = mask_rate
        inimg = inimg.astype('float32')
        self.n_r, self.n_c = inimg.shape
        
        self.imbywin = image_by_windows(
            inimg.shape, win_shape, skip, method = 'fixed')
        self.Y_label = self.imbywin.image2views(inimg).copy()
        self.Y_label = np.array([self.Y_label]).swapaxes(0, 1)

        self.GNDTruth = self.imbywin.image2views(groundtruth).copy()
        self.GNDTruth = np.array([self.GNDTruth]).swapaxes(0, 1)
        
        self.n_pts = self.imbywin.n_pts

        self.randomize()

    def randomize(self):
        self.X_in = self.Y_label.copy()
        for cnt, y_lbl in enumerate(self.Y_label):
            self.X_in[cnt] = mask_random_pixels(y_lbl, self.mask_rate)

    def reconstruct(self, outimg_viewed):
        return self.imbywin.views2image(outimg_viewed)
            
    def dist2Truth(self, pred, ind):
        return np.linalg.norm(pred - self.GNDTruth[ind])
    
    def dist2label(self, pred, ind):
        return np.linalg.norm(pred - self.Y_label[ind])
    
    def __call__(self, inds):
        try:
            _ = inds.shape[0]
        except:
            inds = np.array([inds])
        return(self.X_in[inds], self.Y_label[inds])

def np_random_poisson_no_zeros(img):
    img_noisy = 0*img.copy()
    while img_noisy.sum() == 0:
        img_noisy = np.random.poisson(img).astype('float32')
    return img_noisy

class data_maker_4D:
    def __init__(self, inimg, groundtruth, len_side = 3,
                 trainable_area_I4D = None):
        
        assert len_side == (len_side//2)*2 + 1,\
            'data_maker_I4D:len_side should be odd'
        self.len_side = len_side
        self.edgew = len_side // 2
        self.inimg_shape = inimg.shape
        n_x, n_y, n_r, n_c = inimg.shape
        self.n_x, self.n_y, self.n_r, self.n_c = inimg.shape
        self.dtype = inimg.dtype

        self.imbywin = mcemtools.image_by_windows(
            (n_x, n_y), (len_side, len_side), method = 'fixed')
        
        self.mask_range = np.ones((len_side, len_side), dtype = 'int')
        self.mask_range[len_side //2, len_side // 2] = 0
        print('mask_range:'); print(self.mask_range)
        self.mask_range = self.mask_range.ravel()
        
        inimg_viewed = self.imbywin.image2views(groundtruth)
        inimg_viewed = inimg_viewed.reshape(
            inimg_viewed.shape[0], self.len_side**2, self.n_r, self.n_c)
        self.GNDTruth = inimg_viewed[:,  self.mask_range == 0].copy()
                
        self.update(inimg)

        self.groundtruth_mu = self.reconstruct2D(
            self.GNDTruth.sum((1, 2, 3)).squeeze())
        self.groundtruth_PACBED = self.GNDTruth.sum((0, 1)).squeeze()
        self.noisy_mu = self.reconstruct2D(
            self.Y_label.sum((1, 2, 3)).squeeze())
        self.noisy_PACBED = self.Y_label.sum((0, 1)).squeeze()
        self.cropped_shape = (n_x - len_side , n_y - len_side, n_r, n_c)
        self.xx = self.imbywin.grid_rows.ravel() + len_side // 2
        self.yy = self.imbywin.grid_clms.ravel() + len_side // 2
        self.n_pts = len(self.imbywin)
        
        if trainable_area_I4D is not None:
            trainable_area = np.arange(n_x* n_y).reshape(n_x, n_y)
            self.trainable_inds = trainable_area[trainable_area_I4D > 0]
    
    def update(self, inimg, update_label = True):
        inimg_viewed = self.imbywin.image2views(inimg)
        inimg_viewed = inimg_viewed.reshape(
            inimg_viewed.shape[0], self.len_side**2, self.n_r, self.n_c)
        
        self.X_in = inimg_viewed[:, self.mask_range == 1]
        if update_label:
            self.Y_label = inimg_viewed[:, self.mask_range == 0]
        
    def reconstruct2D(self, outimg_viewed, indices = None):
        # The input shape is grid.npts
        img = self.imbywin.views2image(
                outimg_viewed, include_inds = indices, method = 'fixed',
                win_shape = (1, 1))
        return img[self.edgew:-self.edgew, self.edgew:-self.edgew]
    
    def reconstruct4D(self, viewed4D, indices = None):
        # The input shape is grid.npts, 1, n_r, n_c
        assert viewed4D.shape[1] == 1
        viewed4D = np.expand_dims(viewed4D, 2)
        img4d = self.imbywin.views2image(viewed4D, indices, method = 'fixed',
                                         win_shape = (1, 1))
        return img4d
                
    def dist2Truth(self, pred, ind):
        return np.linalg.norm(pred - self.GNDTruth[ind])
    
    def dist2label(self, pred, ind):
        return np.linalg.norm(pred - self.Y_label[ind])    

    def __call__(self, inds):
        try:
            _ = inds.shape[0]
        except:
            inds = np.array([inds])
        return(self.X_in[inds], self.Y_label[inds])
    
class feature_maker_4D:
    def __init__(self, inimg, groundtruth, len_side = 3,
                 trainable_area_I4D = None):
        assert len_side == (len_side//2)*2 + 1,\
            'data_maker_I4D:len_side should be odd'
        self.len_side = len_side
        self.inimg_shape = inimg.shape
        n_x, n_y, n_r, n_c = inimg.shape
        self.n_r = n_r
        self.n_c = n_c
        self.n_x = n_x
        self.n_y = n_y
        grid_x = np.arange(len_side // 2, n_x - len_side // 2, 1, dtype='int')
        grid_y = np.arange(len_side // 2, n_y - len_side // 2, 1, dtype='int')
        yy, xx = np.meshgrid(grid_y, grid_x)
        xx = xx.ravel()
        yy = yy.ravel()
        n_pts = xx.shape[0]
        self.n_xx = grid_x.shape[0]
        self.n_yy = grid_y.shape[0]
        self.n_pts = n_pts
        self.X_in     = \
            np.zeros((n_pts, len_side*len_side - 1, n_r, n_c), dtype='float32')
        self.Y_label  = \
            np.zeros((n_pts,                 1, n_r, n_c), dtype='float32')
        self.GNDTruth = \
            np.zeros((n_pts,                 1, n_r, n_c), dtype='float32')
        self.xx = xx.copy()
        self.yy = yy.copy()
        mask_range = np.ones(len_side*len_side).astype('int')
        mask_range[(len_side * len_side) // 2] = 0        
        self.mask_range = mask_range.copy()
        print(f'mask_range:{mask_range}')
                
        for gpt_cnt in range(n_pts):     
            a_tile = groundtruth[
                xx[gpt_cnt] - len_side // 2 : 
                    xx[gpt_cnt] + len_side // 2 + 1,
                yy[gpt_cnt] - len_side // 2 : 
                    yy[gpt_cnt] + len_side // 2 + 1].copy()
            a_tile = a_tile.reshape(len_side*len_side, n_r, n_c)
            self.GNDTruth[gpt_cnt] = a_tile[mask_range == 0].copy()

        self.update(inimg)

        self.groundtruth_mu = self.reconstruct2D(
            self.GNDTruth.sum(3).sum(2).sum(1).squeeze())
        self.groundtruth_PACBED = self.GNDTruth.sum(1).sum(0).squeeze()
        self.noisy_mu = self.reconstruct2D(
            self.Y_label.sum(3).sum(2).sum(1).squeeze())
        self.noisy_PACBED = self.Y_label.sum(1).sum(0).squeeze()
        self.cropped_shape = (grid_x.shape[0], grid_y.shape[0], n_r, n_c)
        
        self.trainable_inds = np.arange(self.n_pts, dtype='int')
    
    def filter(self, mch_img):                                                   ############################
        for cnt, img in enumerate(mch_img):
            
            img = np_random_poisson_no_zeros(img)
            # thresh = np.percentile(img, (cnt + 1) * 7)
            # img[img<thresh] = 0
            # img[img>=thresh] = 1
            
            mch_img[cnt] = img.copy()
        return mch_img
    
    def update(self, inimg):
        for gpt_cnt in range(self.n_pts):
            a_tile = inimg[
                self.xx[gpt_cnt] - self.len_side // 2 : 
                    self.xx[gpt_cnt] + self.len_side // 2 + 1,
                self.yy[gpt_cnt] - self.len_side // 2 : 
                    self.yy[gpt_cnt] + self.len_side // 2 + 1].copy()
            a_tile = a_tile.reshape(
                self.len_side*self.len_side, self.n_r, self.n_c)
            self.X_in[gpt_cnt] = np.tile(a_tile[self.mask_range == 0].copy(), (1, 8, 1, 1)) ###########################
            self.X_in[gpt_cnt] = self.filter(self.X_in[gpt_cnt])                         #####################################
            self.Y_label[gpt_cnt] = a_tile[self.mask_range == 0].copy()
    
    def reconstruct1D(self, out1D_viewed):
        n_pts = self.xx.shape[0]
        out1D_viewed = out1D_viewed.squeeze()
        output = np.zeros((self.inimg_shape[0],
                           self.inimg_shape[1], 2), dtype='float32')
        for gpt_cnt in range(n_pts):
            output[self.xx[gpt_cnt], self.yy[gpt_cnt]] = \
                out1D_viewed[gpt_cnt].copy()
        output = output[self.len_side//2 : -(self.len_side//2),
                        self.len_side//2 : -(self.len_side//2)].copy()
        return output              
    
    def reconstruct2D(self, outimg_viewed, indices = None):
        _outimg_viewed = np.zeros((self.n_xx * self.n_yy), dtype='float32')
        if indices is None:
            _outimg_viewed = outimg_viewed.copy()
        else:
            _outimg_viewed[indices] = outimg_viewed.copy()
        return _outimg_viewed.reshape(self.n_xx, self.n_yy)
    
    def reconstruct4D(self, viewed4D, indices = None):
        if indices is None:
            indices = range(self.xx.shape[0])
        viewed4D = viewed4D.squeeze()
        output = np.zeros(self.inimg_shape, dtype='float32')
        for gpt_cnt, gpt_ind in enumerate(indices):
            output[self.xx[gpt_ind], self.yy[gpt_ind]] = \
                viewed4D[gpt_cnt].copy()
        output = output[self.len_side//2 : -(self.len_side//2),
                        self.len_side//2 : -(self.len_side//2)].copy()
        return output              
        
    def dist2Truth(self, pred, ind):
        return np.linalg.norm(pred - self.GNDTruth[ind])
    
    def dist2label(self, pred, ind):
        return np.linalg.norm(pred - self.Y_label[ind])    

    def __call__(self, inds):
        try:
            _ = inds.shape[0]
        except:
            inds = np.array([inds])
        return(self.X_in[inds], self.Y_label[inds])    

def np_random_poisson_no_zeros(data4D_nonoise):
    data4D_noisy = 0*data4D_nonoise.copy()
    print('All patterns must have at least one electron', end = '')
    while (data4D_noisy.sum((2, 3)) == 0).sum() > 0:
        ne_ele_patterns = data4D_noisy.sum((2, 3)) == 0
        _data4D_noisy = np.random.poisson(
            data4D_nonoise[ne_ele_patterns]).astype('float32')
        data4D_noisy[ne_ele_patterns] = _data4D_noisy.copy()
        print(f', {(data4D_noisy.sum((2, 3)) == 0).sum()}', 
              end = '', flush = True)
    print(', done!')
    return data4D_noisy

class segmented_to_4D:
    def __init__(self, channel_based_data, detector_response):
        self.channel_based_data = channel_based_data
        self.detector_response = detector_response
        self.n_x, self.n_y, self.n_ch = channel_based_data.shape
        self.n_ch_det, self.n_r, self.n_c = detector_response.shape
        self.shape = (self.n_x, self.n_y, self.n_r, self.n_c)
        self.dtype = self.channel_based_data.dtype
        
        assert self.n_ch ==  self.n_ch_det, \
            'your detector response number of channels is different from data'

    def __getitem__(self, index):
        # Handle the case where index is a tuple of slices or integers
        if isinstance(index, tuple):
            row_index, col_index = index
            
            if isinstance(row_index, slice) or isinstance(col_index, slice):
                # Compute the actual slice ranges
                row_range = range(*row_index.indices(self.n_x)) \
                    if isinstance(row_index, slice) else range(self.n_x)
                col_range = range(*col_index.indices(self.n_y)) \
                    if isinstance(col_index, slice) else range(self.n_y)
                
                # Initialize the output array
                cbed_slices = np.zeros((len(row_range), len(col_range), self.n_r, self.n_c))
                
                force_print = False

                pbar = printprogress(self.n_ch, print_function=None)
                for segcnt in range(self.n_ch):
                    mask = self.detector_response[segcnt]
                    
                    # Apply mask and slice
                    data_slice = self.channel_based_data[row_index, col_index, segcnt]
                    cbed_slices += data_slice * mask
                    # np.tile(np.expand_dims(data_slice, -1), (1, 1, mask.sum()))
                    
                    ETA = pbar()
                    if (ETA > 120) & (pbar.in_print_function is None):
                        pbar = printprogress(self.n_ch - 1)
                
                return cbed_slices
            
            elif isinstance(row_index, int) and isinstance(col_index, int):
                # Handle single integer index
                row, col = row_index, col_index
                cbed = np.zeros((self.n_r, self.n_c))
                
                for segcnt in range(self.n_ch):
                    mask = self.detector_response[segcnt].copy()
                    ch_cbed = self.channel_based_data[row, col, segcnt].copy()
                    ch_cbed *= mask
                    cbed += ch_cbed.copy()
                
                return cbed
            
            else:
                raise IndexError("Index must be a tuple of slices or integers")
        
        else:
            raise IndexError("Index must be a tuple")

    def __len__(self):
        return self.n_x
    
    def __repr__(self):
        return (f"<ChannelTo4D: n_ch={self.n_ch}, n_x={self.n_x}," +
                f" n_y={self.n_y}, n_r={self.n_r}, n_c={self.n_c}>")
    
    def get_BF_weight_per_channel(self, BF_rad, max_rad):
        radius = BF_rad / max_rad * self.n_r / 2.0
        mask2d = mcemtools.annular_mask((self.n_r, self.n_c), radius = radius)
        
        w_BF = np.zeros(self.n_ch)
        w_DF = np.zeros(self.n_ch)
        for lblcnt in range(self.n_ch):
            w_BF[lblcnt] = ((self.detector_response[lblcnt]) * (mask2d == 1)).sum()\
                                               / (mask2d == 1).sum()
            w_DF[lblcnt] = ((self.detector_response[lblcnt]) * (mask2d == 0)).sum()\
                                               / (mask2d == 0).sum()
    
        return w_BF, w_DF, mask2d
    
    def get_stat_old(self, weights = None, normalize_for_com = True):
        data_by_ch, detector_response = self.channel_based_data , self.detector_response
        if weights is None:
            weights = np.ones(self.n_ch)
        weights_sum = weights.sum()

        stem = (data_by_ch * weights[None, None]).sum(-1)

        pacbed = np.zeros((self.n_r, self.n_c))
        for cnt in range(self.n_ch):
            pacbed += data_by_ch[..., cnt].mean() * weights[cnt] * detector_response[cnt]
        pacbed_com_x, pacbed_com_y = scipy.ndimage.center_of_mass(pacbed)

        if 0:
            data_by_ch = data_by_ch.copy()
            if normalize_for_com:
                data_by_ch_sum = np.expand_dims(data_by_ch.sum(-1), -1)
                data_by_ch_sum = np.tile(data_by_ch_sum, (1, 1, data_by_ch.shape[-1]))
                data_by_ch[data_by_ch_sum != 0] /= data_by_ch_sum[data_by_ch_sum != 0]
                data_by_ch[data_by_ch_sum == 0] = 0

            cent_x, cent_y = self.n_r//2, self.n_c//2
            com_x_ch = np.zeros(data_by_ch.shape)
            com_y_ch = np.zeros(data_by_ch.shape)
            for cnt in range(self.n_ch):
                mask_com_x, mask_com_y = scipy.ndimage.center_of_mass(detector_response[cnt])
                com_x_ch[..., cnt] = data_by_ch[..., cnt]*(mask_com_x - cent_x)
                com_y_ch[..., cnt] = data_by_ch[..., cnt]*(mask_com_y - cent_y)
            com_x = (com_x_ch * weights[None, None]).sum(-1)
            com_y = (com_y_ch * weights[None, None]).sum(-1)
        else:
            import torch
            n_x, n_y, n_ch = data_by_ch.shape
            label_CoM_data = torch.from_numpy(data_by_ch.copy())
            label_CoM_data /= label_CoM_data.sum(-1)[:, :, None]
            label_CoM_data = label_CoM_data.view(-1, n_ch)
            det_CoMs = mcemtools.analysis.CoM_detector(detector_response)
            com_x, com_y = mcemtools.analysis.CoM_channel_torch(label_CoM_data, det_CoMs)
            com_x = com_x.numpy().reshape(n_x, n_y)
            com_y = com_y.numpy().reshape(n_x, n_y)
            
        # if weights_sum:
        #     com_x = com_x / weights_sum
        #     com_y = com_y / weights_sum
        #     stem = stem / weights_sum

        return stem, pacbed, com_x, com_y, pacbed_com_x, pacbed_com_y
    
    def get_stat(self, normalize_for_com = True):
        data_by_ch, detector_response = self.channel_based_data , self.detector_response
        n_x, n_y, n_ch = data_by_ch.shape
        det_CoMs = mcemtools.analysis.CoM_detector(detector_response)

        stem = data_by_ch.sum(-1)

        pacbed = np.zeros((self.n_r, self.n_c))
        for cnt in range(self.n_ch):
            pacbed += data_by_ch[..., cnt].mean() * detector_response[cnt]
        pacbed_com_x, pacbed_com_y = scipy.ndimage.center_of_mass(pacbed)

        import torch
        label_CoM_data = torch.from_numpy(data_by_ch.copy())
        if normalize_for_com:
            label_CoM_data /= label_CoM_data.sum(-1)[:, :, None]
        label_CoM_data = label_CoM_data.reshape(-1, n_ch)
        com_x, com_y = mcemtools.analysis.CoM_channel_torch(label_CoM_data, det_CoMs)
        com_x = com_x.reshape(n_x, n_y).numpy()
        com_y = com_y.reshape(n_x, n_y).numpy()
            
        return stem, pacbed, com_x, com_y, pacbed_com_x, pacbed_com_y
    
    def filtered_by_kernel(self, coords, win_side, weights):
    
        coords += win_side
        coords_max = int(coords.max() + 1)
        kernel = np.zeros((coords_max, coords_max))
        for win_cnt_i, win_cnt_j in product(range(win_side), range(win_side)):
            kernel[coords[:, 0] - win_cnt_i, coords[:, 1] - win_cnt_j] = weights

        _, _, com_x, com_y, _, _ = self.get_stat()

        filtered_com_x = scipy.signal.convolve2d(com_x, kernel)
        filtered_com_y = scipy.signal.convolve2d(com_y, kernel)

        filtered_com_x = filtered_com_x[coords_max:-coords_max, coords_max:-coords_max]
        filtered_com_y = filtered_com_y[coords_max:-coords_max, coords_max:-coords_max]

        return filtered_com_x, filtered_com_y, kernel
    
def apply_detector_response_old(d4d, detector_response, verbose = False,
                            return_by_channle = True, segment_is_one_pixels = True):
    """
    Replace values in a multi-dimensional image based on a segmented labeled image.
    
    Parameters:
    ----------
    d4d : np.ndarray
        The multi-dimensional image array with shape (n_x, n_y, n_r, n_c),
        where each (n_r, n_c) slice is a single image.
    detector_response : np.ndarray of shape n_ch x n_r x n_c
        The area around the detector should be set to 0, and each segment
        should appear in a single channle with its artifacts.

    Returns:
    -------
    np.ndarray
        The modified image with each segment's values replaced by their sums.

    Raises:
    ------
    ValueError
        If detector_response does not match the last two dimensions of d4d.
    
    Notes:
    ------
    This function sums the values in each segment of d4d, defined by detector_response.
    Each segment (where detector_response == i) is replaced by the segment's sum in the
    modified d4d.
    """
    if detector_response.shape[1:] != d4d.shape[-2:]:
        raise ValueError("The shape of detector_response must match the last two dimensions of d4d.")

    
    if return_by_channle:
        data_by_ch = np.zeros(
            (d4d.shape[0], d4d.shape[1], len(detector_response)), dtype = d4d.dtype)
    else:
        modified_d4d = d4d.copy()

    if verbose: pbar = printprogress(d4d.shape[0] * d4d.shape[1])
    for i, j in np.ndindex(d4d.shape[:2]):
        segments_sum = d4d[None, i, j] * detector_response
        if return_by_channle:
            data_by_ch[i, j] = segments_sum.sum((1, 2))
        else:
            if segment_is_one_pixels:
                img_by_ch = segments_sum.sum((1, 2))
                modified_d4d[i, j] = (img_by_ch[:, None, None] * detector_response).sum(0)
            else:
                modified_d4d[i, j] = segments_sum.sum(0)
        if verbose: pbar()
                

    if return_by_channle:
        return data_by_ch
    else:
        return modified_d4d
    
def apply_detector_response(d4d, detector_response, detector_response_input = None, 
                            verbose = False, return_by_channle = True, 
                            segment_is_one_pixels = True):
    """
    Replace values in a multi-dimensional image based on a segmented labeled image.
    
    Parameters:
    ----------
    d4d : np.ndarray
        The multi-dimensional image array with shape (n_x, n_y, n_r, n_c),
        where each (n_r, n_c) slice is a single image.
    detector_response : np.ndarray of shape n_ch x n_r x n_c
        The area around the detector should be set to 0, and each segment
        should appear in a single channle with its artifacts.

    Returns:
    -------
    np.ndarray
        The modified image with each segment's values replaced by their sums.

    Raises:
    ------
    ValueError
        If detector_response does not match the last two dimensions of d4d.
    
    Notes:
    ------
    This function sums the values in each segment of d4d, defined by detector_response.
    Each segment (where detector_response == i) is replaced by the segment's sum in the
    modified d4d.
    """
    if detector_response_input is None: detector_response_input = detector_response.copy()
    # if detector_response.shape[1:] != d4d.shape[-2:]:
    #     raise ValueError("The shape of detector_response must match the last two dimensions of d4d.")

    
    if return_by_channle:
        data_by_ch = np.zeros(
            (d4d.shape[0], d4d.shape[1], len(detector_response)), dtype = d4d.dtype)
    else:
        modified_d4d = np.zeros(d4d.shape[:2] + detector_response.shape[1:], dtype = d4d.dtype)

    if verbose: pbar = printprogress(d4d.shape[0] * d4d.shape[1])
    for i, j in np.ndindex(d4d.shape[:2]):
        segments_sum = d4d[None, i, j] * detector_response_input
        if return_by_channle:
            data_by_ch[i, j] = segments_sum.sum((1, 2))
        else:
            if segment_is_one_pixels:
                img_by_ch = segments_sum.sum((1, 2))
                modified_d4d[i, j] = (img_by_ch[:, None, None] * detector_response).sum(0)
            else:
                modified_d4d[i, j] = segments_sum.sum(0)
        if verbose: pbar()
                

    if return_by_channle:
        return data_by_ch
    else:
        return modified_d4d

def generate_indices(labels_shape, batch_size, method = 'class_based'):
    """
    Generate indices for sampling data while ensuring specific class coverage requirements.

    Parameters
    ----------
    labels_shape : tuple
        A tuple `(n_classes, n_sample_per_class)` where:
        - `n_classes` is the number of classes.
        - `n_sample_per_class` is the number of samples available for each class.

    batch_size : int
        The number of samples per batch. Must evenly divide `n_classes`.

    method : str, optional
        The method to generate indices. Options are:
        - `'random'`: Randomly shuffle all indices and group them into batches.
        - `'class_based'`: Ensures each batch contains a balanced distribution of class indices,
          and consecutive samples attempt to provide broad class coverage.
        Default is `'class_based'`.

    Returns
    -------
    samples : ndarray
        A 2D NumPy array of shape `(n_batches, batch_size)`, where `n_batches` is the total
        number of batches. Each row contains indices representing a batch of samples.

    Raises
    ------
    AssertionError
        If `n_classes` is not a multiple of `batch_size`.

    Notes
    -----
    - For the `'random'` method, indices are shuffled globally before grouping into batches.
    - For the `'class_based'` method, samples are shuffled within each class, and batches
      are generated such that each group of `batch_size` indices maintains balance across classes.

    Examples
    --------
    Generate indices with class-based sampling:
    >>> labels_shape = (52, 100)  # 52 classes, 100 samples per class
    >>> batch_size = 4
    >>> samples = generate_indices(labels_shape, batch_size, method='class_based')
    >>> samples.shape
    (1300, 4)  # Total batches: (52*100) / 4 = 1300

    Generate indices with random sampling:
    >>> samples = generate_indices(labels_shape, batch_size, method='random')
    >>> samples.shape
    (1300, 4)
    """
    n_classes, n_sample_per_class = labels_shape
    n_groups_of_classes = n_classes // batch_size
    assert n_classes // batch_size == n_classes / batch_size, \
        'n_classes must be a product of batch_size'
    total_points = n_classes * n_sample_per_class
    n_samples = total_points // batch_size
    indices = np.arange(total_points)
    
    if method == 'random':
        np.random.shuffle(indices)
        samples = indices.reshape((n_samples // batch_size, batch_size))
    elif method == 'class_based':
        print('class based shuffling of data')
        indices = indices.reshape(labels_shape)
        for inds_cnt , inds in enumerate(indices):
            np.random.shuffle(inds)
            indices[inds_cnt] = inds.copy()
        indices = indices.swapaxes(0, 1)
        samples = np.zeros((n_samples, batch_size), dtype=int)
        
        for sample_cnt in range(len(indices)):
            bunch_of_samples = indices[sample_cnt]
            np.random.shuffle(bunch_of_samples)
            for gcnt in range(n_groups_of_classes):
                if gcnt == 0:
                    samples[sample_cnt*n_groups_of_classes] = \
                        bunch_of_samples[:batch_size]
                elif gcnt == n_groups_of_classes - 1:
                    samples[sample_cnt*n_groups_of_classes + gcnt] = \
                        bunch_of_samples[-batch_size:]
                else:
                    samples[sample_cnt*n_groups_of_classes + gcnt] = \
                        bunch_of_samples[gcnt * batch_size: (gcnt + 1) * batch_size]
        
    return samples


def segmented_detector_maker(image_length: int,
                             rings_radii_ranges: list,
                             rings_num_segments: list,
                             centre: tuple = None):
    """
    Create a segmented annular detector mask (generalized multi-ring design).

    The detector is divided radially and angularly:
      - The radial boundaries are defined by `radii`.
      - Each ring (region between successive radii) is split into a given
        number of angular segments.

    This generalizes detectors such as PANTHER by allowing arbitrary
    radii and segment counts per ring.

    Parameters
    ----------
    length : int
        Linear size (in pixels) of the square diffraction pattern or detector.
    radii : list of float
        Sorted list of outer radii (in pixels) for each ring.
        Example: [32, 64, 96] defines four rings with boundaries
        [0–32], [32–64], [64–96], [96–∞].
    num_segments : list of int
        List of the same length as `radii`, specifying how many
        angular segments each ring should be divided into.
        Example: [4, 8, 12, 16].
    centre : tuple of float, optional
        (row, column) coordinates of the detector center.
        If None, defaults to the image midpoint.

    Returns
    -------
    detector_masks : np.ndarray
        Array of shape (total_segments, length, length) containing binary masks.
        Each plane corresponds to one segment of the detector.
        Masks are 1 where the segment covers and 0 elsewhere.

    Example
    -------
    >>> det = segmented_detector_maker(
    ...     length=128,
    ...     radii=[32, 64, 96],
    ...     num_segments=[4, 8, 12, 16]
    ... )
    >>> det.shape
    (4 + 8 + 12 + 16, 128, 128)
    >>> np.sum(det, axis=0).max()
    1.0  # Each pixel belongs to exactly one segment
    """
    # --- Setup
    
    try:
        if rings_num_segments == int(rings_num_segments):
            rings_num_segments = np.array([rings_num_segments]*len(rings_radii_ranges))
    except: pass
    
    if len(rings_radii_ranges) == len(rings_num_segments) - 1:
        rings_radii_ranges.append([rings_radii_ranges[-1][1], np.inf])


    n_rings = len(rings_radii_ranges)
    total_segments = sum(rings_num_segments)
    masks = np.zeros((total_segments, image_length, image_length))

    seg_idx = 0
    for ring_idx in range(n_rings):
        n_seg = rings_num_segments[ring_idx]
        in_r = rings_radii_ranges[ring_idx][0]
        out_r = rings_radii_ranges[ring_idx][1]
        # Angular division for this ring
        for s in range(n_seg):
            start_angle = 2 * np.pi * s / n_seg
            finish_angle = 2 * np.pi * (s + 1) / n_seg

            masks[seg_idx] = mcemtools.annular_mask(
                (image_length, image_length),
                centre=centre,
                outer_radius=out_r,
                inner_radius=in_r,
                start_angle=start_angle,
                finish_angle=finish_angle,
            )
            seg_idx += 1
    masks[:, masks.sum(0) > 0] = masks[:, masks.sum(0) > 0] / masks.sum(0)[None, masks.sum(0) > 0]
    return masks

# def test_segmented_detector_maker():
#     det = segmented_detector_maker(128, [[0, 16], [16, 32]], [4, 4, 4])
#     printv(det)
#     det_labels = (det * np.arange(len(det))[:, None, None]).sum(0)
#     _ = plt_imshow_subplots(np.concatenate([det, det_labels[None]], axis=0))

def panther_maker(length, bf_radius, centre=None, RING_0_to_1_RATIO=0.53125):
    """
    Construct a PANTHER-style segmented detector mask for 4D-STEM analysis.

    The PANTHER detector concept divides the bright-field (BF) disk into 
    three concentric annular rings, each further subdivided into four 
    quadrants (12 total segments). This function generates corresponding
    binary masks for each segment in reciprocal-space coordinates.

    The generated masks can be used to simulate or analyze detector
    responses by integrating the diffraction pattern intensity within 
    each segment.

    Parameters
    ----------
    length : int
        Linear size (in pixels) of the square diffraction pattern or detector.
    bf_radius : float
        Radius (in pixels) of the bright-field (BF) disk.
    centre : tuple of float, optional
        (row, column) coordinates of the disk center. If None, the center
        is assumed to be at the image midpoint.
    RING_0_to_1_RATIO : float, optional
        Fractional radius defining the inner ring boundary relative to the
        bright-field radius (default = 0.53125).

    Returns
    -------
    det_resp_new : np.ndarray
        Array of shape (12, length, length) containing the PANTHER detector masks.
        Each plane `det_resp_new[i]` corresponds to one detector segment.
        Masks are normalized such that the sum across all 12 segments equals 1.

    Notes
    -----
    - The detector is built from three concentric rings:
        1. Inner disk to `RING_0_to_1_RATIO * bf_radius`
        2. `RING_0_to_1_RATIO * bf_radius` to `bf_radius`
        3. `bf_radius` to the outer edge of the array
      Each ring is split into 4 quadrants (π/2 angular width).
    
    - The initial (0–11) channel ordering is rearranged to match the 
      PANTHER geometric channel layout.

    - The function depends on `mcemtools.annular_mask`, which should return 
      a binary mask for a given annular sector.

    Example
    -------
    >>> panther = panther_maker(length=256, bf_radius=80)
    >>> panther.shape
    (12, 256, 256)
    >>> np.sum(panther, axis=0).max()
    1.0  # Each pixel belongs to exactly one detector segment
    """
    mask_ring = np.zeros((3, 4, length, length))
    for cnt in range(4):
        mask_ring[0, cnt] = mcemtools.annular_mask(
            (length, length),
            centre=centre,
            radius=RING_0_to_1_RATIO * bf_radius,
            start_angle=cnt * np.pi/2,
            finish_angle=(cnt + 1) * np.pi/2,
        )
        mask_ring[1, cnt] = mcemtools.annular_mask(
            (length, length),
            centre=centre,
            in_radius=RING_0_to_1_RATIO * bf_radius,
            radius=bf_radius,
            start_angle=cnt * np.pi/2,
            finish_angle=(cnt + 1) * np.pi/2,
        )
        mask_ring[2, cnt] = mcemtools.annular_mask(
            (length, length),
            centre=centre,
            in_radius=bf_radius,
            start_angle=cnt * np.pi/2,
            finish_angle=(cnt + 1) * np.pi/2,
        )

    det_resp = mask_ring.reshape(-1, length, length)

    det_resp_new = 0 * det_resp.copy() - 1
    current_ch = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    new_ch = [0, 9, 6, 3, 1, 10, 7, 4, 2, 11, 8, 5]
    for ch, new_ch in zip(current_ch, new_ch):
        det_resp_new[new_ch] = det_resp[ch].copy()

    det_resp_new = det_resp_new / det_resp_new.sum(0)
    if (det_resp_new.astype('int') == det_resp_new).all():
        det_resp_new = det_resp_new.astype('int')

    return det_resp_new

# def test_panther_maker():
#     panth_response = panther_maker(length = 96, bf_radius = 26, centre = (48, 48)).astype('float32')
#     panth_response /= panth_response.sum(0)
#     plt_imshow(panth_response.sum(0))
#     print(np.unique(panth_response.ravel()))
#     plt_imshow_subplots(panth_response, frame_shape = (4, 3))
#     plt_imshow((panth_response*np.arange(12)[:, None, None]).sum(0), figsize = (18, 18), show_values=True)

#     print(f'panth_response: {np.unique(panth_response.ravel())}')


def apply_detresp_memory_efficient(exp_data, detector_response, tile_x=16, tile_y=16, verbose = True):
    """
    exp_data: (n_x, n_y, n_r, n_c)
    detector_response: (n_ch, n_r, n_c)
    returns: (n_x, n_y, n_ch)
    """
    n_x, n_y, _, _ = exp_data.shape
    n_ch = detector_response.shape[0]
    exp_data_ch = np.empty((n_x, n_y, n_ch), dtype=exp_data.dtype)
    if verbose:
        pbar = printprogress(tile_x * tile_y, 
            title = f'memory efficient detector response for {tile_x * tile_y} tiles')
    for ix in range(0, n_x, tile_x):
        ix_end = min(ix + tile_x, n_x)
        for iy in range(0, n_y, tile_y):
            iy_end = min(iy + tile_y, n_y)

            sub = exp_data[ix:ix_end, iy:iy_end]

            sub_ch = (sub[:, :, None, :, :] * detector_response[None, None, :, :, :]).sum(axis=(-1, -2))

            exp_data_ch[ix:ix_end, iy:iy_end] = sub_ch
            if verbose: pbar()
    if verbose: del pbar
    return exp_data_ch

def unsplit_electrons(
    exp_data,
    first_electron_peak_ADU=591.0,
    threshold=0.75,
    kernel=None,
):
    """
    Recombine split electron events in pixelated detector data.

    When a single electron hits a pixelated detector, its charge often spreads
    over adjacent pixels. This function estimates the central pixel associated
    with each electron event and suppresses the surrounding split signals.

    Parameters
    ----------
    exp_data : np.ndarray
        Input detector data, typically of shape (n_x, n_y, n_r, n_c). 
        The function will convolve each frame independently along the last two axes.

    first_electron_peak_ADU : float, default=591.0
        Expected signal amplitude (in ADU) of a single-electron peak.
        Used to set the intensity threshold.

    threshold : float, default=0.75
        Fraction of `first_electron_peak_ADU` used as the cutoff for identifying
        electron event centers. Pixels above this threshold (after convolution)
        are kept; others are suppressed.

    kernel : np.ndarray or None, optional
        Convolution kernel used to merge split charges. If `None`, a default
        3×3 kernel is used:
            [[1, 1, 1],
             [1, 2, 1],
             [1, 1, 1]] / 2
        (expanded to 4D for compatibility with input dimensions).

    Returns
    -------
    np.ndarray
        Processed array of the same shape as `exp_data`, where pixels not
        belonging to identified electron events have been suppressed.
    """
    from scipy.ndimage import convolve, maximum_filter

    if kernel is None:
        kernel = np.array([[1, 1, 1],
                           [1, 2, 1],
                           [1, 1, 1]], dtype=np.float32)[None, None, :, :] / 2

    out = convolve(exp_data, kernel, mode="constant", cval=0)
    mask = out > first_electron_peak_ADU * threshold
    mask_expanded = maximum_filter(mask.astype(np.int32), size=(1, 1, 3, 3))
    exp_data_filtered = exp_data * mask_expanded

    return exp_data_filtered