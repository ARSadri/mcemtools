import re
import os
import numpy as np
from .masking import image_by_windows
import mcemtools
from lognflow import printprogress
import scipy
from itertools import product

def channel_to_image(data_ch, det_geo):
    img = np.zeros(det_geo.shape)
    for cnt, lbl in enumerate(np.unique(det_geo.ravel())[1:]):
        img[det_geo == lbl] = data_ch[cnt]
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

def load_raw(filename, scanSize: tuple[int, int], detSize: tuple[int, int]):
    dt = np.dtype([('data',  f'({detSize[0]},{detSize[1]})single'),
                   ('footer',f'{scanSize[0]}single')])
    data = np.fromfile(file=filename,dtype=dt)["data"].reshape(scanSize+detSize)
    return data

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
    def __init__(self, channel_based_data, detector_geometry):
        self.channel_based_data = channel_based_data
        self.detector_geometry = detector_geometry
        self.n_ch, self.n_x, self.n_y = channel_based_data.shape
        self.n_r, self.n_c = detector_geometry.shape
        self.shape = (self.n_x, self.n_y, self.n_r, self.n_c)
        self.dtype = self.channel_based_data.dtype
        
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
                    mask = self.detector_geometry == segcnt + 1
                    
                    # Apply mask and slice
                    data_slice = self.channel_based_data[segcnt, row_index, col_index]
                    cbed_slices[:, :, mask] += np.tile(
                        np.expand_dims(data_slice, -1), (1, 1, mask.sum()))
                    
                    ETA = pbar()
                    if (ETA > 120) & (pbar.in_print_function is None):
                        pbar = printprogress(self.n_ch - 1)
                
                return cbed_slices
            
            elif isinstance(row_index, int) and isinstance(col_index, int):
                # Handle single integer index
                row, col = row_index, col_index
                cbed = np.zeros((self.n_r, self.n_c))
                
                for segcnt in range(self.n_ch):
                    mask = (self.detector_geometry == segcnt)
                    cbed[mask] = self.channel_based_data[segcnt, row, col]
                
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
        
        w_BF = np.zeros(self.detector_geometry.max())
        w_DF = np.zeros(self.detector_geometry.max())
        for lblcnt, lbl in enumerate(np.unique(self.detector_geometry)[1:]):
            w_BF[lblcnt] = ((self.detector_geometry == lbl) & (mask2d == 1)).sum()\
                                               / (self.detector_geometry == lbl).sum()
            w_DF[lblcnt] = ((self.detector_geometry == lbl) & (mask2d == 0)).sum()\
                                               / (self.detector_geometry == lbl).sum()
    
        return w_BF, w_DF, mask2d
    
    def get_stat(self, weights = None, normalize_for_com = True):
        data_per_ch, det_geo = self.channel_based_data , self.detector_geometry
        if weights is None:
            weights = np.ones(self.n_ch)

        stem = (data_per_ch * weights[
            :, np.newaxis, np.newaxis]).sum(0) / weights.sum()
        
        pacbed = np.zeros(det_geo.shape)
        for cnt, label in enumerate(np.unique(det_geo.ravel())[1:]):
            pacbed[det_geo == label] += data_per_ch[cnt].mean() * weights[cnt]
        pacbed_com_x, pacbed_com_y = scipy.ndimage.center_of_mass(pacbed)

        data_per_ch = data_per_ch.copy()
        if normalize_for_com:
            data_per_ch_sum = np.expand_dims(data_per_ch.sum(0), 0)
            data_per_ch_sum = np.tile(data_per_ch_sum, (data_per_ch.shape[0], 1, 1))
            data_per_ch[data_per_ch_sum != 0] /= data_per_ch_sum[data_per_ch_sum != 0]
            data_per_ch[data_per_ch_sum == 0] = 0

        cent_x, cent_y = det_geo.shape[0]//2, det_geo.shape[1]//2
        com_x_ch = np.zeros(data_per_ch.shape)
        com_y_ch = np.zeros(data_per_ch.shape)
        for cnt, label in enumerate(np.unique(det_geo.ravel())[1:]):
            mask_com_x, mask_com_y = scipy.ndimage.center_of_mass(det_geo == label)
            com_x_ch[cnt] = data_per_ch[cnt]*(mask_com_x - cent_x)
            com_y_ch[cnt] = data_per_ch[cnt]*(mask_com_y - cent_y)
        com_x = (com_x_ch * weights[:, np.newaxis, np.newaxis]).sum(0) / weights.sum()
        com_y = (com_y_ch * weights[:, np.newaxis, np.newaxis]).sum(0) / weights.sum()

        return stem, pacbed, com_x, com_y, pacbed_com_x, pacbed_com_y
    
    def filtered_by_kernel(self, coords, win_side, weights):
    
        coords += win_side
        coords_max = int(coords.max() + 1)
        kernel = np.zeros((coords_max, coords_max))
        for win_cnt_i, win_cnt_j in product(range(win_side), range(win_side)):
            kernel[coords[:, 0] - win_cnt_i, coords[:, 1] - win_cnt_j] = weights

        _, _, com_x, com_y, _, _ = self.get_stat()

        # com_x = -scipy.ndimage.rotate(com_x, -90)
        # com_y = -scipy.ndimage.rotate(com_y, -90)

        filtered_com_x = scipy.signal.convolve2d(com_x, kernel)
        filtered_com_y = scipy.signal.convolve2d(com_y, kernel)

        filtered_com_x = filtered_com_x[coords_max:-coords_max, coords_max:-coords_max]
        filtered_com_y = filtered_com_y[coords_max:-coords_max, coords_max:-coords_max]

        return filtered_com_x, filtered_com_y, kernel
    
def apply_segment_sums(img, det_geo, return_by_channle = False):
    """
    Replace values in a multi-dimensional image based on a segmented labeled image.
    
    Parameters:
    ----------
    img : np.ndarray
        The multi-dimensional image array with shape (n_x, n_y, n_r, n_c),
        where each (n_r, n_c) slice is a single image.
    det_geo : np.ndarray
        The labeled segmentation image with the same shape as img[-2:].
        The area around the detector should be set to 0, and each segment
        should be labeled with a channel number from 1 to 12.

    Returns:
    -------
    np.ndarray
        The modified image with each segment's values replaced by their sums.

    Raises:
    ------
    ValueError
        If det_geo does not match the last two dimensions of img.
    
    Notes:
    ------
    This function sums the values in each segment of img, defined by det_geo.
    Each segment (where det_geo == i) is replaced by the segment's sum in the
    modified img.
    """
    # Validate that det_geo matches the last two dimensions of img
    if det_geo.shape != img.shape[-2:]:
        raise ValueError("The shape of det_geo must match the last two dimensions of img.")

    # Copy img to avoid modifying the original image
    
    if return_by_channle:
        data_by_ch = np.zeros(
            (img.shape[0], img.shape[1], len(np.unique(det_geo)) - 1), 
            dtype = img.dtype)
    else:
        modified_img = img.copy()
    # Iterate through each image slice (n_r, n_c)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # For each unique label, sum values in img and replace within the segment
            for label in np.unique(det_geo):
                if label == 0:
                    continue  # Skip the area around the detector

                # Calculate the sum of values within the current segment
                segment_sum = img[i, j][det_geo == label].sum()
                
                # Assign the sum to the segment in the modified image
                
                if return_by_channle:
                    data_by_ch[i, j][label - 1] = segment_sum
                else:
                    modified_img[i, j][det_geo == label] = segment_sum

    if return_by_channle:
        return data_by_ch
    else:
        return modified_img
