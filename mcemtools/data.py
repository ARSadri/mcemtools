import re
import os
import numpy as np
from .masking import image_by_windows

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
        
        self.imbywin = image_by_windows(inimg.shape, win_shape, skip)
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
    
    def update(self, inimg):
        for gpt_cnt in range(self.n_pts):
            a_tile = inimg[
                self.xx[gpt_cnt] - self.len_side // 2 : 
                    self.xx[gpt_cnt] + self.len_side // 2 + 1,
                self.yy[gpt_cnt] - self.len_side // 2 : 
                    self.yy[gpt_cnt] + self.len_side // 2 + 1].copy()
            a_tile = a_tile.reshape(
                self.len_side*self.len_side, self.n_r, self.n_c)
            self.X_in[gpt_cnt] = a_tile[self.mask_range == 1].copy()
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
