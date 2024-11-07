import matplotlib.pyplot as plt
from matplotlib.widgets import RangeSlider, Slider
import numpy as np

def mask2D_to_4D(mask2D, data4D_shape):
    n_x, n_y, n_r, n_c = data4D_shape
    assert len(mask2D.shape) == 2, 'mask should be 2d'
    assert mask2D.shape[0] == n_r, 'mask should have same shape as patterns'
    assert mask2D.shape[1] == n_c, 'mask should have same shape as patterns'
        
    _mask4D = np.array([np.array([mask2D.copy()])])
    _mask4D = np.tile(_mask4D, (n_x, n_y, 1, 1))
    return _mask4D

def annular_mask(image_shape : tuple, 
                 centre:tuple = None, radius:float=None, in_radius:float=None):
    """make a circle bianry pattern in a given window
    This simple function makes a circle filled with ones for where the circle is
    in a window and leaves the rest of the elements to remain zero.
    Parameters
    ----------
        :param image_shape:
            a tuple of the shape of the image
        :param centre: 
            tuple of two float scalars
            Intensity difference threshold.
        :param radius :
            float radius of the circle, if in_radius is None, inside this 
            radius is filledup with 1.
        :param in_radius :
            float radius of the circle inside where it is not masked. If given
            the annular ring between in_radius and radius will be 1.
    Returns
    -------
        : np.ndarray of type uint8
            An image of size h x w where all elements that are closer 
            to the origin of a circle with centre at centre and radius 
            radius are one and the rest are zero. We use equal or greater 
            than for both radius and in_radius.
    """
    n_r, n_c = image_shape
    if centre is None: # use the middle of the image
        # centre = (int(n_r/2), int(n_c/2))
        centre = [n_r/2, n_c/2]

    Y, X = np.ogrid[:n_c, :n_r]
    
    if n_r/2 == n_r//2:
        Y = Y + 0.49
    else:
        centre[1] = centre[1] - 0.5
    if n_c/2 == n_c//2:
        X = X + 0.49
    else:
        centre[0] = centre[0] - 0.5
    
    if radius is None: 
        # use the smallest distance between the centre and image walls
        radius = np.floor(np.minimum(*centre))

    dist_from_centre = np.sqrt((X - centre[0])**2 + (Y-centre[1])**2)

    mask = dist_from_centre <= radius
    
    if(in_radius is not None):
        mask *= in_radius <= dist_from_centre

    return mask.astype('uint8') 

def is_torch(data):
    try:
        _ = data.is_cuda
        return True
    except AttributeError:
        return False

def crop_or_pad(data, new_shape, padding_value = 0, shift = None):
    """shape data to have the new shape new_shape
    Parameters
    ----------
        :param data
        :param new_shape
        :param padding_value
            the padded areas fill value
    Returns
    -------
        : np.ndarray of type data.dtype of shape new_shape
            If a dimension of new_shape is smaller than a, a will be cut, 
            if bigger, a will be put in the middle of padded zeros.
    """
    data_shape = data.shape
    data_is_torch = is_torch(data)
    assert len(data_shape) == len(new_shape), \
        'put np.ndarray a in b, the length of their shapes should be the same.' \
        f'currently, data shape is {data_shape} and new shape is {new_shape}'

    if shift is not None:
        assert len(shift) == len(data_shape)
    else:
        shift = np.zeros(len(data_shape), dtype='int')

    for dim in range(len(data_shape)):
        if data_shape[dim] != new_shape[dim]:
            if data_is_torch:
                data = data.transpose(0, dim)
            else:
                data = data.swapaxes(0, dim)
                
            if data_shape[dim] > new_shape[dim]:
                start = int((data_shape[dim] - new_shape[dim])/2) + shift[dim]
                finish = int((data_shape[dim] + new_shape[dim])/2) + shift[dim]
                data = data[start : finish]
            elif data_shape[dim] < new_shape[dim]:
                pad_left = -int((data_shape[dim] - new_shape[dim])/2)  - shift[dim]
                pad_right = int(np.ceil((new_shape[dim] - data_shape[dim])/2)) + shift[dim]
                if data_is_torch:
                    pad_left_tensor = padding_value + torch.zeros((pad_left,) + data.shape[1:], dtype=data.dtype, device=data.device)
                    pad_right_tensor = padding_value + torch.zeros((pad_right,) + data.shape[1:], dtype=data.dtype, device=data.device)
                    data = torch.cat((pad_left_tensor, data, pad_right_tensor), dim=0)
                else:
                    data = np.vstack(
                        (padding_value + np.zeros(((pad_left, ) + data.shape[1:]),
                                          dtype=data.dtype),
                         data,
                         padding_value + np.zeros(((pad_right, ) + data.shape[1:]),
                                          dtype=data.dtype)))
            if data_is_torch:
                data = data.transpose(0, dim)
            else:
                data = data.swapaxes(0, dim)
    return data

class image_by_windows:
    def __init__(self, 
                 img_shape: tuple[int, int], 
                 win_shape: tuple[int, int],
                 skip: tuple[int, int] = (1, 1),
                 method = 'linear'):
        """image by windows
        
            I am using OOP here because the user pretty much always wants to
            transform results back to the original shape. It is different
            from typical transforms, where the processing ends at the other
            space.
        
            Parameters
            ----------
            :param img_shape:
                pass your_data.shape. First two dimensions should be for the
                image to be cropped.
            :param win_shape:
                the cropping windows shape
            :param skip:
                The skipping length of windows
            :param method:
                default is linear, it means that if it cannot preserve the skip
                it will not, but the grid will be spread evenly among windows.
                If you wish to keep the skip exact, choose fixed. If the size
                of the image is not dividable by the skip, it will have to
                change the location of last window such that the entire image
                is covered. This emans that the location of the grid will be 
                moved to the left. 
        """
        self.img_shape = img_shape
        self.win_shape = win_shape
        self.skip      = skip
        
        n_r, n_c = self.img_shape[:2]
        skip_r, skip_c = self.skip
        
        assert win_shape[0]<= n_r, 'win must be smaller than the image'
        assert win_shape[1]<= n_c, 'win must be smaller than the image'

        if(method == 'fixed'):
            
            rows = np.arange(0, n_r - win_shape[0] + 1, skip_r)
            clms = np.arange(0, n_c - win_shape[1] + 1, skip_c)
            warning = False
            if rows[-1] < n_r - win_shape[0]:
                rows = np.concatenate((rows, np.array([n_r - win_shape[0]])))
                warning = True
            if clms[-1] < n_c - win_shape[1]:
                clms = np.concatenate((clms, np.array([n_c - win_shape[1]])))
                warning = True
            if warning:
                print('WARNING by image_by_windows.init: when using fixed, '
                      'you may wish to make sure img_shape is divisible by skip. '
                      'With the current setting, you may have artifacts.')
        if(method == 'linear'):
            rows = np.linspace(
                0, n_r - win_shape[0],n_r // skip_r, dtype = 'int')
            rows = np.unique(rows)
            clms = np.linspace(
                0, n_c - win_shape[1],n_r // skip_c, dtype = 'int')
            clms = np.unique(clms)
        self.grid_clms, self.grid_rows = np.meshgrid(clms, rows)
        self.grid_shape = (len(rows), len(clms))
        self.grid = np.array([self.grid_rows.ravel(), self.grid_clms.ravel()]).T
        self.n_pts = self.grid.shape[0]
        
    def image2views(self, img):
        all_other_dims = ()
        if (len(img.shape)>2):
            all_other_dims = img.shape[2:]
        try: #numpy
            img_dtype = img.dtype
            views = np.zeros(
                (self.grid.shape[0], self.win_shape[0], self.win_shape[1]
                 ) + all_other_dims,
                dtype = img_dtype)
            for gcnt, grc in enumerate(self.grid):
                gr, gc = grc
                views[gcnt] = img[
                    gr:gr + self.win_shape[0], gc:gc + self.win_shape[1]].copy()
        except:#torch or others
            views = []
            for gcnt, grc in enumerate(self.grid):
                gr, gc = grc
                views.append(
                    img[gr:gr + self.win_shape[0], gc:gc + self.win_shape[1]])
        return views
    
    def views2image(self, views, include_inds = None, method = 'linear',
                    win_shape = None):
        if win_shape is None:
            win_shape = self.win_shape

        win_start = ((self.win_shape[0] - win_shape[0])//2,
                     (self.win_shape[1] - win_shape[1])//2)
        
        img_shape = (self.img_shape[0], self.img_shape[1])
        if (len(views.shape) == 5):
            img_shape += views.shape[3:]

        assert len(views.shape) != 2, 'views2image: views cannot be 2D yet!'

        if include_inds is None:
            grid = self.grid.copy()
        else:
            grid = self.grid[include_inds].copy()

        if(method == 'linear'):
            img = np.zeros(img_shape, dtype = views.dtype)
            visited = np.zeros(img_shape, dtype = views.dtype)
            for gcnt, grc in enumerate(grid):
                gr, gc = grc
                img[gr:gr + win_shape[0], 
                    gc:gc + win_shape[1]] += views[gcnt]
                visited[gr:gr + win_shape[0], 
                        gc:gc + win_shape[1]] += 1
            img[visited>0] = img[visited>0] / visited[visited>0]
        elif(method == 'fixed'):
            img = np.zeros(img_shape, dtype = views.dtype)
            for gcnt, grc in enumerate(grid):
                gr, gc = grc
                img[gr + win_start[0]:gr + win_start[0] + win_shape[0], 
                    gc + win_start[1]:gc + win_start[1] + win_shape[1]] = \
                        views[gcnt]
        else:
            img = np.zeros(
                (win_shape[0]*win_shape[1],) + img_shape, views.dtype)
            visited = np.zeros((win_shape[0] * win_shape[1], 
                                img_shape[0], img_shape[1]), dtype='int')
            for gcnt, grc in enumerate(grid):
                gr, gc = grc
                level2use = visited[:, gr:gr + win_shape[0], 
                                       gc:gc + win_shape[1]].max(2).max(1)
                level = np.where(level2use == 0)[0][0]
                img[level, gr:gr + win_shape[0],
                           gc:gc + win_shape[1]] += views[gcnt]
                visited[level, 
                    gr:gr + win_shape[0], gc:gc + win_shape[1]] = 1
            if(method == 'max'):
                img = img.max(0).squeeze()
            if(method == 'min'):
                img = img.min(0).squeeze()
        return img
    
    def __len__(self):
        return self.n_pts
    
def test_image_by_windows():
    img = np.ones((25, 25))
    imbywin = image_by_windows(img_shape = (img.shape[0], img.shape[1]), 
                               win_shape = (8, 8),
                               skip = (8, 8),
                               method = 'fixed')
    img_windowed = imbywin.image2views(img)
    print(f'img_windowed shape: {img_windowed.shape}')
    img_recon = imbywin.views2image(img_windowed, method = 'fixed')
    print(f'img_recon shape: {img_recon.shape}')
    from lognflow import plt_imshow
    plt_imshow(img_recon); plt.show(); exit()

class markimage:
    def __init__(self, 
                 in_image, 
                 mark_shape = 'circle',
                 figsize=(10, 5),
                 shape_color = "k",
                 **kwargs_for_imshow):
        self.mark_shape = mark_shape
        self.fig, axs = plt.subplots(1, 2, figsize=figsize)
        self.fig.subplots_adjust(bottom=0.4)
        # cm = plt.colormaps["Spectral"]
        self.im = axs[0].imshow(in_image, **kwargs_for_imshow)
        cm = self.im.get_cmap()
        _, bins, patches = axs[1].hist(in_image.flatten(), bins='auto')
        bin_centres = 0.5 * (bins[:-1] + bins[1:])
        # scale values to interval [0,1]
        col = bin_centres - min(bin_centres)
        col /= max(col)
        for c, p in zip(col, patches):
            plt.setp(p, 'facecolor', cm(c))
        axs[1].set_title('Histogram of pixel intensities')
        
        # Create the RangeSlider
        slider_ax = self.fig.add_axes([0.25, 0.25, 0.6, 0.03])
        slider = RangeSlider(slider_ax, "Threshold", 
                             in_image.min(), 
                             in_image.max(), 
                             valinit=(in_image.min(), in_image.max()))
        # Create the Vertical lines on the histogram
        self.lower_limit_line = axs[1].axvline(slider.val[0], color='k')
        self.upper_limit_line = axs[1].axvline(slider.val[1], color='k')
        slider.on_changed(self.update)
        
        if(self.mark_shape == 'circle'):
            cx, cy = in_image.shape
            cx = cx / 2
            cy = cy / 2
            circle_radius = cx if cx < cy else cy
            sl1 = plt.axes([0.25, 0.15, 0.6, 0.03])
            sl2 = plt.axes([0.25, 0.1,  0.6, 0.03])
            sl3 = plt.axes([0.25, 0.05, 0.6, 0.03])
            self.markshape = plt.Circle(
                (cy,cx), circle_radius, ec=shape_color, fc = 'None')
            axs[0].add_patch(self.markshape)
            self.slider_r = Slider(sl1, 
                'radius', 0.0, self.im.get_array().shape[0]/2, 
                valinit = circle_radius)
            self.slider_cx = Slider(sl2,
                'centre_x', 0.0, self.im.get_array().shape[0], valinit = cx)
            self.slider_cy = Slider(sl3, 
                'centre_y', 0.0, self.im.get_array().shape[1], valinit = cy)
            self.slider_r.on_changed(self.update2)
            self.slider_cx.on_changed(self.update2)
            self.slider_cy.on_changed(self.update2)

        if(self.mark_shape == 'rectangle'):
            bot_right_r, bot_right_c = in_image.shape
            top_left_r = bot_right_r * 0.1
            top_left_c = bot_right_c * 0.1
            bot_right_r = bot_right_r * 0.9
            bot_right_c = bot_right_c * 0.9

            s_top_left_r = plt.axes([0.25, 0.2, 0.6, 0.03])
            s_top_left_c = plt.axes([0.25, 0.15, 0.6, 0.03])
            s_bot_right_r = plt.axes([0.25, 0.1, 0.6, 0.03])
            s_bot_right_c = plt.axes([0.25, 0.05, 0.6, 0.03])

            self.markshape = plt.Rectangle(
                (top_left_r, top_left_c), 
                bot_right_r - top_left_r, 
                bot_right_c - top_left_c,
                ec=shape_color, fc = 'None')
            axs[0].add_patch(self.markshape)
            self.slider_top_left_r = Slider(
                s_top_left_r, 'top_left_x', 0.0, 
                self.im.get_array().shape[0]/2, valinit = top_left_r)
            self.slider_top_left_c = Slider(
                s_top_left_c, 'top_left_y', 0.0, 
                self.im.get_array().shape[0], valinit = top_left_c)
            self.slider_bot_right_r = Slider(
                s_bot_right_r, 'bot_right_x', 0.0, 
                self.im.get_array().shape[1], valinit = bot_right_r)
            self.slider_s_bot_right_c = Slider(
                s_bot_right_c, 's_bot_right_y', 0.0, 
                self.im.get_array().shape[1], valinit = bot_right_c)

            self.slider_top_left_r.on_changed(self.update2)
            self.slider_top_left_c.on_changed(self.update2)
            self.slider_bot_right_r.on_changed(self.update2)
            self.slider_s_bot_right_c.on_changed(self.update2)
        
        plt.show()
    
    def update(self, val):
        # The val passed to a callback by the RangeSlider will
        # be a tuple of (min, max)
    
        # Update the image's colormap
        self.im.norm.vmin = val[0]
        self.im.norm.vmax = val[1]
    
        # Update the position of the vertical lines
        self.lower_limit_line.set_xdata([val[0], val[0]])
        self.upper_limit_line.set_xdata([val[1], val[1]])
    
        # Redraw the figure to ensure it updates
        self.fig.canvas.draw_idle()

    def update2(self, val):
        if(self.mark_shape == 'circle'):
            r = self.slider_r.val
            cx = self.slider_cx.val
            cy  = self.slider_cy.val
            self.markshape.set_center((cy, cx))
            self.markshape.set_radius(r)
            
        if(self.mark_shape == 'rectangle'):
            self.slider_top_left_r.val
            self.slider_top_left_c.val
            self.slider_bot_right_r.val
            self.slider_s_bot_right_c.val

            self.markshape.set_xy = ((
                self.slider_top_left_r.val,
                self.slider_top_left_c.val))
            self.markshape.set_width(
                self.slider_bot_right_r.val - self.slider_top_left_r.val)
            self.markshape.set_height(
                self.slider_s_bot_right_c.val - self.slider_top_left_c.val)

        self.fig.canvas.draw_idle()

def remove_labels(label_map, labels_to_remove):
    if(labels_to_remove.shape[0] > 0):
        label_map_shape = label_map.shape
        label_map = label_map.ravel()
        label_map[np.in1d(label_map, labels_to_remove)] = 0
        label_map = label_map.reshape(label_map_shape)
    return(label_map)

def remove_islands_by_size(
        binImage, min_n_pix = 1, max_n_pix = np.inf, logger = None):    
    import scipy.ndimage
    
    segments_map, n_segments = scipy.ndimage.label(binImage)
    if(logger):
        logger(f'counted {n_segments} segments!')
    segments_labels, n_segments_pix = np.unique(segments_map.ravel(),
                                         return_counts = True)

    labels_to_remove = segments_labels[(n_segments_pix < min_n_pix) |
                                       (n_segments_pix > max_n_pix)]
    if(logger):
        logger(f'counted {labels_to_remove.shape[0]} too small segments!')
    segments_map = remove_labels(segments_map, labels_to_remove)
    segments_map[segments_map > 0] = 1

    if(logger):
        logger('number of remaining segments pixels')
        n_p = n_segments_pix[n_segments_pix > min_n_pix]
        logger(f'{np.sort(n_p)}')
   
    return (segments_map)

if __name__ == '__main__':
    test_image_by_windows()
