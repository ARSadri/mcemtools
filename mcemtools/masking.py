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
            An image of size h x w where all elements that are closer to the origin
            of a circle with centre at centre and radius radius are one and the rest
            are zero. We use equal or greater than for both radius and in_radius.
    """
    n_r, n_c = image_shape
    if centre is None: # use the middle of the image
        centre = (int(n_r/2), int(n_c/2))
    if radius is None: # use the smallest distance between the centre and image walls
        radius = np.minimum(centre[0], centre[1])

    Y, X = np.ogrid[:n_r, :n_c]
    dist_from_centre = np.sqrt((X - centre[0])**2 + (Y-centre[1])**2)

    mask = dist_from_centre <= radius
    
    if(in_radius is not None):
        mask *= in_radius <= dist_from_centre

    return mask.astype('uint8') 

class image_by_windows:
    def __init__(self, 
                 img_shape: tuple[int, int], 
                 win_shape: tuple[int, int],
                 skip: tuple[int, int] = (1, 1)):
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
        """
        self.img_shape = img_shape
        self.win_shape = win_shape
        self.skip = skip
        n_r, n_c = img_shape[:2]
        rows = np.arange(0, n_r - win_shape[0] + 1, skip[0])
        clms = np.arange(0, n_c - win_shape[1] + 1, skip[1])
        
        if rows[-1] < n_r - win_shape[0]:
            rows = np.concatenate(rows, n_r - win_shape[0])
        if clms[-1] > n_c - win_shape[1]:
            clms = np.concatenate(clms, n_c - win_shape[1])
    
        grid_clms, grid_rows = np.meshgrid(clms, rows)
    
        self.grid = np.array([grid_rows.ravel(), grid_clms.ravel()]).T
        self.visited = np.zeros(self.img_shape, dtype='int')
        for grc in self.grid:
            self.visited[grc[0]:grc[0] + self.win_shape[0], 
                         grc[1]:grc[1] + self.win_shape[1]] += 1
        self.n_pts = self.grid.shape[0]
        
    def image2views(self, img):
        all_other_dims = ()
        if (len(self.img_shape)>2):
            all_other_dims = self.img_shape[2:]
            
        viewed = np.zeros(
            (self.grid.shape[0], self.win_shape[0], 
                                 self.win_shape[1]) + all_other_dims,
            dtype = img.dtype)
        for gcnt, grc in enumerate(self.grid):
            gr, gc = grc
            viewed[gcnt] = img[
                gr:gr + self.win_shape[0], gc:gc + self.win_shape[1]]
        return viewed
    
    def views2image(self, viewed, method = 'linear'):
        if(method == 'linear'):
            img = np.zeros(self.img_shape, dtype = 'float')
            for gcnt, grc in enumerate(self.grid):
                gr, gc = grc
                img[gr:gr + self.win_shape[0], gc:gc + self.win_shape[1]] += \
                    viewed[gcnt]
            img[self.visited>0, ...] = \
                img[self.visited>0, ...] / self.visited[self.visited>0]
            img = img.astype(viewed.dtype)
        else:
            img = np.zeros(
                (self.win_shape[0]*self.win_shape[1],) + self.img_shape, 
                viewed.dtype)
            visited = np.zeros(
                (self.win_shape[0] * self.win_shape[1], 
                 self.img_shape[0], self.img_shape[1]), dtype='int')
            for gcnt, grc in enumerate(self.grid):
                gr, gc = grc
                
                level2use = visited[
                    :, gr:gr + self.win_shape[0], 
                       gc:gc + self.win_shape[1]].max(2).max(1)
                level = np.where(level2use == 0)[0][0]
                
                img[level, gr:gr + self.win_shape[0], gc:gc + self.win_shape[1]] += \
                    viewed[gcnt]
                visited[level, 
                    gr:gr + self.win_shape[0], gc:gc + self.win_shape[1]] = 1
            if(method == 'max'):
                img = img.max(0).squeeze()
        return img

class markimage:
    def __init__(self, 
                 in_image, 
                 mark_shape = 'circle',
                 figsize=(10, 5),
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
            self.markshape = plt.Circle((cy,cx), circle_radius, ec="k", fc = 'None')
            axs[0].add_patch(self.markshape)
            self.slider_r = Slider(sl1, 
                'radius', 0.0, self.im.get_array().shape[0]/2, valinit = circle_radius)
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
                ec="k", fc = 'None')
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