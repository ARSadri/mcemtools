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

import numpy as np

def annular_mask(image_shape : tuple,
                 centre:tuple = None, outer_radius:float=None, inner_radius:float=None,
                 start_angle:float = 0, finish_angle:float = 2*np.pi,
                 radius = None, in_radius = None):
    
    if outer_radius is not None:
        radius = outer_radius
    if inner_radius is not None:
        in_radius = inner_radius
    
    n_r, n_c = image_shape
    if centre is None: # use the middle of the image
        centre = [n_r/2, n_c/2]

    X, Y = np.ogrid[:n_r, :n_c]

    # Adjust for pixel centers if image dimensions are even
    if n_r % 2 == 0:
        Y = Y + 0.5
    else:
        centre[1] = centre[1] - 0.5
    if n_c % 2 == 0:
        X = X + 0.5
    else:
        centre[0] = centre[0] - 0.5

    if radius is None:
        # use the smallest distance between the centre and image walls
        if in_radius is None:
            radius = np.floor(np.minimum(*centre))
        else:
            radius = np.inf

    dist_from_centre = np.sqrt((X - centre[0])**2 + (Y-centre[1])**2)

    mask = dist_from_centre <= radius

    if(in_radius is not None):
        mask *= in_radius <= dist_from_centre

    # Calculate angles
    angles = np.arctan2(Y - centre[1], X - centre[0]) # arctan2 handles all quadrants

    # Normalize angles to be within [0, 2*pi)
    angles = (angles + 2 * np.pi) % (2 * np.pi)

    # Apply angle mask
    if start_angle <= finish_angle:
        angle_mask = (angles >= start_angle) * (angles <= finish_angle)
    else:
        # Handles cases where the angle range crosses the 0/2*pi boundary (e.g., 3*pi/2 to pi/2)
        angle_mask = (angles >= start_angle) + (angles <= finish_angle)

    mask *= angle_mask

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
    if data_is_torch:
        import torch
    assert len(data_shape) == len(new_shape), \
        'put np.ndarray a in b, the length of their shapes should be the same.' \
        f'currently, data shape is {data_shape} and new shape is {new_shape}'

    if shift is not None:
        assert len(shift) == len(data_shape)
    else:
        shift = np.zeros(len(data_shape), dtype='int')
    
    for dim in range(len(data_shape)):
        if (new_shape[dim] > 0) & (data_shape[dim] != new_shape[dim]):
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
        grid_locations=0*self.grid.copy()
        grid_locations[:, 0] = self.grid[:, 1].copy()
        grid_locations[:, 1] = (self.grid[:, 0].copy().max() - self.grid[:, 0].copy())
        self.grid_locations_for_subplots = grid_locations.copy()
        
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

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RangeSlider, TextBox
import numpy as np

def extract_pixels_on_patch(image, patch, linewidth=1.0):
    """
    Extract pixels from the image that lie on the boundary of a matplotlib patch.

    Parameters:
        image (np.ndarray): Input image, shape (H, W) or (H, W, C).
        patch (matplotlib.patches.Patch): A matplotlib patch object (e.g., Circle, Rectangle, Ellipse).
        linewidth (float): Width of the boundary to consider in pixels.

    Returns:
        pixels (np.ndarray): Pixel values on the patch edge.
        mask (np.ndarray): Boolean mask of shape (H, W), True for edge pixels.
    """
    H, W = image.shape[:2]

    # Create coordinate grid
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    coords = np.stack([xx.ravel(), yy.ravel()], axis=1)

    # Get the transformed path
    path = patch.get_path().transformed(patch.get_transform())

    # Distance to the path (boundary)
    distances = path.to_polygons()[0]  # Only outer path
    path_poly = Path(distances)

    # Compute signed distance from path
    inside = path_poly.contains_points(coords)
    
    # Erode path to get interior just inside the stroke
    path_in = patch.get_path().transformed(patch.get_transform().frozen())
    shrink_patch = patch  # For true shrinking, you'd need custom geometry ops

    # Recreate inner path with a slightly smaller shape
    shrink_path = patch.get_path().transformed(
        patch.get_transform().frozen().scale(1 - linewidth / patch.get_radius() if hasattr(patch, "get_radius") else 1.0)
    ) if hasattr(patch, "get_radius") else None

    # Approximate: take edge pixels as the difference between two masks
    if hasattr(patch, "get_radius"):
        from skimage.draw import disk
        cy, cx = patch.center
        r_outer = patch.get_radius()
        r_inner = r_outer - linewidth / 2.0
        outer_mask = np.zeros((H, W), dtype=bool)
        inner_mask = np.zeros((H, W), dtype=bool)
        rr, cc = disk((cy, cx), r_outer, shape=(H, W))
        outer_mask[rr, cc] = True
        rr, cc = disk((cy, cx), r_inner, shape=(H, W))
        inner_mask[rr, cc] = True
        edge_mask = outer_mask & ~inner_mask
    else:
        raise NotImplementedError("Only Circle patches are supported in this version.")

    # Extract pixels
    if image.ndim == 2:
        pixels = image[edge_mask]
    else:
        pixels = image[edge_mask, :]

    return pixels, edge_mask

class markimage:
    def __init__(self, 
                 in_image, 
                 mark_shape='circle',
                 figsize=(10, 5),
                 slider_start = 0.1,
                 kwargs_shape=dict(ec='pink', fc='None', linewidth=1),
                 **kwargs_for_imshow):
        kwargs_shape.setdefault('ec', 'pink')
        kwargs_shape.setdefault('fc', 'None')
        kwargs_shape.setdefault('linewidth', 1)

        assert len(in_image.shape) == 2, 'mcemtools.markimage, input must be 2D image'

        self.mark_shape = mark_shape
        self.fig, axs = plt.subplots(1, 3, figsize=figsize)
        self.fig.subplots_adjust(bottom=0.45)
        self.im = axs[0].imshow(in_image, **kwargs_for_imshow)
        cm = self.im.get_cmap()
        self.in_image = in_image
        _, bins, patches = axs[1].hist(in_image.flatten(), bins='auto')
        bin_centres = 0.5 * (bins[:-1] + bins[1:])
        col = (bin_centres - np.min(bin_centres)) / np.ptp(bin_centres)
        for c, p in zip(col, patches):
            plt.setp(p, 'facecolor', cm(c))
        axs[1].set_title('Histogram of pixel intensities')
        self.ax2 = axs[2]
        slider_ax = self.fig.add_axes([slider_start, 0.3, 0.4, 0.03])
        self.slider_thresh = RangeSlider(
            slider_ax, "", in_image.min(), in_image.max(), 
            valinit=(in_image.min(), in_image.max()))
        
        tb_min_ax = self.fig.add_axes([0.83, 0.3, 0.06, 0.03])
        tb_max_ax = self.fig.add_axes([0.93, 0.3, 0.06, 0.03])
        self.tb_min = TextBox(tb_min_ax, 'min', initial=f'{in_image.min():.5f}')
        self.tb_max = TextBox(tb_max_ax, 'max', initial=f'{in_image.max():.5f}')

        self.lower_limit_line = axs[1].axvline(self.slider_thresh.val[0], color='k')
        self.upper_limit_line = axs[1].axvline(self.slider_thresh.val[1], color='k')

        self.slider_thresh.on_changed(self.update)
        self.tb_min.on_submit(self.on_min_thresh)
        self.tb_max.on_submit(self.on_max_thresh)

        if self.mark_shape == 'circle':
            cx, cy = in_image.shape
            cx /= 2
            cy /= 2
            circle_radius = min(cx, cy)

            self.markshape = plt.Circle((cy, cx), circle_radius, **kwargs_shape)
            axs[0].add_patch(self.markshape)

            sl1 = self.fig.add_axes([slider_start, 0.2, 0.5, 0.03])
            sl2 = self.fig.add_axes([slider_start, 0.15, 0.5, 0.03])
            sl3 = self.fig.add_axes([slider_start, 0.1, 0.5, 0.03])

            self.slider_r = Slider(sl1, "", 0.0, cx, valinit=circle_radius)
            self.slider_cx = Slider(sl2, "", 0.0, in_image.shape[0], valinit=cx)
            self.slider_cy = Slider(sl3, "", 0.0, in_image.shape[1], valinit=cy)

            tb_r_ax = self.fig.add_axes([0.87, 0.2, 0.12, 0.03])
            tb_cx_ax = self.fig.add_axes([0.87, 0.15, 0.12, 0.03])
            tb_cy_ax = self.fig.add_axes([0.87, 0.1, 0.12, 0.03])

            self.tb_r = TextBox(tb_r_ax, 'radius', initial=str(circle_radius))
            self.tb_cx = TextBox(tb_cx_ax, 'centre_x', initial=str(cx))
            self.tb_cy = TextBox(tb_cy_ax, 'centre_y', initial=str(cy))

            self.slider_r.on_changed(self.sync_radius)
            self.slider_cx.on_changed(self.sync_cx)
            self.slider_cy.on_changed(self.sync_cy)

            self.tb_r.on_submit(self.set_radius)
            self.tb_cx.on_submit(self.set_cx)
            self.tb_cy.on_submit(self.set_cy)

        if self.mark_shape == 'rectangle':
            h, w = in_image.shape
            tl_r = h * 0.1
            tl_c = w * 0.1
            br_r = h * 0.9
            br_c = w * 0.9

            self.markshape = plt.Rectangle(
                (tl_c, tl_r), br_c - tl_c, br_r - tl_r, **kwargs_shape)
            axs[0].add_patch(self.markshape)

            sliders = {
                'top_left_r': [slider_start, 0.2, tl_r, h],
                'top_left_c': [slider_start, 0.15, tl_c, w],
                'bot_right_r': [slider_start, 0.1, br_r, h],
                'bot_right_c': [slider_start, 0.05, br_c, w],
            }

            for i, (name, (x, y, val, vmax)) in enumerate(sliders.items()):
                setattr(self, f'slider_{name}', Slider(
                    self.fig.add_axes([x, y, 0.5, 0.03]), "", 0.0, vmax, valinit=val))
                setattr(self, f'tb_{name}', TextBox(
                    self.fig.add_axes([0.87, y, 0.12, 0.03]), name, initial=str(val)))
                slider = getattr(self, f'slider_{name}')
                textbox = getattr(self, f'tb_{name}')
                slider.on_changed(self.sync_rect)
                textbox.on_submit(lambda text, s=slider: s.set_val(float(text)))

        plt.show()

    def update(self, val):
        self.im.norm.vmin = val[0]
        self.im.norm.vmax = val[1]
        self.lower_limit_line.set_xdata([val[0], val[0]])
        self.upper_limit_line.set_xdata([val[1], val[1]])
        self.tb_min.set_val(str(val[0]))
        self.tb_max.set_val(str(val[1]))
        self.update2()
        self.fig.canvas.draw_idle()

    def on_min_thresh(self, text):
        try:
            val = float(text)
            self.slider_thresh.set_val((val, self.slider_thresh.val[1]))
        except ValueError:
            pass

    def on_max_thresh(self, text):
        try:
            val = float(text)
            self.slider_thresh.set_val((self.slider_thresh.val[0], val))
        except ValueError:
            pass

    def sync_radius(self, val):
        self.tb_r.set_val(str(val))
        self.update2(val)

    def sync_cx(self, val):
        self.tb_cx.set_val(str(val))
        self.update2(val)

    def sync_cy(self, val):
        self.tb_cy.set_val(str(val))
        self.update2(val)

    def set_radius(self, text):
        try:
            self.slider_r.set_val(float(text))
        except ValueError:
            pass

    def set_cx(self, text):
        try:
            self.slider_cx.set_val(float(text))
        except ValueError:
            pass

    def set_cy(self, text):
        try:
            self.slider_cy.set_val(float(text))
        except ValueError:
            pass

    def sync_rect(self, val):
        for name in ['top_left_r', 'top_left_c', 'bot_right_r', 'bot_right_c']:
            textbox = getattr(self, f'tb_{name}')
            slider = getattr(self, f'slider_{name}')
            textbox.set_val(str(slider.val))
        self.update2(val)

    import numpy as np

    def circle_indices(self, shape, radius, center = None, tol=0.5):
        """
        Return the indices (i,j) of pixels lying on a circle of given radius
        centered at the middle of the image.

        Parameters
        ----------
        shape : tuple
            Image shape (n_rows, n_cols).
        radius : float
            Radius of the circle in pixels.
        tol : float
            Tolerance: pixels are included if their distance from center
            is within [r - tol, r + tol].

        Returns
        -------
        indi, indj : arrays of int
            Row and column indices of pixels on the circle.
        """
        nrows, ncols = shape
        if center is None:
            ic, jc = nrows/2, ncols/2  # center of image
        else:
            ic, jc = center

        y, x = np.ogrid[:nrows, :ncols]
        dist = np.sqrt((y - ic)**2 + (x - jc)**2)

        mask = np.abs(dist - radius) <= tol
        indi, indj = np.nonzero(mask)

        return indi, indj


    def update2(self, val = None):
        if self.mark_shape == 'circle':
            r = self.slider_r.val
            cx = self.slider_cx.val
            cy = self.slider_cy.val
            self.markshape.set_center((cy, cx))
            self.markshape.set_radius(r)

            indsi, indsj = self.circle_indices(self.in_image.shape, r, center = (cx, cy))
            underlying_pixels = self.in_image[indsi, indsj]
            self.ax2.cla()
            self.ax2.plot(underlying_pixels, '.-')
            self.ax2.set_ylim([self.im.norm.vmin, self.im.norm.vmax])

        if self.mark_shape == 'rectangle':
            r1 = self.slider_top_left_r.val
            c1 = self.slider_top_left_c.val
            r2 = self.slider_bot_right_r.val
            c2 = self.slider_bot_right_c.val
            self.markshape.set_xy((c1, r1))
            self.markshape.set_width(abs(c2 - c1))
            self.markshape.set_height(abs(r2 - r1))
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
