#!/usr/bin/env python

"""Tests for `mcemtools` package."""

import pytest
import mcemtools

import numpy as np
import matplotlib.pyplot as plt

def test_markimage():
    mcemtools.markimage(np.random.rand(100, 100), 'circle', cmap = 'jet')

def test_annular_mask():
    mask = mcemtools.annular_mask((100, 100), 
                 centre = (40, 60), radius = 30, in_radius = 16)
    plt.imshow(mask)
    plt.show()

def test_image_by_windows():
    image = mcemtools.annular_mask((100, 100), 
                 centre = (40, 60), radius = 30, in_radius = 16)
    im_by_win = mcemtools.image_by_windows( 
                 img_shape = image.shape, 
                 win_shape = (25, 25),
                 skip = (4, 7))
    viewed = im_by_win.image2views(image)
    image_rec = im_by_win.views2image(viewed)
    assert (image == image_rec).all()
    
def test_mask2D_to_4D():
    mask = mcemtools.annular_mask((100, 100), 
                 centre = (40, 60), radius = 30, in_radius = 16)
    mask4D = mcemtools.mask2D_to_4D(mask, (50, 60, 100, 100))
    print(mask4D.shape)

if __name__ == '__main__':
    test_markimage()
    test_annular_mask()
    test_image_by_windows()
    test_mask2D_to_4D()    