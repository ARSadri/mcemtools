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
                 centre = (40, 60), radius = 30, in_radius = 16).astype('float')
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
    
def test_remove_islands_by_size():
    ...

def test_new_shape():
    
    in_list = [[4, 4], [4, 4], [5, 5], [9, 5]]
    new_shape_list = [[6, 2], [7, 3], [3, 7], [4, 8]]
    
    for in_shape, new_shape in zip(in_list, new_shape_list):
        tens = (np.random.rand(*in_shape) * 100).astype('int')
        tens_new_shape = mcemtools.masking.crop_or_pad(tens, new_shape)
        print(tens, '\n' + '-'*5 + '\n', tens_new_shape, '\n' + '='*20)


    mcemtools.masking.crop_or_pad(np.zeros((3,4,5,6)), (3,4,3,3))

if __name__ == '__main__':
    test_new_shape()
    test_markimage()
    test_annular_mask()
    test_image_by_windows()
    test_mask2D_to_4D()
    test_remove_islands_by_size()