#!/usr/bin/env python

"""Tests for `mcemtools` package."""

import pytest
import numpy as np
import mcemtools
import matplotlib.pyplot as plt

#get_polar_coords, polar2image, image2polar, bin_4D, normalize_4D

def test_image2polar():
    print('test_image2polar')
    print('%'*60)
    data = plt.np.random.rand(300, 200)
    print(data.shape)
    n_distinct_angles = 720
    n_distinct_rads = 360
    
    polar_image, polar_imageq, polar_mask, _ = mcemtools.image2polar(data)
    print(f'polar_image.shape: {polar_image.shape}')
    
    recon_image, recon_mask = mcemtools.polar2image(polar_image, 
                                          data.shape,
                                          polar_imageq)
    
    plt.figure(), plt.imshow(data, cmap = 'gray'), plt.colorbar()
    plt.figure(), plt.imshow(polar_image)
    plt.figure(), plt.imshow(polar_imageq)
    plt.figure(), plt.imshow(recon_image, cmap = 'gray'), plt.colorbar()
    plt.show()

def test_polar2image():
    print('test_polar2image')
    print('%'*60)
    test_image2polar()

def test_get_polar_coords():
    print('test_polar2image')
    print('%'*60)
    img = np.random.rand(100, 100)
    get_polar_coords_output = mcemtools.get_polar_coords(
        img.shape, (360 - 1, 170 - 1), None)
    
    mcemtools.image2polar(
        img, get_polar_coords_output = get_polar_coords_output)

if __name__ == '__main__':
    test_image2polar()
    test_polar2image()
    test_get_polar_coords()