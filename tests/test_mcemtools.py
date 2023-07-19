#!/usr/bin/env python

"""Tests for `mcemtools` package."""

import pytest
import mcemtools

import numpy as np
import matplotlib.pyplot as plt

def test_numbers_as_images():
    dataset_shape = (10, 10, 32, 32)
    fontsize = 20
    dataset = mcemtools.numbers_as_images(dataset_shape, fontsize)

    ##########################################################################
    n_x, n_y, n_r, n_c = dataset_shape
    txt_width = int(np.log(np.maximum(n_x, n_y))
                    /np.log(np.maximum(n_x, n_y))) + 1
    number_text_base = '{ind_x:0{width}}, {ind_y:0{width}}'
    for ind_x, ind_y in zip([0,     n_x//3, n_x//2, n_x-1], 
                            [n_x-1, n_x//2, n_x//3, 0    ]):
        plt.figure()
        plt.imshow(dataset[ind_x, ind_y], cmap = 'gray') 
        plt.title(number_text_base.format(ind_x = ind_x, ind_y = ind_y,
                                          width = txt_width))
    plt.show()

def test_locate_atoms():
    """
    """
    
def test_pltfig_to_numpy():
    fig, ax = plt.subplots(111)
    ax[0].imshow(np.random.rand(100, 100))
    np_data = mcemtools.pltfig_to_numpy(fig)
    print(np_data.shape)

def test_viewer_4D():
    data4D = np.random.rand(32, 16, 24, 16)
    mcemtools.viewer_4D(data4D)

if __name__ == '__main__':
    test_numbers_as_images()
    test_locate_atoms()
    test_pltfig_to_numpy()
    test_viewer_4D()