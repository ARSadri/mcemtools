#!/usr/bin/env python

"""Tests for `mcemtools` package."""

import pytest
import numpy as np
import mcemtools

def test_CoM4D():
    print('test_CoM4D')
    print('%'*60)
    data4D = np.random.rand(10, 11, 12, 13)
    CoM = mcemtools.annular_CoM4D(data4D)
    
def test_SymmSTEM():
    print('test_SymmSTEM')
    print('%'*60)
    data4D = np.random.rand(10, 11, 12, 13)
    mcemtools.SymmSTEM(data4D)
    
def test_numbers_as_images():
    dataset_shape = (10, 10, 32, 32)
    fontsize = 20
    dataset = numbers_as_images(dataset_shape, fontsize)

    ##########################################################################
    n_x, n_y, n_r, n_c = dataset_shape
    txt_width = int(np.log(np.maximum(n_x, n_y))
                    /np.log(np.maximum(n_x, n_y))) + 1
    logger = lognflow()
    logger.log_single(f'numbers4D_{n_x}x{n_y}x{n_r}x{n_c}', 
                      dataset, time_tag = False)
    
    logger.log_multichannel_by_subplots(
        'sample_4x4', 
        dataset[:4, :4].reshape(16, n_r, n_c).swapaxes(0,1).swapaxes(1,2),
        time_tag = False)
    number_text_base = '{ind_x:0{width}}, {ind_y:0{width}}'
    for ind_x, ind_y in zip([0,     n_x//3, n_x//2, n_x-1], 
                            [n_x-1, n_x//2, n_x//3, 0    ]):
        plt.figure()
        plt.imshow(dataset[ind_x, ind_y], cmap = 'gray') 
        plt.title(number_text_base.format(ind_x = ind_x, ind_y = ind_y,
                                          width = txt_width))
    plt.show()

if __name__ == '__main__':
    test_CoM4D()
    test_SymmSTEM()