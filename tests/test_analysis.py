#!/usr/bin/env python

"""Tests for `mcemtools` package."""

import pytest
import mcemtools
import lognflow
import numpy as np
import matplotlib.pyplot as plt
import pathlib

def test_cross_correlation_4D():
    print('test_cross_correlation_4D')
    print('%'*60)
    data4D = np.random.rand(10, 11, 12, 13)
    ccorr = mcemtools.cross_correlation_4D(data4D, data4D)

def test_SymmSTEM():
    print('test_SymmSTEM')
    print('%'*60)
    data4D = np.random.rand(10, 11, 12, 13)
    symms = mcemtools.SymmSTEM(data4D)

def test_centre_of_mass_4D():
    print('test_centre_of_mass_4D')
    print('%'*60)
    data4D = np.random.rand(10, 11, 12, 13)
    COM = mcemtools.centre_of_mass_4D(data4D)
    
def test_sum_4D():
    print('test_sum_4D')
    print('%'*60)
    data4D = np.random.rand(10, 11, 12, 13)
    STEM, PACBED = mcemtools.sum_4D(data4D)

def test_bin_4D():
    print('bin_4D')
    print('%'*60)
    data4D = np.ones((100, 80, 60, 40))
    binned_data4D = mcemtools.bin_4D(data4D,2, 2)
    assert (binned_data4D == 4).all()
    print(binned_data4D.shape)
    
def test_normalize_4D():
    print('test_normalize_4D')
    print('%'*60)
    data4D = np.random.rand(100, 80, 60, 40)
    normalized_data4D = mcemtools.normalize_4D(data4D)
    print(normalized_data4D.shape)

def test_SymmSTEM_with_data():
    data4D = np.load('data4D.npy')
    mask2D = mcemtools.annular_mask(
        (data4D.shape[2], data4D.shape[3]), radius = 5)
    sym = mcemtools.SymmSTEM(data4D, mask2D)
    im = plt.imshow(sym[..., 90])
    lognflow.plt_colorbar(im)
    plt.show()
    
def test_normalize_4D_with_data():
    data4D = np.load('data4D.npy')
    print(f'data4D.shape: {data4D.shape}')
    
    mask2D = mcemtools.annular_mask(
        (data4D.shape[2], data4D.shape[3]), radius = 12)
    mask4D = mcemtools.mask2D_to_4D(mask2D, data4D.shape)
    data4D[mask4D == 0] = 0
    data4D_normalized = mcemtools.normalize_4D(data4D, mask4D)
    n_x, n_y, _, _ = data4D.shape
    avgs = np.zeros((n_x, n_y))
    stds = np.zeros((n_x, n_y))
    for xc in range(n_x):
        for yc in range(n_y):
            avgs[xc, yc] = data4D_normalized[xc,yc][mask4D[xc,yc] != 0].mean()
            stds[xc, yc] = data4D_normalized[xc,yc][mask4D[xc,yc] != 0].std()
    plt.figure()
    im = plt.imshow(avgs); lognflow.plt_colorbar(im)
    plt.figure()
    im = plt.imshow(stds); lognflow.plt_colorbar(im)
    plt.show()
    
if __name__ == '__main__':
    test_bin_4D()
    test_cross_correlation_4D()
    test_SymmSTEM()
    test_centre_of_mass_4D()
    test_sum_4D()
    test_normalize_4D()

    if(pathlib.Path('location_of_a_4DSTEM_dataset.npy').is_file()):
        test_SymmSTEM_with_data()
        test_normalize_4D_with_data()

