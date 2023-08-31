#!/usr/bin/env python

"""Tests for `mcemtools` package."""

import pytest
import mcemtools

import numpy as np
import matplotlib.pyplot as plt

def test_pyMSSE():
    print('test_pyMSSE')
    print('%'*60)
    fitting_errors = np.random.rand(100)
    fitting_errors[10] = 100
    a,b,c,d = mcemtools.pyMSSE(fitting_errors)

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

if __name__ == '__main__':
    test_bin_4D()
    test_pyMSSE()
    test_cross_correlation_4D()
    test_SymmSTEM()
    test_centre_of_mass_4D()
    test_sum_4D()
    test_normalize_4D()
