#!/usr/bin/env python

"""Tests for `mcemtools` package."""

import pytest
import mcemtools

import numpy as np
import matplotlib.pyplot as plt

def test_svd_fit():
    data = np.random.rand(100, 90, 80)
    mcemtools.svd_fit(data, rank = (10, 10, 10))
    
def test_svd_eval():
    data4D = np.random.randn(32, 16, 8)
    rank = (16, 16, 8)
    # scree = mcemtools.tensor_svd.scree_plots(data)
    # plt.plot(scree[0], '-*'), plt.plot(scree[1], '-*'), 
    # plt.plot(scree[2], '-*')

    data = data4D.copy()
    data_mean_along_pts = np.expand_dims(data.mean(0).squeeze(), axis = 0)
    data -= np.tile(data_mean_along_pts, (data.shape[0], 1, 1))
    U, ordered_indexes = mcemtools.svd_fit(data, rank)
    model = ((U, ordered_indexes), data_mean_along_pts)
    
    model, data_mean_along_pts = model
    data = data4D.copy()
    data -= np.tile(data_mean_along_pts, (data.shape[0], 1, 1))
    projection = mcemtools.svd_eval(data, *model)
    projection += np.tile(data_mean_along_pts, (data.shape[0], 1, 1))
    
    plt.plot(data4D.ravel() , projection.ravel(), '*')
    plt.show()
    
if __name__ == '__main__':
    test_svd_fit()
    test_svd_eval()
