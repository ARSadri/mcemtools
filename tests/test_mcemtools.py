#!/usr/bin/env python

"""Tests for `mcemtools` package."""

import pytest
import mcemtools
import pathlib
import numpy as np
import matplotlib.pyplot as plt

def test_locate_atoms():
    data4D = np.load('data4D.npy')
    print(f'data4D.shape: {data4D.shape}')
    locs = mcemtools.locate_atoms(data4D)
    print(locs)
    

def test_viewer_4D():
    data4D = np.load('data4D.npy')
    print(f'data4D.shape: {data4D.shape}')
    mcemtools.viewer_4D(data4D)

if __name__ == '__main__':
    if(pathlib.Path('./data4D.npy').is_file()):
        test_locate_atoms()
        test_viewer_4D()
