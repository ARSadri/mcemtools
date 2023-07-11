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
    
if __name__ == '__main__':
    test_CoM4D()
    test_SymmSTEM()