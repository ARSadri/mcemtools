#!/usr/bin/env python

"""Tests for `mcemtools` package."""

import pytest
import mcemtools

import numpy as np
import matplotlib.pyplot as plt

def test_locate_atoms():
    """
    """

def test_viewer_4D():
    data4D = np.random.rand(12, 12, 12, 12)
    mcemtools.viewer_4D(data4D)

if __name__ == '__main__':
    test_locate_atoms()
    test_viewer_4D()