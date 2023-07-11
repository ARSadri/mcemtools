#!/usr/bin/env python

"""Tests for `mcemtools` package."""

import pytest

import mcemtools

def test_CoM4D():
    data4D = np.random.rand(10, 11, 12, 13)
    CoM = CoM4D(data4D)