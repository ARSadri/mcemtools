import unittest
import numpy as np
from lognflow import printprogress
from mcemtools.data import segmented_to_4D
import scipy.ndimage
import scipy.signal
from itertools import product

def test_segmented_to_4D():
    # Sample data for testing
    n_ch, n_x, n_y = 3, 10, 10  # 3 channels, 10x10 data
    n_r, n_c = 8, 8  # 8x8 detector geometry

    # Create random channel-based data
    channel_based_data = np.random.rand(n_ch, n_x, n_y)

    # Create a detector geometry with segments labeled as 1, 2, 3 and 0 as background
    detector_geometry = np.zeros((n_r, n_c), dtype=int)
    detector_geometry[1:4, 1:4] = 1  # Segment 1
    detector_geometry[4:7, 4:7] = 2  # Segment 2
    detector_geometry[0:3, 5:8] = 3  # Segment 3

    # Initialize the segmented_to_4D object
    segment_to_4D_obj = segmented_to_4D(channel_based_data, detector_geometry)

    # Test __getitem__ with a single pixel
    cbed_single = segment_to_4D_obj[1, 1]
    assert cbed_single.shape == (n_r, n_c), "Single pixel access failed"

    # Test __getitem__ with a slice
    cbed_slice = segment_to_4D_obj[1:3, 1:3]
    assert cbed_slice.shape == (2, 2, n_r, n_c), "Slice access failed"

    # Test get_stat function
    stem, pacbed, com_x, com_y, pacbed_com_x, pacbed_com_y = segment_to_4D_obj.get_stat()
    assert stem.shape == (n_x, n_y), "get_stat stem calculation failed"
    assert pacbed.shape == (n_r, n_c), "get_stat pacbed calculation failed"
    assert com_x.shape == (n_x, n_y), "get_stat com_x calculation failed"
    assert com_y.shape == (n_x, n_y), "get_stat com_y calculation failed"

    # Test filtered_by_kernel function
    coords = np.array([[1, 1], [2, 2], [3, 3]])
    win_side = 3
    weights = np.array([0.1, 0.5, 0.9])

    filtered_com_x, filtered_com_y, kernel = segment_to_4D_obj.filtered_by_kernel(coords, win_side, weights)
    print(f'filtered_com_x:{filtered_com_x.shape}')

    print("All tests passed!")

# Run the test
test_segmented_to_4D()
