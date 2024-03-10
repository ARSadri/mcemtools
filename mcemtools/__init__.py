# -*- coding: utf-8 -*-

"""Top-level package for mcemtools."""

__author__ = """Alireza Sadri"""
__email__ = 'alireza.sadri@monash.edu'
__version__ = '0.9.2'

from .mcemtools import viewer_4D

from .analysis   import (cross_correlation_4D,
                       SymmSTEM,
                       centre_of_mass_4D,
                       sum_4D,
                       locate_atoms,
                       bin_image,
                       bin_4D,
                       normalize_4D)

from .masking import (annular_mask,
                      image_by_windows,
                      markimage,
                      mask2D_to_4D,
                      remove_islands_by_size)

from .tensor_svd import svd_fit, svd_eval

from .transforms import (get_polar_coords,
                         polar2image,
                         image2polar,
                         data4D_to_frame,
                         revalue_elements)

from .data import (open_muSTEM_binary,
                   load_dm4,
                   load_raw,
                   mesh_maker_2D,
                   data_maker_2D,
                   data_maker_4D,
                   feature_maker_4D,
                   )

from .denoise.denoise4_tsvd     import denoise4_tsvd
from .denoise.denoise4net       import denoise4net
from .denoise.cluster4net       import cluster4net
from .denoise.DATOS             import DATOS
from .denoise.nn_from_torch     import nn_from_torch