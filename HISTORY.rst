=======
History
=======

0.1.0 (2022-09-07)
------------------

* First release on PyPI.

0.2.0 (2023-07-11)
------------------

* Many functions used for 4D-STEM analysis have been added.
* OOP is avoided as much as possible.

0.3.0 (2023-07-11)
------------------

* Tests are increasing in number
* init includes import of mcemtools

0.4.0 (2023-07-11)
------------------

* Many small bugs are fixed

0.5.0 (2023-07-12)
------------------
* Added binning to transforms

0.6.0 (2023-07-14)
------------------
* More tests are added
* Names are consistant accross the package.

0.7.0 (2023-07-15)
------------------
* markimage bug is fixed

0.8.0 (2023-07-19)
------------------
* markimage bug is fixed really as US version of centre is center.
* napari based GUI is added to mcemtools.
* bianary files can be read too.

0.8.1 (2023-07-27)
------------------
* viewer_4D is a lot more concise and bug free.

0.8.2 (2023-07-27)
------------------
* by pressing i, you get shapes info, m shows the mask and F5 updates the mask.

0.8.3 (2023-07-28)
------------------
* critical bug was fixed in viewer_4D

0.8.4 (2023-07-28)
------------------
* critical bug was fixed in viewer_4D

0.8.5 (2023-08-03)
------------------
* bug fixed in image_by_windows

0.8.6 (2023-08-05)
------------------
* viewer_4D handles arrows to move the selected objects.

0.8.7 (2023-08-31)
------------------
* added load_raw and load_dm4.
* bug fixed in viewer_4D, moves a lot smoother than before.
* viewer_4D shows a single image when size of the mask is 1 pixel, rapidly.
* conv_4D added to analysis

0.8.8 (2023-09-04)
------------------
* critical bug fixed in bin_4D

0.8.9 (2023-09-20)
------------------
* bug removed from Normalize_4D
* bug removed from SymmetrySTEM
* tets added to locate atom and ...

0.8.10 (2023-09-12)
-------------------
* bug removed from SymmetrySTEM and mirror is added
* data_maker_4D can be updated
* remove_islands_by_size bug fixed

0.8.11 (2023-09-25)
-------------------
* Bug fixed in viewer_4D

0.9.0 (2023-10-13)
-------------------
* added denoise4net
* added clustering4net and feature_maker_4D

0.9.1 (2024-02-23)
-------------------
* added denoise4_tsvd
* faster bin_4D
* torch handler does not update the model if the loss is inf or nan

0.9.2 (2024-03-23)
-------------------
* annular mask can handle even sized image

0.9.2 (2024-05-02)
-------------------
* viewer_4D takes second mask to subtract

0.9.3 (2024-05-24)
-------------------
* viewer_4D takes second mask to subtract

0.9.4 (2024-06-01)
-------------------
* minimal requirenments and test scripts for denoising is added
* first denoising example added
* many bugs fixed for the denoiser
* More concerete example is added for geometric flow based denoising for the paper

0.9.5 (2024-11-08)
-------------------
* Annular mask fixed
* added supervised 2D denoiser
* renamed denoise4net to denoise4_unet

0.9.6 (2024-11-08)
-------------------
* denoiser needed a fix

0.9.7 (2024-11-15)
-------------------
* critical bug fixed in applying spatial incoherence

0.9.8 (2024-11-15)
-------------------
* Maybe the installer should not install napari stuff
* removed autoconvolution from symmstem