from pathlib import Path as pathlib_Path
import mcemtools

"""
This dir should include a ref directory that has at least noisy.npy in it
noisy.npy must give a 4D dataset larger than (2*n_prob+1, 2*n_prob+1, 16, 16)
you will set the n_prob in hyps_I4D below
The output log directory will be made right beside the ref directory.

from the following website:
https://bridges.monash.edu/articles/dataset/Experimental_data/25815436

download SrTiO3_High_mag_Low_dose.npy and rename it to noisy.npy
download SrTiO3_High_mag_High_dose.npy and rename it to nonoise.npy

then make data4D_shape = np.array(noisy.shape) and save it as data4D_shape.npy

put all three in a folder caled ref....then give the noisy.npy location to the
rest of this code.

| log_dir
|---- ref
|-------- noisy.npy
|-------- nonoise.npy (optional for comparison)
|-------- data4D_shape.npy
|---- output_dir (will be made here)

"""

noisy4D_fpath = pathlib_Path(r"noisy.npy")

#~~~~~~~~~~~~~~~~~~ START ~~~~~~~~~~~~~~~~~

assert noisy4D_fpath.is_file(), f'File not found!'
logs_root = noisy4D_fpath.parent.parent

"""
This code can receive many options but the following are hyper-parameters of the 
un-supervised deep operation 
total number of epochs will be n_refine_steps * n_ksweeps * n_epochs
decays will be multiplied to rates
"""
n_refine_steps              = 4     #number of refinements: each refinment adds up the denoised with noisy
hyps_I4D = dict(
    n_prob                  = 3,     #number of probe positions in the window around a pattern
    learning_rate           = 1e-4, #with synthetic data : 1e-3
    learning_momentum       = 1e-5,  #always 1/10 of lr
    mbatch_size             = 2,     
    n_epochs                = 8,        #go over a set of batches for this number of epochs 
    n_segments              = 1,   #treat the entire dataset as one dataset when set to 1                
    n_kernels               = 32,    #between 8 to 64
    infer_size_I4D          = 50,    #inference batch size
    PAC_loss_factor         = 0/100.0,   #when including PACBED in loss percentage of loss that goes to PACBED, e.g. 0.5%
    mSTEM_loss_factor       = 0/100.0,   #percentage of loss that goes to STEM, e.g. 0.5%
    n_ksweeps               = 4,    #rebatch into random samples this number of times per refinement
    n_ksweeps_last          = 4,
    n_refine_steps          = n_refine_steps,
    learning_rate_decay     = 0.1**(1/(n_refine_steps - 1)),
    learning_momentum_decay = 0.1**(1/(n_refine_steps - 1)),
    reset_on_refine         = True,     #We use a clear network at the begining of each refinment as if this is a new denoising
    test_mode               = False,
    use_mu_eaxct            = False,    #when you have somehow denoised your STEM image and would like to force it at the output
    rejection_ratio_list    = [70, 60, 50, 0],  #less than this percentage in the patterns are set to zero at the correcponding refinement
    refine_by_labels        = True, #that is to use the refined solution as input of the next refinement step

    # not all patterns in the total STEM image have same number of electrons, 
    # #cluster the total STEM image into a number of segments depeneding on 
    # #the value and treat each segment inequall according to a list: 
    # e.g. [1, 2] will cluster STEM images into two groups based on their Ne,
    # then trains for the upper cluster half less than the patterns of the 
    # lower cluster. This way, patterns that have less Ne will be 
    # mentioned to the network twice as much.
    repeat_by_scattering    = None, 
    
    # you can decide where to take patterns for training from, 
    # give an image that has 0 and 1 and patterns for training 
    # will come from the area that has 1 on it....currently np.ones(STEM.shape)
    trainable_area          = None,
    
    # we do not wish to train for the dark field if it is large you can
    # mask the PACBED
    PACBED_mask             = None,
    )

logger = mcemtools.denoise.denoise4_unet.denoise4D_unet(logs_root, hyps_I4D)
logger.log_code()