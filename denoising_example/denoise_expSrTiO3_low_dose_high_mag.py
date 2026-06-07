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
n_refine_steps              = 4
hyps_I4D = dict(
    n_prob                  = 3,     
    learning_rate           = 1e-4,  
    learning_momentum       = 1e-5,  
    mbatch_size             = 2,     
    n_epochs                = 8,     
    n_segments              = 1,                    
    n_kernels               = 32,    
    infer_size_I4D          = 50,    
    PAC_loss_factor         = 0.0,   
    mSTEM_loss_factor       = 0.0,   
    n_ksweeps               = 4,
    n_ksweeps_last          = 4,
    n_refine_steps          = n_refine_steps,
    learning_rate_decay     = 0.1**(1/(n_refine_steps - 1)),
    learning_momentum_decay = 0.1**(1/(n_refine_steps - 1)),
    reset_on_refine         = True,
    test_mode               = False,
    use_mu_eaxct            = False,
    rejection_ratio_list    = [70, 60, 50, 0],
    refine_by_labels        = True,
    repeat_by_scattering    = None,
    trainable_area          = None,
    PACBED_mask             = None,
    )

logger = mcemtools.denoise.denoise4_unet.denoise4D_unet(logs_root, hyps_I4D)
logger.log_code()