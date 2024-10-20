from pathlib import Path as pathlib_Path
import mcemtools

"""
This dir should include a ref directory that has at least noisy.npy in it
noisy.npy must give a 4D dataset larger than (2*n_prob+1, 2*n_prob+1, 16, 16)
you will set the n_prob in hyps_I4D below
The output log directory will be made right beside the ref directory.

| logs_root
|---- ref
|-------- noisy.npy
|-------- nonoise.npy (optional)
|---- output_dir (will be made here)

"""

noisy4D_fpath = './../../mcemtools_logs/EFSTEM_TiPost_463eV_10eVSlit/ref/noisy.npy'

noisy4D_fpath = pathlib_Path(noisy4D_fpath)
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
    learning_rate           = 1e-3,  
    learning_momentum       = 1e-4,  
    mbatch_size             = 2,     
    n_epochs                = 8,     
    n_segments              = 1,                    
    n_kernels               = 16,    
    infer_size_I4D          = 50,    
    PAC_loss_factor         = 0.0,   
    mSTME_loss_factor       = 0.0,   
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
    )

mcemtools.denoise.denoise4_unet.denoise4D_unet(logs_root, hyps_I4D)