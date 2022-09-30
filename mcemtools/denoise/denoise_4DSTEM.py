"""
Module to denoise 4DSTEM data.
"""

class denoise_4DSTEM:
    """ 
    4D STEM denoiing
    
    This is the main class for denoising the dataset using unsupervised ML.
    
    Unsupervised denoising
    
    It is only natural to assume that noise is the common mode of all indipendent
    data points. This is different from outliers that are not going to be modeled.
    
    As such if there is a function that can map every data point to another, that
    function can only capture the noise that is common between all data points.
    
    One way to visualize this is to give the pair-wise error matrix for all pixels
    and show it as an image, and then, trying to fit a planar model to such an image. 
    If the noise is not hetrosedastic,  this will look like a plane for 
    a bell shaped type of PDF such as a Gaussian. Underfitting this with a plane 
    gives some noise model. Overfitting to it using a NN will give a better 
    boise model, especially if the noise is not modeled by a bell shaped or is 
    hetrosedastic.
    
    This principle allows us to create a dataset out of the input dataset that 
    is similar to input but lacks some data or is heavily interleaved and shuffled.
    
    In this class of problem/solution, we have the following functions:
        * Class initialization: all training parameters go here
        * training dataset generator: This is given here and the output will 
            be another dataset
        * solve_step: which will go through the solution for a step
        * solve: You can check attributes for logging to see if you'd like to stop
            Then it provides you with the output labeling function.
    
    Attributes
    ----------
        lr: float
            basic learning rate of the NN
    
    Example
    -------
        Create a denoiser
            import mcemtools.denoise_4DSTEM
            denoiser = mcemtools.denoise_4DSTEM(lr = 1e-3)
    """
    def __init__(self, lr : float = 1e-3):
        """ initialize a denoising problem for training of a NN for 4DSTEM data
         Parameters
        ----------
            lr : float
                basic learning rate of the NN
        """
        self.base_lr = lr
    
    def generate_datasets(self, data):
        """ generate datasets for training out of a 4D numpy array
        
        Note
        ----
            This function expects a numpy array as input with some directives
            on how to generate a dataset. You could also directly set the datasets 
            of this object to point to your own.

        Parameters
        ----------
            data: numpy.ndarray
                 This is a 4D numpy nd array that will contain real_x, real_y,
                 invers_u and inverse_v data in complex format.
            win_size: int, default 5
                This is window size to set how many probe positions are included
                in the analysis
        Returns
        -------
            X : .. module:: numpy.ndArray
                Inputs of the network and their number is X.shape[0]
            Y : .. module:: numpy.ndArray
                Outputs of the network and their number is Y.shape[0]
        """
        X = np.random.rand(10, 10)
        Y = np.random.rand(10, 10)
        return X, Y