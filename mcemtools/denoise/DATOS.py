"""TODO
* support fit2FixedK

"""
import numpy as np

from typing import Literal

def pyMSSE(res, MSSE_LAMBDA = 3, k = 12) -> tuple[np.ndarray, np.ndarray]:
    res_sq_sorted = np.sort(res**2)
    res_sq_cumsum = np.cumsum(res_sq_sorted)
    cumsums = res_sq_cumsum[:-1]/np.arange(1, res_sq_cumsum.shape[0])
    cumsums[cumsums==0] = cumsums[cumsums>0].min()
    adjacencies = (res_sq_sorted[1:]/cumsums) ** 0.5
    adjacencies[:k] = 0
    inds = np.where(adjacencies <= MSSE_LAMBDA)[0]
    return inds[-1], adjacencies

class DATOS:
    def __init__(self, 
                 N           : int,
                 classes     : np.ndarray = None,
                 mbatch_size : int = 1,
                 n_segments  : int = 1,
                 n_epochs    : int = 1,
                 sort_after  : Literal['each_segment', 'all_segments', None]
                     = 'all_segments',
                 ):
        self.N           = N              
        self.classes     = classes        
        self.mbatch_size = mbatch_size     
        self.sort_after  = sort_after
        self.n_segments  = n_segments
        self.n_epochs    = n_epochs
        
        if self.classes is None:
            self.classes = np.ones(self.N, dtype='int')
            
        self.classes_names, self.classes_counts = \
            np.unique(self.classes, return_counts = True)
        self.n_classes = self.classes_names.shape[0]

        assert self.classes.shape[0] == self.N,\
            'classes should be as many as number of data points N'
        self.prepare_segments()
        
    def prepare_segments(self, n_segments = None, n_epochs = None):
        if(n_epochs is not None):
            self.n_epochs = n_epochs    
        if(n_segments is not None):
            self.n_segments = n_segments
        pts_range = np.arange(self.N)
        self.segments = np.linspace(0, self.n_segments - 1, self.N, dtype='int')
        _, self.segments_sizes = np.unique(self.segments, return_counts = True)
        self.pos_indices = np.zeros(self.N * self.n_epochs, dtype='int')
        current_ind = 0
        for segcnt in np.unique(self.segments):
            segrange = pts_range[self.segments == segcnt]
            segrange = np.tile(segrange, self.n_epochs)
            self.pos_indices[current_ind: current_ind + segrange.shape[0]] = \
                segrange.copy()
            current_ind += segrange.shape[0]
        self.n_pos = self.pos_indices.shape[0]
        self.segment_cnt_perv = 0
        
    def sort(self, 
             fitting_errors : np.ndarray = None, 
             order : Literal['ascend', 'descend', 'random', None] = 'descend',
             fit2Outliers = False):
        '''
            Here we put the sorted indices for each class into a row of a block
            All rows have the same size. We fill the empty slots with -1
            (There will be empty slots because some classes have less number of
            members). Then use neighbour interpolation to fill up all the slots.
        '''
        if fitting_errors is not None:
            assert len(fitting_errors.shape) == 1,\
                'fitting_errors should be a 1d np.ndarray'
            assert fitting_errors.shape[0] == self.N,\
                f'fitting_errors are {fitting_errors.shape[0]} but must be {self.N}'
        next_in_line = 0
        adjacencies = None  # @UnusedVariable
        match order:
            case 'ascend':
                sort_inds = np.argsort(fitting_errors)
                if fit2Outliers:
                    outlier_inline, adjacencies = \
                        pyMSSE(fitting_errors[sort_inds].copy(), 
                               MSSE_LAMBDA = 3, k = 12)
                    next_in_line = outlier_inline + 1
                    if(next_in_line > self.N - self.mbatch_size):
                        next_in_line = self.N - self.mbatch_size
            case 'descend':
                sort_inds = np.argsort(fitting_errors)[::-1]
            case 'random':
                sort_inds = np.arange(self.N, dtype='int')
                np.random.shuffle(sort_inds)
            case _:
                sort_inds = np.arange(self.N, dtype='int')
        
        self.make_indices(sort_inds, next_in_line)
        return adjacencies
        
    def make_indices_interleave(
            self, sort_inds : np.ndarray, next_in_line : int = 0):
        self.classes_sorted = self.classes[sort_inds]
        
        self.sort_inds_byclass = np.zeros((self.n_classes, self.N))
        for class_cnt, class_name in enumerate(self.classes_names):
            class_inds = sort_inds[self.classes_sorted == class_name]
            class_inds_cnt = 0
            for ptcnt in range(self.N):
                self.sort_inds_byclass[class_cnt, ptcnt] = \
                    class_inds[class_inds_cnt]
                if self.classes_sorted[ptcnt] == class_name:
                    class_inds_cnt += 1
                if class_inds_cnt >= class_inds.shape[0]:
                    break
        
        if((self.sort_after == 'all_segments') | (self.sort_after is None)):
            self.next_in_line = next_in_line
            
        if(self.sort_after == 'each_segment'):
            if(self.next_in_line >= self.pos_indices.shape[0]):
                self.next_in_line = next_in_line
        self.status = True
    
    def make_indices(self, sort_inds : np.ndarray, next_in_line : int = 0):
        self.classes_sorted = self.classes[sort_inds]
        
        self.sort_inds_byclass = np.zeros((self.n_classes, self.N))
        for class_cnt, class_name in enumerate(self.classes_names):
            class_inds = sort_inds[self.classes_sorted == class_name]
            class_inds = np.tile(
                class_inds, int(np.ceil(self.N/class_inds.shape[0])) )
            class_inds = class_inds[:self.N]
            self.sort_inds_byclass[class_cnt] = class_inds
        
        if((self.sort_after == 'all_segments') | (self.sort_after is None)):
            self.next_in_line = next_in_line
            
        if(self.sort_after == 'each_segment'):
            if(self.next_in_line >= self.pos_indices.shape[0]):
                self.next_in_line = next_in_line
        self.status = True
    
    def make_lrates_and_epochs(self):
        """ Data driven relative lrates and number of epochs
        
            As proposed in the original DATOS, it is possible to drive lrate
            and number of epochs from the affinities, given a lower and upper
            bounderies for them.
            First we should define affinities using the adjacencies provided by
            MSSE and then, use that to define lrate and n-epochs.
        """
        ...
    
    def __call__(self):
        if(self.status):
            
            if(self.next_in_line >= self.n_pos):
                self.status = False
                inds = None
                return (inds, self.status)

            if((self.sort_after == 'all_segments') | (self.sort_after is None)):
                if(self.next_in_line + self.mbatch_size > self.n_pos):
                    self.next_in_line = self.n_pos - self.mbatch_size

            if(self.sort_after == 'each_segment'):
                if(self.segments[self.next_in_line] != self.segment_cnt_perv):
                    self.segment_cnt_perv = self.segments[self.next_in_line]
                    self.status = False
                    inds = None
                elif(self.segments[self.next_in_line + self.mbatch_size] != \
                        self.segment_cnt_perv):
                    self.next_in_line = \
                        int((self.segments == self.segment_cnt_perv).sum()) - \
                            self.mbatch_size
                            
            inds = np.zeros((self.n_classes, self.mbatch_size), dtype='int')
            for getcnt in range(self.mbatch_size):
                inds[:, getcnt] = \
                    self.sort_inds_byclass[:, \
                        self.pos_indices[self.next_in_line]]
                self.next_in_line += 1
            inds = inds.ravel()
            
            return (inds, self.status)
        else:
            self.status = False
            inds = None
            return (inds, self.status)    

if __name__ == '__main__':
    N = 100
    n_sweeps = 10
    n_epochs = 1
    n_segments = 1
    mbatch_size = 2
    # classes = np.concatenate((np.array([0]*int(0.25*N)), 
    #                           np.array([1]*int(0.40*N)), 
    #                           np.array([2]*int(0.35*N)))).ravel()
    classes = np.ones(N, dtype='int')
    DATOS_sampler = DATOS(N,
                          classes = classes,
                          mbatch_size = mbatch_size,
                          n_segments = n_segments,
                          n_epochs = n_epochs)
    for _ in range(n_sweeps):
        DATOS_sampler.sort(np.random.rand(N))
        while(True):
            inds, status = DATOS_sampler()
            if not status:
                break
            print(inds)