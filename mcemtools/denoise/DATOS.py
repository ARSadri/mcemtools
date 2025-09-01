"""TODO
* support fit2FixedK

"""
import numpy as np
import torch
from lognflow import printprogress
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
        adjacencies = None
        if order == 'ascend':
            sort_inds = np.argsort(fitting_errors)
            if fit2Outliers:
                outlier_inline, adjacencies = \
                    pyMSSE(fitting_errors[sort_inds].copy(), 
                           MSSE_LAMBDA = 3, k = 12)
                next_in_line = outlier_inline + 1
                if(next_in_line > self.N - self.mbatch_size):
                    next_in_line = self.N - self.mbatch_size
        elif order == 'descend':
            sort_inds = np.argsort(fitting_errors)[::-1]
        elif order == 'random':
            sort_inds = np.arange(self.N, dtype='int')
            np.random.shuffle(sort_inds)
        else:
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

DATOS_OUT_OF_MEMORY_MSG = \
    'DATOS Inference: Can not allocate memory for predictions with shape:'\
    ' {predictions_shape}.' \
    ' Maybe call infer(indices, ...) with only a chunk of indices of all'\
    ' data points. Or maybe the statfunctions would suffice? i.e. you probably'\
    ' need the predictions to get some statistics out of it. If so, put'\
    ' callable_that_gives_statistics(prediction, data_index) in'\
    ' predictions_statfunc_list as a list of such functions in the input.'\
    ' Currently, predictions will not be returned because it is too big.'

class nn_from_torch:
    def __init__(self,
                 data_generator,
                 torchModel,
                 lossFunc,
                 device,
                 logger = print,
                 pass_indices_to_model = False,
                 learning_rate = 1e-6,
                 momentum = 1e-7,
                 fix_during_infer = False,
                 optimizer = None,
                 preds_as_tuple_index = None,
                 test_mode = False):
        """ Using pytorch 
            The optimizer must be SGD. So we only accept the 
            learning_rate and momentum for SGD.
            
            data_generator is a callable function that given indices for data
            points will produce data and labels
            data, labels = self.data_generator(indices)
            
            
        """
        self.preds_as_tuple_index = preds_as_tuple_index
        self.data_generator = data_generator
        self.torchModel = torchModel
        self.lossFunc = lossFunc
        self.device = device
        self.logger = logger
        self.infer_size = None
        self.test_mode = test_mode
        try:
            self.lossFunc = lossFunc.float()
            self.lossFunc = self.lossFunc.to(device)
        except:
            pass
        if test_mode:
            self.optimizer = None
        elif optimizer is None:
            self.optimizer = torch.optim.SGD(self.torchModel.parameters(),
                                             lr = learning_rate,
                                             momentum = momentum)
        else:
            self.optimizer = optimizer
        
        
        self.pass_indices_to_model = pass_indices_to_model
        
        self.fix_during_infer = fix_during_infer

    def update_learning_rate(self, lr):
        if self.optimizer is not None:
            for g in self.optimizer.param_groups:
                g['lr'] = lr
        
    def update_momentum(self, momentum):
        if self.optimizer is not None:
            for g in self.optimizer.param_groups:
                g['momentum'] = momentum
    
    def reset(self):
        try:
            self.torchModel.reset()
        except Exception as e:
            print('Trying to reset the model.')
            print(e)
    
    def load_parameters(self, fpath):
        self.torchModel.load_state_dict(torch.load(fpath), strict=False)
    
    def update(self, indices):
        """use the netowrk by PytTorch
            Call this function when infering data for sorting or for trianing
            Input arguments:
            ~~~~~~~~~~~~~~~~
            indices: indices of data points to update the network with
                default: False
            Output argument:
            ~~~~~~~~~~~~~~~~
            tuple of three: losses, list_of_stats, predictions
        """
        data, labels = self.data_generator(indices)
        if(not torch.is_tensor(data)):
            data = torch.from_numpy(data).float().to(self.device)
        self.optimizer.zero_grad()
        if(self.pass_indices_to_model):
            preds = self.torchModel(data, indices)
        else:
            preds = self.torchModel(data)
        if(not torch.is_tensor(labels)):
            labels = torch.from_numpy(labels).float().to(self.device)
        if(self.pass_indices_to_model):
            loss = self.lossFunc(preds, labels, indices)
        else:
            loss = self.lossFunc(preds, labels)
        if torch.isinf(loss) | torch.isnan(loss):
            self.optimizer.zero_grad()
        else:
            loss.backward()
            self.optimizer.step()            
        loss = loss.detach().to('cpu').numpy()
        torch.cuda.empty_cache()
        return loss
    
    def infer(self, indices,
                   return_predictions = False,
                   infer_size = 1,
                   show_progress = False,
                   predictions_statfunc_list = None):
        with torch.no_grad():
            # self.torchModel.eval()
            if infer_size is None:
                if self.infer_size is None:
                    pt_stop = 1
                    while(True):
                        try:
                            data, labels = \
                                self.data_generator(indices[:pt_stop])
                            if(not torch.is_tensor(data)):
                                data = torch.from_numpy(\
                                    data).float().to(self.device)
                            if(self.pass_indices_to_model):
                                preds = self.torchModel(data, indices[:pt_stop])
                            else:
                                preds = self.torchModel(data)
                            data = data.detach().to('cpu')
                            del data
                            if(not torch.is_tensor(labels)):
                                labels = torch.from_numpy(\
                                    labels).float().to(self.device)
                            
                            if self.fix_during_infer:
                                attrs_to_restore = (
                                    self.lossFunc.accumulated_PACBED + 1,
                                    self.lossFunc.accumulated_n_images + 1,
                                    self.lossFunc.accumulated_mSTEM + 1,
                                    self.lossFunc.mSTEM_loss_factor + 1)
                                
                            loss = self.lossFunc(preds, labels, indices[:pt_stop])
                        
                            if self.fix_during_infer:    
                                self.lossFunc.accumulated_PACBED, \
                                    self.lossFunc.accumulated_n_images, \
                                    self.lossFunc.accumulated_mSTEM, \
                                    self.lossFunc.mSTEM_loss_factor = (
                                        attrs_to_restore[0] - 1,
                                        attrs_to_restore[1] - 1,
                                        attrs_to_restore[2] - 1,
                                        attrs_to_restore[3] - 1)                           
                            else:
                                loss = self.lossFunc(preds, labels, indices[:pt_stop])
                                
                            pt_stop += 1
                            torch.cuda.empty_cache()
                        except:
                            self.infer_size = pt_stop - 2
                            if(self.infer_size<=0):
                                self.infer_size = 1
                            self.logger(f'infer_size: {self.infer_size}')
                            break
                        torch.cuda.empty_cache()
                infer_size = self.infer_size
                
            predictions = None
            predictions_stat_list = []
            n_pts = indices.shape[0]
            losses = np.zeros(n_pts, dtype='float32')
            pt_start = 0
            if(show_progress):
                pbar = printprogress(n_pts, title = f'Inferring {n_pts} points')
            while(pt_start < n_pts):
                pt_stop = pt_start + infer_size
                if pt_stop > n_pts:
                    pt_stop = n_pts
                _indices = np.array(indices[pt_start:pt_stop]).copy()
                data, labels = self.data_generator(_indices)
                if(not torch.is_tensor(data)):
                    data = torch.from_numpy(data).float().to(self.device)
                if(self.pass_indices_to_model):
                    preds = self.torchModel(data, _indices)
                else:
                    preds = self.torchModel(data)
                if(not torch.is_tensor(labels)):
                    labels = torch.from_numpy(labels).float().to(self.device)
                if self.test_mode:
                    loss = np.zeros(len(preds))
                else:
                    if self.fix_during_infer:    
                        attrs_to_restore = (
                            self.lossFunc.accumulated_PACBED,
                            self.lossFunc.accumulated_n_images,
                            self.lossFunc.accumulated_mSTEM,
                            self.lossFunc.mSTEM_loss_factor)
                        
                    if(self.pass_indices_to_model):
                        loss = self.lossFunc(preds, labels, _indices)
                    else:
                        loss = self.lossFunc(preds, labels)
        
                    if self.fix_during_infer:    
                        self.lossFunc.accumulated_PACBED, \
                            self.lossFunc.accumulated_n_images, \
                            self.lossFunc.accumulated_mSTEM, \
                            self.lossFunc.mSTEM_loss_factor = attrs_to_restore
                        
                    loss = loss.detach().to('cpu').numpy()

                data = data.detach().to('cpu')
                del data
                labels = labels.detach().to('cpu')
                del labels
                losses[pt_start:pt_stop] = loss.copy()
                del loss
                if self.preds_as_tuple_index is not None:
                    preds = preds[self.preds_as_tuple_index]
                if( (return_predictions == True) | 
                    (predictions_statfunc_list is not None)):
                    preds = preds.detach().to('cpu').numpy()
                if(return_predictions):
                    if(pt_start==0):
                        predictions_shape = ((n_pts,) + tuple(preds.shape[1:]))
                        try:
                            predictions = np.zeros(predictions_shape,
                                                   dtype=preds.dtype)
                        except Exception as e:
                            self.logger(DATOS_OUT_OF_MEMORY_MSG.format(
                                    predictions_shape = predictions_shape))
                            raise e
                if(return_predictions):
                    predictions[pt_start:pt_stop] = preds.copy()
                
                if(predictions_statfunc_list is not None):
                    for preds_cnt, _ind in enumerate(_indices):
                        for func_cnt, predictions_statfunc in \
                                enumerate(predictions_statfunc_list):
                            predstat = predictions_statfunc(\
                                preds[preds_cnt], _ind)
                            
                            if preds_cnt == 0:
                                predictions_stat_list.append(
                                    np.zeros(((n_pts,) + \
                                              tuple(predstat.shape[1:])),
                                              dtype = predstat.dtype)
                                    )
                            predictions_stat_list[func_cnt][
                                pt_start + preds_cnt] = predstat
                        
                del preds
                torch.cuda.empty_cache()
                if(show_progress):
                    pbar(pt_stop - pt_start)
                pt_start = pt_stop
        torch.cuda.empty_cache()
        # self.torchModel.train()
        return(losses, predictions_stat_list, predictions)

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
    DATOS_sampler = DATOS(
        N,
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
            # print(inds)