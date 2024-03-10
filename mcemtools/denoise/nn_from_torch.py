import torch
import numpy as np
from lognflow import printprogress

DATOS_OUT_OF_MEMORY = \
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
                 momentum = 0,
                 fix_during_infer = False):
        """ Using pytorch 
            The optimizer must be SGD. So we only accept the 
            learning_rate and momentum for SGD.
        """
        self.data_generator = data_generator
        self.torchModel = torchModel
        self.lossFunc = lossFunc
        self.device = device
        self.logger = logger
        self.infer_size = None
        self.lossFunc = lossFunc.float().to(device)
        self.optimizer = torch.optim.SGD(self.torchModel.parameters(),
                                         lr = learning_rate,
                                         momentum = momentum)
        self.pass_indices_to_model = pass_indices_to_model
        
        self.fix_during_infer = fix_during_infer

    def update_learning_rate(self, lr):
        for g in self.optimizer.param_groups:
            g['lr'] = lr
        
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
        loss = self.lossFunc(preds, labels, indices)
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
            self.torchModel.eval()
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
                                attrs_to_retore = (
                                    self.lossFunc.accumulated_PACBED + 1,
                                    self.lossFunc.accumulated_n_images + 1,
                                    self.lossFunc.accumulated_mSTEM + 1,
                                    self.lossFunc.mSTEM_loss_factor + 1)
                                
                            loss = self.lossFunc(preds, labels, indices[:pt_stop])
                        
                            if self.fix_during_infer:    
                                self.lossFunc.accumulated_PACBED, \
                                    self.lossFunc.accumulated_n_images, \
                                    self.lossFunc.accumulated_mSTEM, \
                                    self.lossFunc.mSTEM_loss_factor = (attrs_to_retore[0] - 1,
                                                              attrs_to_retore[1] - 1,
                                                              attrs_to_retore[2] - 1,
                                                              attrs_to_retore[3] - 1)                           
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
                data = data.detach().to('cpu')
                del data
                if(not torch.is_tensor(labels)):
                    labels = torch.from_numpy(labels).float().to(self.device)
                
                if self.fix_during_infer:    
                    attrs_to_retore = (self.lossFunc.accumulated_PACBED + 1,
                                       self.lossFunc.accumulated_n_images + 1,
                                       self.lossFunc.accumulated_mSTEM + 1,
                                       self.lossFunc.mSTEM_loss_factor + 1)
                    
                loss = self.lossFunc(preds, labels, _indices)
    
                if self.fix_during_infer:    
                    self.lossFunc.accumulated_PACBED, \
                        self.lossFunc.accumulated_n_images, \
                        self.lossFunc.accumulated_mSTEM, \
                        self.lossFunc.mSTEM_loss_factor = (attrs_to_retore[0] - 1,
                                                  attrs_to_retore[1] - 1,
                                                  attrs_to_retore[2] - 1,
                                                  attrs_to_retore[3] - 1)
                    
                # loss = self.lossFunc(preds, labels)
                labels = labels.detach().to('cpu')
                del labels
                loss = loss.detach().to('cpu').numpy()
                losses[pt_start:pt_stop] = loss.copy()
                del loss
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
                            self.logger(DATOS_OUT_OF_MEMORY.format(
                                    predictions_shape = predictions_shape))
                            self.torchModel.train()
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
        self.torchModel.train()
        return(losses, predictions_stat_list, predictions)