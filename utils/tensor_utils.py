import torch
import numbers
import numpy as np 

def to_numpy(tensor):
    """Wrapper around .detach().cpu().numpy() """
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        return tensor
    elif isinstance(tensor,numbers.Number):
        return np.array(tensor)
    else:
        raise NotImplementedError

def to_tensor(ndarray):
    if isinstance(ndarray, torch.Tensor):
        return ndarray
    elif isinstance(ndarray, np.ndarray):
        return torch.from_numpy(ndarray)
    elif isinstance(ndarray,numbers.Number):
        return torch.tensor(ndarray)
    else:
        raise NotImplementedError

def to_number(ndarray):
    if isinstance(ndarray, torch.Tensor) or isinstance(ndarray, np.ndarray):
        return ndarray.item()
    elif isinstance(ndarray,numbers.Number):
        return ndarray
    else:
        raise NotImplementedError

def mean_non_zero(tensor,axis):
    mask = tensor!=0
    tensor_mean = (tensor*mask).sum(dim=axis)/mask.sum(dim=axis)
    return tensor_mean


def recursive_dict_tensor_to_cuda(dict_of_tensors: dict, subset_batch=None):
    for k in dict_of_tensors.keys():
        if isinstance(dict_of_tensors[k], dict):
            recursive_dict_tensor_to_cuda(dict_of_tensors[k], subset_batch)
        else:
            if(isinstance(dict_of_tensors[k], torch.Tensor)):
                dict_of_tensors[k] = dict_of_tensors[k].cuda()
                if(subset_batch):
                    dict_of_tensors[k]=dict_of_tensors[k][:subset_batch]
    return dict_of_tensors
