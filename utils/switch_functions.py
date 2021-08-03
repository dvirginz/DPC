"""For all general switch functions based on flags."""

from data.point_cloud_db.tosca import TOSCA
from data.point_cloud_db.smal import SMAL
from data.point_cloud_db.surreal import surreal
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F

def model_class_pointer(task_name, model_name):
    """Get pointer to class base on flags.
    
    Arguments:
        task_name {str} -- the task name, shape_correspondence etc
        model_name {str} -- coup, etc
    
    Raises:
        Exception: If unknown task,model_name
    
    Returns:
        torch.nn.Module -- The module to train on
    """
    if task_name == "shape_corr":
        if model_name == 'DeepPointCorr':
            from models.DeepPointCorr.DeepPointCorr import DeepPointCorr
            return DeepPointCorr

    raise Exception("Unkown arch")


def get_dataloader(task_name, params):
    """
        Returns the relevant dataloader per task.
        
        Args:
            task_name (str, optional): Define which dataloader to choose from. .
        
        Returns:
            Dataloader
        """
    if task_name == "shape_corr":
        return DataLoader
    else:
        raise Exception("No match for task_name in load_dataset")


def load_dataset(params):
    """
    Return the correct dataset.

    Args:
        params (dict): The hyper-parameters
    """
    from data.point_cloud_db.shrec import SHREC

    if params.dataset_name in ["shrec"]:
        dataset = SHREC(params,"train")
        train_size = int(len(dataset) * params.train_val_split)
        train_dataset,val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
        test_dataset = SHREC(params,"test")
        return train_dataset,val_dataset,test_dataset

    if params.dataset_name in ["smal"]:
        dataset = SMAL(params,"train")
        train_size = int(len(dataset) * params.train_val_split)
        train_dataset,val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
        test_dataset = SMAL(params,"test")

        if(params.test_on_tosca):
            test_dataset = TOSCA(params,"test")

        return train_dataset,val_dataset,test_dataset

    if params.dataset_name in ["tosca"]:
        dataset = TOSCA(params,"train")
        train_size = int(len(dataset) * params.train_val_split)
        train_dataset,val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
        test_dataset = TOSCA(params,"test")
        return train_dataset,val_dataset,test_dataset

    if params.dataset_name in ["surreal"]:
        train_dataset = surreal(params,"train")
        val_dataset = surreal(params,"val")
        if(params.test_on_shrec):
            test_dataset = SHREC(params,"test")
        else:
            test_dataset = surreal(params,"test")

        return train_dataset,val_dataset, test_dataset

    raise Exception("No match for task_name in load_dataset")


def choose_optimizer(params, network_parameters):
    """
    Choose the optimizer from params.optimizer flag
    
    Args:
        params (dict): The input flags
        network_parameters (dict): from net.parameters()
    
    Raises:
        Exception: If not matched optimizer
    """
    if params.optimizer == "adam":
        optimizer = torch.optim.Adam(network_parameters, lr=params.lr, weight_decay=params.weight_decay)
    elif params.optimizer == "adamW":
        optimizer = torch.optim.AdamW(network_parameters, lr=params.lr,)
    elif params.optimizer == "sgd":
        optimizer = torch.optim.SGD(network_parameters, lr=params.lr, weight_decay=params.weight_decay, momentum=0.9,nesterov=True)
    
    elif params.optimizer == "rmsprop":
        optimizer = torch.optim.RMSprop(network_parameters, lr=params.lr, weight_decay=params.weight_decay,)    
    else:
        raise Exception("No valid optimizer provided")
    return optimizer



def measure_similarity(similarity_init, source_encoded, target_encoded):
    """
    Measure the similarity between two batched matrices vector by vector

    Args:
        similarity_init : The method to calculate similarity with(e.g cosine)
        source_encoded (BxNxF Tensor): The input 1 matrix
        target_encoded (BxNxF Tensor): The input 2 matrix
    """
    "multiplication", "cosine", "difference"
    if similarity_init == "cosine":
        a_norm = source_encoded / source_encoded.norm(dim=-1)[:, :, None]
        b_norm = target_encoded / target_encoded.norm(dim=-1)[:, :, None]
        return torch.bmm(a_norm, b_norm.transpose(1, 2))
    if similarity_init == "mult":
        return torch.bmm(source_encoded, target_encoded.transpose(1, 2))
    if similarity_init == "l2":
        diff = torch.cdist(source_encoded,target_encoded)
        return diff.max() - diff
    if similarity_init == "negative_l2":
        diff = -torch.cdist(source_encoded,target_encoded)
        return diff
    if similarity_init == "difference_exp":
        dist = torch.cdist(source_encoded.contiguous(), target_encoded.contiguous())
        return torch.exp(-dist * 2 * source_encoded.shape[-1])
    if similarity_init == "difference_inverse":
        # TODO maybe (max - tensor) instead of 1/tensor ?
        EPS = 1e-6
        return 1 / (torch.cdist(source_encoded.contiguous(), target_encoded.contiguous()) + EPS)
    if similarity_init == "difference_max_norm":
        dist = torch.cdist(source_encoded.contiguous(), target_encoded.contiguous())
        return (dist.max() - dist) / dist.max()
    if similarity_init == "multiplication":
        return torch.bmm(source_encoded, target_encoded.transpose(1, 2))


def normalize_P(P, p_normalization, dim=None):
    """
    The method to normalize the P matrix to be "like" a statistical matrix.
    
    Here we assume that P is Ny times Nx, according to coup paper the columns (per x) should be statistical, hence normalize column wise
    """
    if dim is None:
        dim = 1 if len(P.shape) == 3 else 0

    if p_normalization == "no_normalize":
        return P
    if p_normalization == "l1":
        return F.normalize(P, dim=dim, p=1)
    if p_normalization == "l2":
        return F.normalize(P, dim=dim, p=2)
    if p_normalization == "softmax":
        return F.softmax(P, dim=dim)
    raise NameError

