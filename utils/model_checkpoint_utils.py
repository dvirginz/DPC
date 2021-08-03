import os
import torch
from pathlib import Path
# from data.generic_shape_dataset import GenericShapeCorrDataSet
import torch_geometric.transforms as T

def extract_model_path_for_hyperparams(start_path, model, hparams):
    keys = ["arch","dataset_name","latent_dim","random_rotate","random_scale","class_choice","experiment_name"]
    path_prefix = []
    for key in keys:
        if hasattr(hparams,key):
            path_prefix.append(f"{key}_{getattr(hparams,key)}")

    
    path = os.path.join(
        start_path, type(model).__name__, *path_prefix)
    
    return path

