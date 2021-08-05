import os
import torch
from pathlib import Path
# from data.generic_shape_dataset import GenericShapeCorrDataSet
import torch_geometric.transforms as T
from datetime import datetime

def extract_model_path_for_hyperparams(start_path, model, hparams):
    keys = ["arch","dataset_name","latent_dim","random_rotate","random_scale","class_choice","experiment_name","current_time"]
    path_prefix = []
    for key in keys:
        if hasattr(hparams,key):
            path_prefix.append(f"{key}_{getattr(hparams,key)}")
        elif(key=='current_time'):
            path_prefix.append(datetime.now().strftime("%d_%m:%H:%M:%S"))
    
    path = os.path.join(
        start_path, type(model).__name__, *path_prefix)
    
    return path

