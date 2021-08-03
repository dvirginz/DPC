
from utils.argparse_init import get_non_default
from argparse import Namespace
import torch

def load_params_from_checkpoint(hparams, parser):
    path = hparams.resume_from_checkpoint
    hparams_model = Namespace(**torch.load(path,map_location=torch.device('cpu'))['hyper_parameters'])
    hparams_model.max_epochs = hparams_model.current_epoch + 30
    for k,v in get_non_default(hparams,parser).items():
        setattr(hparams_model,k,v)
    hparams_model.show_vis = hparams.show_vis
    hparams_model.gpus = hparams.gpus
    for key in vars(hparams):
        if(key not in vars(hparams_model)):
            setattr(hparams_model,key,getattr(hparams,key,None))
    hparams = hparams_model
    if(not hparams.do_train):
        hparams.metric_to_track = hparams.metric_to_track.replace('val','test')
    return hparams
    