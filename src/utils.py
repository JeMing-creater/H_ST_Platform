import os
import random
import sys
from collections import OrderedDict
from copy import deepcopy
import math
import numpy as np
import torch
from accelerate import Accelerator
from einops.layers.torch import Rearrange
from timm.models.layers import trunc_normal_
from torch import nn
from pathlib import Path
import numpy as np
import nibabel as nib


class Logger(object):
    def __init__(self, logdir: str):
        self.console = sys.stdout
        if logdir is not None:
            os.makedirs(logdir)
            self.log_file = open(logdir + '/log.txt', 'w')
        else:
            self.log_file = None
        sys.stdout = self
        sys.stderr = self

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.log_file is not None:
            self.log_file.write(msg)

    def flush(self):
        self.console.flush()
        if self.log_file is not None:
            self.log_file.flush()
            os.fsync(self.log_file.fileno())

    def close(self):
        self.console.close()
        if self.log_file is not None:
            self.log_file.close()

def same_seeds(seed):
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True

def load_model_dict(download_path, save_path=None, check_hash=True) -> OrderedDict:
    if download_path.startswith('http'):
        state_dict = torch.hub.load_state_dict_from_url(download_path, model_dir=save_path, check_hash=check_hash, map_location=torch.device('cpu'))
    else:
        state_dict = torch.load(download_path, map_location=torch.device('cpu'))
    return state_dict

def load_pretrain_model(pretrain_path: str, model: nn.Module, accelerator: Accelerator):
    try:
        state_dict = load_model_dict(pretrain_path)
        model.load_state_dict(state_dict)
        accelerator.print(f'Successfully loaded the training model for ', pretrain_path)
        return model
    except Exception as e:
        try:
            state_dict = load_model_dict(pretrain_path)
            new_state_dict = {}
            for key in state_dict.keys():
                new_state_dict[key.replace('module.', '')] = state_dict[key]
            model.load_state_dict(new_state_dict)
            accelerator.print(f'Successfully loaded the training modelfor ', pretrain_path)
            return model
        except Exception as e:
            accelerator.print(e)
            accelerator.print(f'Failed to load the training model！')
            return model

def resume_train_state_SP(model: torch.nn.Module, 
                          path: str, 
                          optimizer: torch.optim.Optimizer, 
                          scheduler: torch.optim.lr_scheduler._LRScheduler, 
                          train_loader: torch.utils.data.DataLoader,
                          accelerator: Accelerator):
    try:
        # Get the most recent checkpoint
        base_path = os.getcwd() + '/' + 'model_store' + '/' + path + '/checkpoint'
        epoch_checkpoint = torch.load(base_path + "/epoch.pth.tar", map_location='cpu')
        starting_epoch = epoch_checkpoint['epoch'] + 1
        best_ci = epoch_checkpoint['best_ci']
        best_bs = epoch_checkpoint['best_bs']
        train_step = epoch_checkpoint['train_step']
        val_step = epoch_checkpoint['val_step']
        model = load_pretrain_model(base_path + "/pytorch_model.bin", model, accelerator)
        optimizer.load_state_dict(torch.load(base_path + "/optimizer.bin"))
        scheduler.load_state_dict(torch.load(base_path + "/scheduler.bin"))
        accelerator.print(f'Loading training state successfully! Start training from {starting_epoch}, Best C1: {best_ci}')
        return model, optimizer, scheduler, train_loader, starting_epoch, train_step, val_step, best_ci, best_bs
    except Exception as e:
        accelerator.print(f'Failed to load training state: {e}')
        return model, optimizer, scheduler, train_loader, 0, 0, 0, 0.0, 99999.0
        