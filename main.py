import os
import cv2
import PIL
import time
import math
import yaml
import numpy as np
import pandas as pd

from PIL import Image
from objprint import objstr
from easydict import EasyDict
from datetime import datetime
from accelerate import Accelerator

from dataloader.TCGA_dataloader import get_TCGA_dataloader
from src import utils



if __name__ == "__main__":
    # Base setting
    config = EasyDict(yaml.load(open('config.yml', 'r', encoding="utf-8"), Loader=yaml.FullLoader))
    # Random seed
    utils.same_seeds(config.trainer.seed)

    # Log dir and GPU setting
    logging_dir = os.getcwd() + '/logs/' + config.finetune.checkpoint + str(datetime.now()).replace(' ','_').replace('-','_').replace(':','_').replace('.','_')
    accelerator = Accelerator(cpu=False, log_with=["tensorboard"], logging_dir=logging_dir)
    utils.Logger(logging_dir if accelerator.is_local_main_process else None)
    accelerator.init_trackers(os.path.split(__file__)[-1].split(".")[0])
    accelerator.print(objstr(config))

    # Load data
    train_loader, val_loader, test_loader = get_TCGA_dataloader(config)

    # check and split in first time
    for i, batch in enumerate(train_loader):
        print(batch[0].shape)
        print(batch[1].shape)
        print(batch[2].shape)
        print(batch[3].shape)
        print(batch[4].shape)
    
    for i, batch in enumerate(val_loader):
        print(batch[0].shape)
        print(batch[1].shape)
        print(batch[2].shape)
        print(batch[3].shape)
        print(batch[4].shape)

    for i, batch in enumerate(test_loader):
        print(batch[0].shape)
        print(batch[1].shape)
        print(batch[2].shape)
        print(batch[3].shape)
        print(batch[4].shape)
        




