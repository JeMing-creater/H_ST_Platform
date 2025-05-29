import os
import cv2
import PIL
import time
import math
import yaml
import torch
import monai

import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
from typing import Dict
from objprint import objstr
from easydict import EasyDict
from datetime import datetime
from accelerate import Accelerator

from timm.optim import optim_factory
from pycox.models.loss import CoxPHLoss

# from dataset.TCGA_dataloader import get_TCGA_dataloader
from dataset.TCGA_dataloader_clam import get_TCGA_data
from src import utils
from src.scheduler import LinearWarmupCosineAnnealingLR
from src.metrics import compute_sp_metrics, km_analysis
from models.get_model import get_model

def train_one_epoch(model: torch.nn.Module, 
                    loss_functions: Dict[str, torch.nn.modules.loss._Loss], 
                    train_loader: torch.utils.data.DataLoader, 
                    optimizer: torch.optim.Optimizer, 
                    scheduler: torch.optim.lr_scheduler._LRScheduler, 
                    accelerator: Accelerator, 
                    epoch: int, 
                    step: int,
                    logging_dir: str,
                    if_km: bool=False,):
    model.train()
    accelerator.print(f'Training...', flush=True)

    # train acc
    all_risks = []
    all_dd_labels = []
    all_vs_labels = []

    loop = tqdm(enumerate(train_loader), 
                total = len(train_loader))
    for i, batch in loop:    
        log = ''
        images = batch['image']  
        dd_labels = batch['dd_label']  
        vs_labels = batch['vs_label']

        # inference
        risk = model(images)

        # loss
        total_loss = 0
        loss_functions['Cox_loss'](risk, dd_labels, vs_labels)
        for name in loss_functions:
            if name == 'Cox_loss':
                loss = loss_functions[name](risk, dd_labels, vs_labels)
            elif name == 'MSE_loss':
                loss = loss_functions[name](risk, dd_labels)

            accelerator.log({'Train/' + name: float(loss)}, step=step)
            total_loss += loss
        
        # loss backward
        accelerator.backward(total_loss)
        optimizer.step()
        optimizer.zero_grad()
        accelerator.log({
            'Train/Total Loss': float(total_loss),
        }, step=step)

        # record risk acc
        all_risks.append(risk)
        all_dd_labels.append(dd_labels)
        all_vs_labels.append(vs_labels)

        # updata loop information
        loop.set_description(f'Epoch [{epoch+1}/{config.trainer.num_epochs}]')
        loop.set_postfix(loss=total_loss)
        step += 1

        
    ci, bs_mean, auc = compute_sp_metrics(all_risks, all_dd_labels, all_vs_labels, accelerator)
    if if_km == True:
        # draw km curve
        km_analysis(all_risks, all_dd_labels, all_vs_labels, accelerator,
                    logging_dir + f'/Train_survival_curve.png')
    # update scheduler    
    scheduler.step(epoch)
    return ci, bs_mean, auc, step

@torch.no_grad()
def val_one_epoch(model: torch.nn.Module,
                  loader: torch.utils.data.DataLoader,
                  loss_functions: Dict[str, torch.nn.modules.loss._Loss], 
                  step: int,
                  accelerator: Accelerator, 
                  logging_dir: str,
                  if_km: bool=False,
                  test: bool=False): 
    model.eval()
    if test == True:
        flag = 'Test'
        accelerator.print(f'Testing...', flush=True)
    else:
        flag = 'Val'
        accelerator.print(f'Valing...', flush=True)
    
    all_risks = []
    all_dd_labels = []
    all_vs_labels = []

    loop = tqdm(enumerate(loader), 
                total = len(loader))
    for i, batch in loop:
        images = batch['image']  
        dd_labels = batch['dd_label']  
        vs_labels = batch['vs_label']

        # inference
        risk = model(images)

        # loss
        total_loss = 0
        for name in loss_functions:
            if name == 'Cox_loss':
                loss = loss_functions[name](risk, dd_labels, vs_labels)
            elif name == 'MSE_loss':
                loss = loss_functions[name](risk, dd_labels)

            accelerator.log({f'{flag}/' + name: float(loss)}, step=step)
            total_loss += loss
        
        all_risks.append(risk)
        all_dd_labels.append(dd_labels)
        all_vs_labels.append(vs_labels)

        # updata loop information
        loop.set_description(f'Validation')
        loop.set_postfix(loss=total_loss)
        step += 1
    
    ci, bs_mean, auc = compute_sp_metrics(all_risks, all_dd_labels, all_vs_labels, accelerator)
    if if_km == True:
        # draw km curve
        km_analysis(all_risks, all_dd_labels, all_vs_labels, accelerator,
                    logging_dir + f'/{flag}_survival_curve.png')
    return ci, bs_mean, auc, step

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
    accelerator.print("Loading data...")
    train_loader, val_loader, test_loader = get_TCGA_data(config)  

    # model
    accelerator.print('Loading model...')
    model = get_model(config)

    # loss functions
    loss_functions = {
        'Cox_loss': CoxPHLoss().to(accelerator.device),
        # 'MSE_loss':  nn.MSELoss().to(accelerator.device),
    }

    # optimizer
    optimizer = optim_factory.create_optimizer_v2(model, 
                                                  opt=config.trainer.optimizer.name,
                                                  weight_decay=float(config.trainer.optimizer.weight_decay),
                                                  lr=float(config.trainer.optimizer.lr), 
                                                  betas=(config.trainer.optimizer.betas[0], config.trainer.optimizer.betas[1]))

    # scheduler
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=config.trainer.warmup,
                                              max_epochs=config.trainer.num_epochs)
    
  
    # start training
    accelerator.print("Start Training! ")

    starting_epoch = 0
    train_step = 0
    val_step = 0
    best_ci = 0.0
    best_auc = 0.0
    best_bs = 99999.0
    


    # resume training state
    if config.trainer.resume == True:
        model, optimizer, scheduler, train_loader, starting_epoch, train_step, val_step, best_ci, best_bs = utils.resume_train_state_SP(model, '{}'.format(
            config.finetune.checkpoint), optimizer, scheduler, train_loader, accelerator)

    # loading to gpus
    model, optimizer, scheduler, train_loader, val_loader, test_loader = accelerator.prepare(model, optimizer, scheduler, train_loader, val_loader, test_loader)

    for epoch in range(starting_epoch, config.trainer.num_epochs):
        accelerator.print(f'Epoch [{epoch+1}/{config.trainer.num_epochs}] - Starting training...')
        # train one epoch
        tr_ci, tr_bs_mean, tr_auc, train_step = train_one_epoch(model = model, 
                                                                loss_functions = loss_functions, 
                                                                train_loader = train_loader, 
                                                                optimizer = optimizer, 
                                                                scheduler = scheduler, 
                                                                accelerator = accelerator, 
                                                                epoch = epoch, 
                                                                logging_dir = logging_dir,
                                                                if_km = True,
                                                                step = train_step)
        
        
        # validate one epoch
        ci, bs_mean, auc, val_step = val_one_epoch(model = model,
                                                   loss_functions = loss_functions, 
                                                   loader = val_loader, 
                                                   step = val_step, 
                                                   accelerator = accelerator,
                                                   logging_dir = logging_dir,
                                                   if_km=True,
                                                   test = False)
        
        accelerator.print(f'Epoch [{epoch+1}/{config.trainer.num_epochs}] - Tr CI: {tr_ci}, TR AUC: {tr_auc} , Tr BS: {tr_bs_mean} | CI: {ci}, AUC: {auc}, BS: {bs_mean}')

        if accelerator.is_main_process:
            # save best model
            if ci > best_ci or (ci == best_ci and bs_mean < best_bs):
                best_ci = ci
                best_auc = auc
                best_bs = bs_mean
                accelerator.print(f'New best model found! Saving model with CI: {best_ci:.4f}, BS: {best_bs:.4f}')
                accelerator.save_state(output_dir=f"{os.getcwd()}/model_store/{config.finetune.checkpoint}/best")
            
            # save last model
            accelerator.print('Cheakpoint...')
            accelerator.save_state(output_dir=f"{os.getcwd()}/model_store/{config.finetune.checkpoint}/checkpoint")
            torch.save({'epoch': epoch, 
                        'best_ci': best_ci,
                        'best_bs': best_bs,
                        'train_step': train_step,
                        'val_step': val_step},
                        f'{os.getcwd()}/model_store/{config.finetune.checkpoint}/checkpoint/epoch.pth.tar')
    
    
    model = utils.load_pretrain_model(f"{os.getcwd()}/model_store/{config.finetune.checkpoint}/best/pytorch_model.bin", model, accelerator)

    ci, bs_mean, auc, val_step = val_one_epoch(model= model,
                                               loss_functions= loss_functions, 
                                               loader = test_loader, 
                                               step=val_step, 
                                               accelerator=accelerator,
                                               if_km=True,
                                               logging_dir = logging_dir,
                                               test=True)
    accelerator.print(f'Best model Test acc: CI: {best_ci}, AUC: {auc}, BS: {best_bs}')






