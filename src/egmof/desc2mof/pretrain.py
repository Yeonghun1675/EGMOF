import os
import sys
import yaml
import torch
import argparse
import datetime
import numpy as np
import pandas as pd
from glob import glob
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from . import __desc2mof_dir__
from .dataset import CSVDataset, Scaler
from .model import Desc2MOF
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy

torch.multiprocessing.set_sharing_strategy("file_system")
os.environ['CUDA_VISIBLE_DEVICES']='0,1'


_IS_INTERACTIVE = hasattr(sys, "ps1")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default = f'{__desc2mof_dir__}/config.yml' )
    parser.add_argument('--is_test', type=bool, default=False)
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--devices', type=int, default = 1)
    parser.add_argument('--scaled', type=bool, default=True)
    parser.add_argument('--log_dir', type=str, default='./logs_desc2mof')
    parser.add_argument('--ckpt_dir', type=str, default='./ckpt_desc2mof')
    args = parser.parse_args()
    
    
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    pl.seed_everything(config['seed'])

    num_workers = config['num_workers']
    
    train_data_dir = config['train_data_dir']
    val_data_dir = config['val_data_dir']
    test_data_dir = config['test_data_dir']    
    




    
    # ckpt 
    ckpt_dir = f'{args.ckpt_dir}/' 
    os.makedirs(ckpt_dir, exist_ok=True)

    
    checkpoint_callback = ModelCheckpoint(
        dirpath = ckpt_dir, 
        verbose=True,
        save_last=True,
        save_top_k=1,
        monitor= 'val/avg_val_loss',  #"val/avg_val_loss",
        mode= 'min', #'min'
    )    
    seed = config['seed']

    

    if not args.is_test:
        os.makedirs(args.log_dir, exist_ok=True)
        logger = pl.loggers.TensorBoardLogger(
            args.log_dir,
            name=f'pretrain_seed{seed}', 
        )        
    
        lr_callback = pl.callbacks.LearningRateMonitor()
        #early_callback = EarlyStopping(monitor="val/avg_val_loss", mode="min",patience=20,)
        
        callbacks = [checkpoint_callback, lr_callback,]
    else:
        logger = False
        callbacks = []


    

    if not config['per_gpu_batchsize']:
        accumulate_grad_batches = 1
    elif num_device == 0:
        accumulate_grad_batches = config["batch_size"] // (
            config["per_gpu_batchsize"] * config["num_nodes"]
        )
    else:
        accumulate_grad_batches = config["batch_size"] // (
            config["per_gpu_batchsize"] * num_device * config["num_nodes"]
        ) 

    
    log_every_n_steps=10
    model = Desc2MOF(config)

    if config['resume_from'] is not None:
        model = Desc2MOF.load_from_checkpoint(config['resume_from'],  config=config, strict=False)

    if _IS_INTERACTIVE:
        strategy = None
    elif pl.__version__ >= '2.0.0':
        strategy = "ddp_find_unused_parameters_true"
    else:
        strategy = "ddp"
    
    trainer = Trainer(
                    
                      accelerator = args.accelerator,
                      devices = args.devices,
                      num_nodes = config['num_nodes'],
                      max_epochs= config['max_epochs'], 
                      logger=logger,
                      accumulate_grad_batches=accumulate_grad_batches,
                      benchmark=True,
                      strategy= strategy, #DDPStrategy(find_unused_parameters=True),
                    
                       log_every_n_steps=log_every_n_steps,
                      callbacks=callbacks
                     )   
    


    # scaler
    feature_name_dir = config.get('feature_name_dir',f'{__desc2mof_dir__}/../desc2mof/data/feature_name.txt')
    with open(feature_name_dir, 'r') as g:
        feature_names = [line.strip() for line in g.readlines()]
        
    mean = pd.read_csv(config['mean_dir'])[feature_names]
    std = pd.read_csv(config['std_dir'])[feature_names]
    scaler = Scaler(np.array(mean).squeeze(), np.array(std).squeeze(), 0, 1)

    # dataset
    train_data = CSVDataset(train_data_dir, max_len = config['max_token_len'], scaled=True, scaler=scaler, feature_name_dir = feature_name_dir)
    val_data = CSVDataset(val_data_dir, max_len = config['max_token_len'], scaled = True, scaler=scaler, feature_name_dir = feature_name_dir)
    test_data = CSVDataset(test_data_dir, max_len = config['max_token_len'], scaled = True, scaler=scaler, feature_name_dir = feature_name_dir)   
               

    train_loader =DataLoader(train_data, batch_size=config['batch_size'] ,
                            num_workers =num_workers,
                             shuffle=True)   
    
    val_loader =DataLoader(val_data, batch_size=config['batch_size'] ,
                            num_workers =num_workers,
                             shuffle=False) 
    
    test_loader =DataLoader(test_data, batch_size=config['batch_size'] ,
                            num_workers =num_workers,
                             shuffle=False) 
    
    if args.is_test:
        best_ckpt_list =  glob(os.path.join(ckpt_dir, '*.ckpt'))
        best_ckpt = [ ckpt for ckpt in best_ckpt_list if os.path.basename(ckpt).startswith('epoch') ][0]
        print(f'best_ckpt: ', best_ckpt)
        model = Desc2MOF.load_from_checkpoint(best_ckpt, config=config, strict=False)          
        trainer.test(model, test_loader)


    else:

        
        trainer.fit(model, train_loader, val_loader,  ckpt_path = config['resume_from'])








