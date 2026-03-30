import os
import sys
import yaml
import torch
import argparse
import numpy as np
import pandas as pd
from glob import glob
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from . import __desc2mof_dir__
from .dataset import MOFGenDataset, Scaler
from .model import Desc2MOF
from lightning import Trainer
import lightning as pl


torch.multiprocessing.set_sharing_strategy("file_system")
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


_IS_INTERACTIVE = hasattr(sys, "ps1")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=f"{__desc2mof_dir__}/config.yml")
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--devices", type=int, default=1)

    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    pl.seed_everything(config["seed"])

    num_workers = config["num_workers"]

    test_data_dir = config["target_data_dir"]

    # ckpt
    ckpt_dir = config["desc2mof_ckpt_dir"]

    seed = config["seed"]

    logger = False
    callbacks = []

    if not config["per_gpu_batchsize"]:
        accumulate_grad_batches = 1
    elif num_device == 0:
        accumulate_grad_batches = config["batch_size"] // (
            config["per_gpu_batchsize"] * config["num_nodes"]
        )
    else:
        accumulate_grad_batches = config["batch_size"] // (
            config["per_gpu_batchsize"] * num_device * config["num_nodes"]
        )

    if _IS_INTERACTIVE:
        strategy = None
    elif pl.__version__ >= "2.0.0":
        strategy = "ddp_find_unused_parameters_true"
    else:
        strategy = "ddp"

    trainer = Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        num_nodes=config["num_nodes"],
        max_epochs=config["max_epochs"],
        logger=logger,
        accumulate_grad_batches=accumulate_grad_batches,
        benchmark=True,
        strategy=strategy,  # DDPStrategy(find_unused_parameters=True),
        #    log_every_n_steps=log_every_n_steps,
        callbacks=callbacks,
    )

    # scaler
    feature_name_dir = config.get(
        "feature_name_dir", f"{__desc2mof_dir__}/../desc2mof/data/feature_name.txt"
    )
    with open(feature_name_dir, "r") as g:
        feature_names = [line.strip() for line in g.readlines()]

    mean = pd.read_csv(config["mean_dir"])[feature_names]
    std = pd.read_csv(config["std_dir"])[feature_names]
    scaler = Scaler(np.array(mean).squeeze(), np.array(std).squeeze(), 0, 1)

    # dataset
    test_data = MOFGenDataset(
        test_data_dir, scaled=True, scaler=scaler, feature_name_dir=feature_name_dir
    )

    test_loader = DataLoader(
        test_data,
        batch_size=config["batch_size"],
        num_workers=num_workers,
        shuffle=False,
    )

    model = Desc2MOF.load_from_checkpoint(ckpt_dir, config=config, strict=False)
    trainer.predict(model, dataloaders=test_loader)
