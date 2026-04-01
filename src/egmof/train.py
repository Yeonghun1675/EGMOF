import os
from pathlib import Path
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from lightning.pytorch import Trainer, seed_everything
import lightning as pl
from lightning.pytorch import callbacks as pl_callbacks

from .desc2mof import Scaler, Desc2MOF as Desc2MOFModel
from .mof2desc import MOF2Desc as MOF2DescModel
from .mof2desc.model.dataset import CSVDataset as MOF2DescCSVDataset
from .desc2mof.dataset import CSVDataset as Desc2MOFCSVDataset
from .egmof import DEFAULT_DESC2MOF_FEATURE_NAME


def create_desc2mof_dataloaders(
    config: dict,
    scaler: Scaler,
    train_dir: str | Path,
    val_dir: str | Path,
    test_dir: str | Path,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create desc2mof dataloaders from config.

    Args:
        config: Config dict with max_token_len, batch_size, num_workers
        scaler: Scaler for descriptor normalization
        train_dir: Path to training CSV
        val_dir: Path to validation CSV
        test_dir: Path to test CSV

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    feature_name_dir = config.get("feature_name_dir", DEFAULT_DESC2MOF_FEATURE_NAME)

    train_data = Desc2MOFCSVDataset(
        train_dir,
        max_len=config.get("max_token_len", 512),
        scaled=True,
        scaler=scaler,
        feature_name_dir=feature_name_dir,
    )
    val_data = Desc2MOFCSVDataset(
        val_dir,
        max_len=config.get("max_token_len", 512),
        scaled=True,
        scaler=scaler,
        feature_name_dir=feature_name_dir,
    )
    test_data = Desc2MOFCSVDataset(
        test_dir,
        max_len=config.get("max_token_len", 512),
        scaled=True,
        scaler=scaler,
        feature_name_dir=feature_name_dir,
    )

    train_loader = DataLoader(
        train_data,
        batch_size=config.get("batch_size", 256),
        num_workers=config.get("num_workers", 4),
        shuffle=True,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=config.get("batch_size", 256),
        num_workers=config.get("num_workers", 4),
        shuffle=False,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=config.get("batch_size", 256),
        num_workers=config.get("num_workers", 4),
        shuffle=False,
    )

    return train_loader, val_loader, test_loader


def create_mof2desc_dataloaders(
    config: dict,
    scaler: Scaler,
    train_dir: str | Path,
    valid_dir: str | Path,
    test_dir: str | Path,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create mof2desc dataloaders from config.

    Args:
        config: Config dict with batch_size, num_workers
        scaler: Scaler for descriptor normalization
        train_dir: Path to training CSV
        valid_dir: Path to validation CSV
        test_dir: Path to test CSV

    Returns:
        Tuple of (train_loader, valid_loader, test_loader)
    """
    feature_name_dir = config.get("feature_name_dir", DEFAULT_DESC2MOF_FEATURE_NAME)

    train_data = MOF2DescCSVDataset(
        train_dir,
        scaled=True,
        scaler=scaler,
        feature_name_dir=feature_name_dir,
    )
    valid_data = MOF2DescCSVDataset(
        valid_dir,
        scaled=True,
        scaler=scaler,
        feature_name_dir=feature_name_dir,
    )
    test_data = MOF2DescCSVDataset(
        test_dir,
        scaled=True,
        scaler=scaler,
        feature_name_dir=feature_name_dir,
    )

    train_loader = DataLoader(
        train_data,
        batch_size=config.get("batch_size", 256),
        num_workers=config.get("num_workers", 4),
        shuffle=True,
    )
    valid_loader = DataLoader(
        valid_data,
        batch_size=config.get("batch_size", 256),
        num_workers=config.get("num_workers", 4),
        shuffle=False,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=config.get("batch_size", 256),
        num_workers=config.get("num_workers", 4),
        shuffle=False,
    )

    return train_loader, valid_loader, test_loader


def train_desc2mof(
    config: dict,
    scaler: Scaler,
    train_dir: str | Path,
    val_dir: str | Path,
    test_dir: str | Path,
    accelerator: str = "gpu",
    devices: int = 1,
    max_epochs: int = 200,
    log_dir: str = "./logs_desc2mof",
    ckpt_dir: str = "./ckpt_desc2mof",
) -> Desc2MOFModel:
    """Train desc2mof model.

    Args:
        config: Config dict
        scaler: Scaler for descriptor normalization
        train_dir: Path to training CSV
        val_dir: Path to validation CSV
        test_dir: Path to test CSV
        accelerator: 'cpu' or 'gpu'
        devices: Number of devices
        max_epochs: Max training epochs
        log_dir: Directory for logs
        ckpt_dir: Directory for checkpoints

    Returns:
        Trained Desc2MOFModel
    """
    seed_everything(config.get("seed", 42), workers=True)

    train_loader, val_loader, test_loader = create_desc2mof_dataloaders(
        config, scaler, train_dir, val_dir, test_dir
    )

    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=ckpt_dir,
        verbose=True,
        save_last=True,
        save_top_k=1,
        monitor="val/avg_val_loss",
        mode="min",
    )
    lr_callback = pl.callbacks.LearningRateMonitor()
    callbacks = [checkpoint_callback, lr_callback]

    os.makedirs(log_dir, exist_ok=True)
    logger = pl.loggers.TensorBoardLogger(log_dir, name=f"desc2mof")

    strategy = "ddp_find_unused_parameters_true"

    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        num_nodes=config.get("num_nodes", 1),
        max_epochs=max_epochs,
        logger=logger,
        benchmark=True,
        strategy=strategy,
        log_every_n_steps=10,
        callbacks=callbacks,
    )

    model = Desc2MOFModel(config)
    trainer.fit(model, train_loader, val_loader)

    return model


def train_mof2desc(
    config: dict,
    scaler: Scaler,
    train_dir: str | Path,
    valid_dir: str | Path,
    test_dir: str | Path,
    accelerator: str = "gpu",
    devices: int = 1,
    max_epochs: int = 500,
    log_dir: str = "./logs_mof2desc",
    ckpt_dir: str = "./ckpt_mof2desc",
) -> MOF2DescModel:
    """Train mof2desc model.

    Args:
        config: Config dict
        scaler: Scaler for descriptor normalization
        train_dir: Path to training CSV
        valid_dir: Path to validation CSV
        test_dir: Path to test CSV
        accelerator: 'cpu' or 'gpu'
        devices: Number of devices
        max_epochs: Max training epochs
        log_dir: Directory for logs
        ckpt_dir: Directory for checkpoints

    Returns:
        Trained MOF2DescModel
    """
    seed_everything(config.get("seed", 42), workers=True)

    train_loader, valid_loader, test_loader = create_mof2desc_dataloaders(
        config, scaler, train_dir, valid_dir, test_dir
    )

    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=ckpt_dir,
        verbose=True,
        save_last=True,
        save_top_k=1,
        monitor="val/avg_val_loss",
        mode="min",
    )
    lr_callback = pl.callbacks.LearningRateMonitor()
    callbacks = [checkpoint_callback, lr_callback]

    os.makedirs(log_dir, exist_ok=True)
    logger = pl.loggers.TensorBoardLogger(log_dir, name=f"mof2desc")

    strategy = "ddp_find_unused_parameters_true"

    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        num_nodes=config.get("num_nodes", 1),
        max_epochs=max_epochs,
        logger=logger,
        benchmark=True,
        strategy=strategy,
        log_every_n_steps=10,
        callbacks=callbacks,
    )

    model = MOF2DescModel(config, scaler)
    trainer.fit(model, train_loader, valid_loader)

    return model
