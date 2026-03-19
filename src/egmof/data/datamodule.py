from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Type, Union

import numpy as np
from lightning import LightningDataModule
from torch.utils.data import DataLoader

from .dataset import CSVDataset, TextSplitDataset, JsonSplitDataset


DatasetCls = Type[Union[CSVDataset, TextSplitDataset, JsonSplitDataset]]


class Datamodule(LightningDataModule):
    """Lightning DataModule for descriptor/target datasets stored on disk."""

    def __init__(
        self,
        path: str | Path,
        batch_size: int,
        num_workers: int,
        dataset_cls: DatasetCls,
        task: Optional[str] = None,
        target: Optional[str] = None,
        pin_memory: bool = True,
        persistent_workers: Optional[bool] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["dataset_cls"])

        self.path = Path(path)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_cls = dataset_cls
        self.task = task
        self.target = target

        self.pin_memory = pin_memory
        if persistent_workers is None:
            self.persistent_workers = num_workers > 0
        else:
            self.persistent_workers = persistent_workers

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.collate_fn = None

    def set_train_dataset(self):
        self.train_dataset = self.dataset_cls(
            direc=self.path,
            split='train',
            task=self.task,
            target=self.target
        )

    def set_val_dataset(self):
        self.val_dataset = self.dataset_cls(
            direc=self.path,
            split='val',
            task=self.task,
            target=self.target
        )

    def set_test_dataset(self):
        self.test_dataset = self.dataset_cls(
            direc=self.path,
            split='test',
            task=self.task,
            target=self.target
        )

    def prepare_data(self) -> None:
        """Hook for downloading/creating data (no-op for local datasets)."""
        return None

    def setup(self, stage: Optional[str] = None) -> None:
        """Create datasets for the given stage ('fit'/'validate'/'test')."""
        if stage in (None, "fit"):
            self.set_train_dataset()
            self.set_val_dataset()

        if stage in (None, "validate"):
            self.set_val_dataset()

        if stage in (None, "test"):
            self.set_test_dataset()

        self.collate_fn = getattr(self.dataset_cls, "collate", None)
        if self.collate_fn is None and self.train_dataset is not None:
            self.collate_fn = getattr(self.train_dataset, "collate", None)
        if self.collate_fn is None:
            raise AttributeError(f"{self.dataset_cls.__name__} must define a staticmethod `collate(batch)`.")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def train_array(self) -> np.ndarray:
        """Return one training batch as numpy arrays."""
        x_batch, y_batch = next(iter(self.train_dataloader()))
        return x_batch.detach().cpu().numpy(), y_batch.detach().cpu().numpy()

    def val_array(self) -> np.ndarray:
        """Return one validation batch as numpy arrays."""
        x_batch, y_batch = next(iter(self.val_dataloader()))
        return x_batch.detach().cpu().numpy(), y_batch.detach().cpu().numpy()

    def test_array(self) -> np.ndarray:
        """Return one test batch as numpy arrays."""
        x_batch, y_batch = next(iter(self.test_dataloader()))
        return x_batch.detach().cpu().numpy(), y_batch.detach().cpu().numpy()

    def get_mean_and_std(self) -> Dict[str, Any]:
        """Compute mean/std from the training dataset."""
        return self.train_dataset.get_mean_and_std()
    
    def get_min_and_max(self) -> Dict[str, Any]:
        """Compute min/max from the training dataset."""
        return self.train_dataset.get_min_and_max()