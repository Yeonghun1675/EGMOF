from __future__ import annotations

import os
from typing import Any, Dict, List, Literal, Optional, Union
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import selfies
import torch
import torch.nn.functional as F
import yaml
from omegaconf import OmegaConf
from lightning.pytorch import Trainer, seed_everything
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from egmof import mof2desc

from .prop2desc import (
    Prop2Desc,
    run_train_prop2desc,
)

from .desc2mof import (
    Desc2MOF as Desc2MOFModel,
    MOFGenDataset,
    Scaler,
    MOF_ENCODE_DICT,
    MOF_DECODE_DICT,
    SOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    SEP_TOKEN,
    CN_IDS,
    decode_token2mof,
    __desc2mof_dir__,
)

from . import __root_dir__
from .data import Datamodule
from .data.dataset import CSVDataset, TextSplitDataset, JsonSplitDataset



# TODO: Implement only useful methods for users (load, train, generate, save). Others should be move to other python files.
class EGMOF:
    """Orchestrator that wires configs → model/datamodule/trainer."""
    def __init__(
        self,
        prop2desc_ckpt_path: Optional[str | Path] = None,
        prop2desc_config_path: Optional[str | Path] = None,
        load_pretrained_modules: bool = True,
        accelerator: Literal["cpu", "cuda"] = "cpu",
        devices: int | List[int] = 1,
    ) -> None:
        self.prop2desc_ckpt_path: Optional[str | Path] = prop2desc_ckpt_path
        self.prop2desc_config_path: Optional[str | Path] = prop2desc_config_path
        self.load_pretrained_modules: bool = load_pretrained_modules

        self.prop2desc: Optional[Prop2Desc] = None
        self.desc2mof: Optional[Desc2MOFModel] = None
        self.mof2desc: Optional[object] =  None   # TODO: Implement Type
        self.sk_model: Optional[object] = None    # TODO: Implement Type
        self.setup()


    def setup(self):
        if self.load_pretrained_modules:
            pass
            # TODO: Implement download pretrained ckpt files
            
            # self.load_desc2mof(
            #     ckpt_path=self.desc2mof_ckpt_path,
            #     config_path=self.desc2mof_config_path,
            # )
            # self.load_mof2desc(
            #     ckpt_path=self.mof2desc_ckpt_path,
            #     config_path=self.mof2desc_config_path,
            # )

        if self.prop2desc_ckpt_path:
            self.load_prop2desc(
                ckpt_path=self.prop2desc_ckpt_path,
                config_path=self.prop2desc_config_path,
            )
    
    def load_prop2desc(self, ckpt_path: str | Path, config_path: str | Path) -> None:
        self.prop2desc = Prop2Desc.load(ckpt_path=ckpt_path, config_path=config_path).to(device=self.accelerator)

    # TODO: Implement loading desc2mof model. (details is not in here, but in desc2mof/model.py)
    def load_desc2mof(self, ckpt_path: str | Path, config_path: str | Path) -> None:
        raise NotImplementedError("Loading desc2mof is not implemented yet.")

    # TODO: Implement loading mof2desc model. (details is not in here, but in mof2desc/model.py)
    def load_mof2desc(self, ckpt_path: str | Path, config_path: str | Path) -> None:
        raise NotImplementedError("Loading mof2desc is not implemented yet.")
    
    def train(
        self,
        prop2desc_config_path: Optional[str | Path] = None,
        desc2mof_config_path: Optional[str | Path] = None,
        mof2desc_config_path: Optional[str | Path] = None,
    ):
        # TODO: Train prop2desc model.
        if self.prop2desc is None:
            self.train_prop2desc(prop2desc_config_path)
        if self.desc2mof is None:
            self.train_desc2mof(desc2mof_config_path)
        if self.mof2desc is None:
            self.train_mof2desc(mof2desc_config_path)

    def train_prop2desc(
        self, 
        data_path: str | Path, 
        target_property: Optional[str] = None, 
        prop2desc_config_path: Optional[str | Path] = None,
    ) -> Prop2Desc:
        """ train prop2desc model """
        self.prop2desc = run_train_prop2desc(
            config_path=prop2desc_config_path,
            data_path=data_path,
            target=target_property,
        )
        return self.prop2desc

    def train_desc2mof(self, desc2mof_config_path: str | Path):
        # TODO: Train desc2mof model. (make desc2mof/train.py and call it here.)
        raise NotImplementedError("Training desc2mof is not implemented yet.")

    def train_mof2desc(self, mof2desc_config_path: str | Path):
        # TODO: Train mof2desc model. (make mof2desc/train.py and call it here.)
        raise NotImplementedError("Training mof2desc is not implemented yet.")

    def train_sk_model(self, sk_model_config_path: str | Path):
        # TODO: Train sklearn model. (make train_sklearn.py and call it here.)
        raise NotImplementedError("Training sklearn model is not implemented yet.")

    def generate(
        self,
        num_samples: int = 100,
        target_value: Optional[float | int] = None,  
        save_descriptor_path: Optional[str] = None,
    ):
        for model in ["prop2desc"]: # , "desc2mof", "mof2desc", "sk_model"]:
            if not getattr(self, model):
                raise ValueError(f"{model} model is not trained yet. Please load or train the model first.")

        desc_tensor = self.prop2desc.sample(
            num_samples = num_samples,
            target = target_value,
        )

        if len(desc_tensor.shape) == 3:
            desc_tensor = desc_tensor.squeeze(1)

        if save_descriptor_path:
            desc_df = pd.DataFrame(desc_tensor.detach().cpu().numpy())
            desc_df.to_csv(save_descriptor_path, index=False)
            print(f"Descriptors saved to {save_descriptor_path}")

        # TODO: Generate descriptors using mof2desc model. (make mof2desc/generate.py and call it here.)
        pass

        # TODO: Generate descriptors using sklearn model. (make train_sklearn.py and call it here.)
        pass

        # TODO: Generate descriptors using prop2desc model. (make prop2desc/generate.py and call it here.)
        pass

        raise NotImplementedError("Generating descriptors is not implemented yet.")
