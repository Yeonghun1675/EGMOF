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

from .prop2desc import Prop2Desc
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


# Default checkpoint paths
DEFAULT_DESC2MOF_CKPT = os.path.join(
    __root_dir__, "checkpoints", "desc2mof", "desc2mof_best.ckpt"
)
DEFAULT_MOF2DESC_CKPT = os.path.join(
    __root_dir__, "checkpoints", "mof2desc", "mof2desc_best.ckpt"
)

# Default desc2mof paths  # TODO: Not in main python file. Move to desc2mof module?
DEFAULT_DESC2MOF_MEAN = os.path.join(__desc2mof_dir__, "data", "mean_all.csv")
DEFAULT_DESC2MOF_STD = os.path.join(__desc2mof_dir__, "data", "std_all.csv")

DEFAULT_DESC2MOF_FEATURE_NAME = os.path.join(
    __desc2mof_dir__, "data", "feature_name.txt"
)
DEFAULT_DESC2MOF_CONFIG = os.path.join(
    __root_dir__, "..", "config", "desc2mof_training_config.yaml"
)
DEFAULT_MOF2DESC_CONFIG = os.path.join(
    __root_dir__, "..", "config", "mof2desc_training_config.yaml"
)

# TODO: Move to utils
def _build_mofgen_dataset(
    desc_tensor: torch.Tensor, scaler: Scaler, feature_name_dir: str
):
    """Build MOFGenDataset from descriptor tensor [N, D]."""
    desc_np = desc_tensor.detach().cpu().numpy()
    df = pd.DataFrame(desc_np, columns=None)
    return MOFGenDataset(
        df, scaled=True, scaler=scaler, feature_name_dir=feature_name_dir
    )

# TODO: Move to utils
def _parse_mof_output(all_output: List, SEP_TOKEN_ID: int):
    """Parse MOF token output, returns (valid_mask, log_list)."""
    from .desc2mof.utils import is_valid as _is_valid

    return _is_valid(all_output, SEP_TOKEN_ID)

# TODO: Move to utils
def cal_wmse(
    pred_desc: torch.Tensor, target_desc: torch.Tensor, weights
) -> torch.Tensor:
    """Calculate weighted MSE between predicted and target descriptors."""
    mse = F.mse_loss(pred_desc, target_desc, reduction="none")
    w = torch.tensor(weights, dtype=mse.dtype, device=mse.device)
    weighted_mse = mse * w
    wmse = weighted_mse.sum(dim=-1) / w.sum()
    return wmse

# TODO: Move to utils
def sk_predict(desc_pred: np.ndarray, sk_model, sk_scaler) -> np.ndarray:
    """Predict property from descriptor using sklearn model."""
    desc_scaled = sk_scaler.encode(desc_pred)
    pred_scaled = sk_model.predict(desc_scaled)
    return sk_scaler.decode_target(pred_scaled)


class EGMOF:
    """Orchestrator that wires configs → model/datamodule/trainer."""

    def __init__(
        self,
        target: Optional[str] = None,
        data_path: Optional[str | Path] = None,
        prop2desc_model_config_path: Optional[str | Path] = None,
        prop2desc_training_config_path: Optional[str | Path] = None,
        desc2mof_ckpt_dir: Optional[str | Path] = DEFAULT_DESC2MOF_CKPT,
        desc2mof_config_path: Optional[str | Path] = DEFAULT_DESC2MOF_CONFIG,
        mof2desc_ckpt_dir: Optional[str | Path] = DEFAULT_MOF2DESC_CKPT,
        mof2desc_config_path: Optional[str | Path] = DEFAULT_MOF2DESC_CONFIG,
        skmodel_ckpt_dir: Optional[str | Path] = None,   # TODO: why ckpt for sklearn?
        skmodel_mean_std_dir: Optional[str | Path] = None,   # TODO: mean_std should be same as other modelss
        desc2mof_mean_dir: Optional[str | Path] = DEFAULT_DESC2MOF_MEAN,
        desc2mof_std_dir: Optional[str | Path] = DEFAULT_DESC2MOF_STD,
        desc2mof_feature_name_dir: Optional[str | Path] = DEFAULT_DESC2MOF_FEATURE_NAME,
        desc2mof_feature_size: int = 183,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.target = target
        self.data_path = Path(data_path) if data_path else None
        self.prop2desc_model_config_path = prop2desc_model_config_path
        self.prop2desc_training_config_path = prop2desc_training_config_path
        self.overrides = overrides or {}

        self.cfg = self._load_config()
        self.prop2desc: Optional[Prop2Desc] = None
        self.datamodule: Optional[Datamodule] = None
        self.trainer: Optional[Trainer] = None

        self.desc2mof_ckpt_dir = desc2mof_ckpt_dir
        self.desc2mof_config_path = desc2mof_config_path
        self.mof2desc_ckpt_dir = mof2desc_ckpt_dir
        self.mof2desc_config_path = mof2desc_config_path
        self.skmodel_ckpt_dir = skmodel_ckpt_dir
        self.skmodel_mean_std_dir = skmodel_mean_std_dir
        self.desc2mof_mean_dir = desc2mof_mean_dir
        self.desc2mof_std_dir = desc2mof_std_dir
        self.desc2mof_feature_name_dir = desc2mof_feature_name_dir
        self.desc2mof_feature_size = desc2mof_feature_size

        self._desc2mof_config: Optional[Dict] = None
        self._desc2mof_scaler: Optional[Scaler] = None  # TODO: unified scaler? there are too many scalers at once. (But if we change scaler, then previous ckpt will be invalidated.)
        self._desc2mof_feature_names: Optional[List[str]] = None
        self._mof2desc_model = None
        self._sk_model = None
        self._sk_scaler = None
        self._sk_feature_importances = None
        self._topo_cn_dict = None
        self._bb_cn_dict = None

    def _load_config(self):
        cfgs = []
        if self.prop2desc_model_config_path:
            cfgs.append(OmegaConf.load(self.prop2desc_model_config_path))
        if self.prop2desc_training_config_path:
            cfgs.append(OmegaConf.load(self.prop2desc_training_config_path))
        if cfgs:
            cfg = OmegaConf.merge(*cfgs)
        else:
            cfg = OmegaConf.create({})

        if self.overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.create(self.overrides))
        return cfg

    def _dataset_cls_from_name(self, name: str):
        name = (name or "csv").lower()
        if name == "csv":
            return CSVDataset
        if name in ("text", "txt"):
            return TextSplitDataset
        if name == "json":
            return JsonSplitDataset
        raise ValueError(f"Unknown dataset_cls: {name}. Use one of [csv, text, json].")

    def build_datamodule(self) -> Datamodule:
        if self.data_path is None:
            raise ValueError(
                "data_path is required for training. Set data_path or use generate() with existing descriptors."
            )
        dm_cfg = self.cfg.get("datamodule", {})
        dataset_cls = self._dataset_cls_from_name(dm_cfg.get("dataset_cls", "csv"))
        self.datamodule = Datamodule(
            path=self.data_path,
            batch_size=int(dm_cfg.get("batch_size", 64)),
            num_workers=int(dm_cfg.get("num_workers", 4)),
            dataset_cls=dataset_cls,
            task=dm_cfg.get("task", None),
            target=self.target,
        )
        return self.datamodule

    def build_model(self, scaler_value: Optional[Dict[str, Any]] = None) -> Prop2Desc:
        m_cfg = self.cfg.get("model", {})
        if scaler_value is None:
            scaler_value = m_cfg.get("scaler_value", None)

        self.prop2desc = Prop2Desc(
            in_channels=int(m_cfg["in_channels"]),
            timestep=int(m_cfg.get("timestep", 1000)),
            lr=float(m_cfg.get("lr", 1e-4)),
            dim=int(m_cfg.get("dim", m_cfg["in_channels"])),
            dim_mults=list(m_cfg.get("dim_mults", [1, 2])),
            condition=m_cfg.get("condition", None),
            out_channels=m_cfg.get("out_channels", None),
            num_classes=int(m_cfg.get("num_classes", 0)),
            cond_dim=int(m_cfg.get("cond_dim", 0)),
            scaler_mode=m_cfg.get("scaler_mode", "standard"),
            scaler_value=scaler_value,
        )
        return self.prop2desc

    def build_trainer(self) -> Trainer:
        t_cfg = self.cfg.get("trainer", {})
        self.trainer = Trainer(**OmegaConf.to_container(t_cfg, resolve=True))
        return self.trainer

    def train(self):
        """Train the EGMOF model"""
        seed = int(self.cfg.get("seed", 42))
        seed_everything(seed, workers=True)

        dm = self.build_datamodule()
        dm.setup("fit")

        model_cfg = self.cfg.get("model", {})
        scaler_value = model_cfg.get("scaler_value", None)
        if scaler_value is None:
            scaler_mode = model_cfg.get("scaler_mode", "standard")
            if scaler_mode == "standard":
                scaler_value = dm.get_mean_and_std()
            elif scaler_mode == "minmax":
                scaler_value = dm.get_min_and_max()
            else:
                raise ValueError(f"Unknown scaler_mode: {scaler_mode}")

        model = self.build_model(scaler_value=scaler_value)
        trainer = self.build_trainer()
        trainer.fit(model, datamodule=dm)

    def _load_desc2mof_config(self) -> Dict:
        """Load desc2mof config from YAML."""
        if self._desc2mof_config is not None:
            return self._desc2mof_config
        cfg_path = str(self.desc2mof_config_path) if self.desc2mof_config_path else None
        if cfg_path:
            with open(cfg_path, "r") as f:
                self._desc2mof_config = yaml.safe_load(f)
        else:
            self._desc2mof_config = {}
        assert self._desc2mof_config is not None
        self._desc2mof_config["feature_size"] = self.desc2mof_feature_size
        return self._desc2mof_config

    def _load_desc2mof_scaler(self) -> Scaler:
        """Load desc2mof scaler (mean/std normalization)."""
        if self._desc2mof_scaler is not None:
            return self._desc2mof_scaler
        feat_path = (
            str(self.desc2mof_feature_name_dir)
            if self.desc2mof_feature_name_dir
            else None
        )
        if feat_path:
            with open(feat_path, "r") as f:
                self._desc2mof_feature_names = [line.strip() for line in f.readlines()]
        mean_path = str(self.desc2mof_mean_dir)
        std_path = str(self.desc2mof_std_dir)
        mean = pd.read_csv(mean_path)[self._desc2mof_feature_names or []]
        std = pd.read_csv(std_path)[self._desc2mof_feature_names or []]
        self._desc2mof_scaler = Scaler(
            np.array(mean).squeeze(), np.array(std).squeeze(), 0, 1
        )
        return self._desc2mof_scaler

    def _load_mof_topo_cn_dict(self) -> Dict:
        """Load topology-CN dictionary for validation."""
        import pickle

        if self._topo_cn_dict is not None:
            return self._topo_cn_dict
        topo_path = Path(__desc2mof_dir__) / "data" / "mof_topo_cn_dict.pkl"
        with open(topo_path, "rb") as f:
            mof_topo_cn_dict = pickle.load(f)
        self._topo_cn_dict = {
            MOF_ENCODE_DICT[k]: v for k, v in mof_topo_cn_dict.items()
        }
        return self._topo_cn_dict

    def _load_bb_cn_dict(self) -> Dict:
        """Load building block-CN dictionary for validation."""
        import pickle

        if self._bb_cn_dict is not None:
            return self._bb_cn_dict
        bb_path = Path(__desc2mof_dir__) / "data" / "bb_cn_dict.pkl"
        with open(bb_path, "rb") as f:
            bbname_cn_dict = pickle.load(f)
        self._bb_cn_dict = {
            MOF_ENCODE_DICT[k]: v
            for k, v in bbname_cn_dict.items()
            if k in MOF_ENCODE_DICT
        }
        return self._bb_cn_dict

    def load(self):
        """Load prop2desc and desc2mof models."""
        prop2desc_ckpt = self.cfg.get("model", {}).get("ckpt_path")
        if prop2desc_ckpt:
            self.prop2desc = Prop2Desc.load_from_checkpoint(prop2desc_ckpt)
        if self.desc2mof_ckpt_dir:
            desc2mof_path = Path(self.desc2mof_ckpt_dir)
            if not desc2mof_path.exists():
                raise FileNotFoundError(
                    f"desc2mof checkpoint not found: {desc2mof_path}\n"
                    "Please download from: https://zenodo.org/records/your-record-id"
                )
            self._desc2mof_config = self._load_desc2mof_config()
            self._desc2mof_scaler = self._load_desc2mof_scaler()
        if self.mof2desc_ckpt_dir:
            mof2desc_path = Path(self.mof2desc_ckpt_dir)
            if not mof2desc_path.exists():
                raise FileNotFoundError(
                    f"mof2desc checkpoint not found: {mof2desc_path}\n"
                    "Please download from: https://zenodo.org/records/your-record-id"
                )
            self._mof2desc_model = self._load_mof2desc()
            self._load_sk_model()

    def _load_mof2desc(self):
        """Load mof2desc model (EGMOF migration pending)."""
        import sys
        from types import ModuleType

        easy_adapt_mof = ModuleType("easy_adapt_mof")
        easy_adapt_mof.mof2desc = ModuleType("easy_adapt_mof.mof2desc")
        sys.modules["easy_adapt_mof"] = easy_adapt_mof
        sys.modules["easy_adapt_mof.mof2desc"] = easy_adapt_mof.mof2desc

        try:
            from .mof2desc import MOF2Desc as MOF2DescModel
            from .mof2desc.model.dataset import Scaler

            easy_adapt_mof.mof2desc.model = ModuleType("easy_adapt_mof.mof2desc.model")
            sys.modules["easy_adapt_mof.mof2desc.model"] = easy_adapt_mof.mof2desc.model
            easy_adapt_mof.mof2desc.model.dataset = ModuleType(
                "easy_adapt_mof.mof2desc.model.dataset"
            )
            sys.modules["easy_adapt_mof.mof2desc.model.dataset"] = (
                easy_adapt_mof.mof2desc.model.dataset
            )
            easy_adapt_mof.mof2desc.model.dataset.Scaler = Scaler
        except ImportError:
            raise ImportError(
                "mof2desc module not found in egmof. "
                "Please migrate mof2desc to src/egmof/mof2desc/ first."
            )
        mof2desc_cfg_path = (
            str(self.mof2desc_config_path) if self.mof2desc_config_path else None
        )
        if mof2desc_cfg_path:
            with open(mof2desc_cfg_path, "r") as f:
                cfg = yaml.safe_load(f)
        else:
            cfg = {}
        scaler = self._load_desc2mof_scaler()
        ckpt = str(self.mof2desc_ckpt_dir) if self.mof2desc_ckpt_dir else None
        return MOF2DescModel.load_from_checkpoint(
            ckpt, config=cfg, scaler=scaler, strict=False, weights_only=False
        )

    def _load_sk_scaler(self) -> tuple[Scaler, np.ndarray | None]:
        path = str(self.skmodel_mean_std_dir)
        if path.endswith(".json"):
            import json

            with open(path, "r") as f:
                yaml_data = json.load(f)
        elif path.endswith(".yaml"):
            with open(path, "r") as f:
                yaml_data = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported skmodel_mean_std format: {path}")

        feature_importances = yaml_data.get("feature_importances", None)
        scaler_dict = yaml_data.get("scaler_value", yaml_data)

        scaler = Scaler(
            scaler_dict["mean"],
            scaler_dict["std"],
            scaler_dict["target_mean"],
            scaler_dict["target_std"],
        )

        return scaler, feature_importances

    def _load_sk_model(self):
        if self.skmodel_ckpt_dir:
            self._sk_model = joblib.load(self.skmodel_ckpt_dir)
            self._sk_feature_importances = self._sk_model.feature_importances_.tolist()

        if self.skmodel_mean_std_dir:
            self._sk_scaler, fi_from_yaml = self._load_sk_scaler()
            if self._sk_feature_importances is None:
                self._sk_feature_importances = fi_from_yaml

    def _is_valid(self, all_output: List, SEP_TOKEN_ID: int):
        """Validate generated MOF tokens against topology/building block constraints."""
        topo_cn_dict = self._load_mof_topo_cn_dict()
        bb_cn_dict = self._load_bb_cn_dict()
        LR_TOKEN = MOF_ENCODE_DICT["[Lr]"]

        def split_list(lst, sep):
            grouped, temp = [], []
            for val in lst:
                if val == sep:
                    grouped.append(temp)
                    temp = []
                else:
                    temp.append(val)
            grouped.append(temp)
            return grouped

        results, log_list, correct = [], [], 0
        for res in all_output:
            seq = [t for t in res if t not in [EOS_TOKEN, PAD_TOKEN]]
            if not seq:
                results.append(False)
                log_list.append(["Not sequence"])
                continue

            topo_id = seq[0]
            if topo_id not in topo_cn_dict:
                results.append(False)
                log_list.append([f"Invalid Topology ID: {topo_id}"])
                continue

            node_cns_arr, edge_count = topo_cn_dict[topo_id]
            expected_cns = [int(x) for x in node_cns_arr] + [2] * int(edge_count)

            body_seq = seq[1:]
            raw_segments = split_list(body_seq, SEP_TOKEN_ID)
            clean_segments = [
                [t for t in seg if t not in CN_IDS] for seg in raw_segments
            ]
            clean_segments = [s for s in clean_segments if s]

            if len(clean_segments) != len(expected_cns):
                results.append(False)
                log_list.append(
                    [
                        f"Segment mismatch: expected {len(expected_cns)}, got {len(clean_segments)}"
                    ]
                )
                continue

            segment_check_fail, fail_reason = False, ""
            for idx, (segment, exp_cn) in enumerate(zip(clean_segments, expected_cns)):
                if len(segment) == 1:
                    token = segment[0]
                    if token not in bb_cn_dict:
                        segment_check_fail = True
                        fail_reason = f"Unknown Single Token at idx {idx}"
                        break
                    if bb_cn_dict[token] != exp_cn:
                        segment_check_fail = True
                        fail_reason = f"CN mismatch at idx {idx}"
                        break
                else:
                    actual_lr_count = segment.count(LR_TOKEN)
                    if actual_lr_count != exp_cn:
                        segment_check_fail = True
                        fail_reason = f"Lr count mismatch at idx {idx}"
                        break
                    try:
                        selfies_str = "".join(
                            [MOF_DECODE_DICT.get(s, str(s)) for s in segment]
                        )
                        if selfies_str != selfies.encoder(selfies.decoder(selfies_str)):
                            segment_check_fail = True
                            fail_reason = f"Invalid SELFIES at {idx}"
                    except Exception as e:
                        segment_check_fail = True
                        fail_reason = f"SELFIES ERROR at {idx}: {e}"

            if segment_check_fail:
                results.append(False)
                log_list.append([fail_reason])
            else:
                results.append(True)
                correct += 1

        acc = correct / len(all_output) if all_output else 0
        print(f"accuracy: {acc:.4f}, correct: {correct}, total: {len(all_output)}")
        return np.array(results), log_list

    def _run_desc2mof(
        self,
        target_desc: torch.Tensor,
        topk: int = 5,
        temperature: float = 1.0,
        use_gnmt_length_penalty: bool = False,
        length_penalty_alpha: float = 0.6,
        batch_size: int = 256,
        num_workers: int = 0,
        device: str = "cuda",
    ):
        """Run desc2mof inference: descriptor → MOF tokens (beam search)."""
        desc2mof = Desc2MOFModel.load_from_checkpoint(
            self.desc2mof_ckpt_dir,
            config=self._load_desc2mof_config(),
            strict=False,
            weights_only=False,
        ).to(device)
        desc2mof.eval()

        scaler = self._load_desc2mof_scaler()
        feature_names = self._desc2mof_feature_names or []
        feature_name_dir = (
            str(self.desc2mof_feature_name_dir)
            if self.desc2mof_feature_name_dir
            else str(Path(__desc2mof_dir__) / "data" / "feature_name.txt")
        )

        desc_np = target_desc.detach().cpu().numpy()
        target_df = pd.DataFrame(desc_np, columns=feature_names)
        target_data = MOFGenDataset(
            target_df, scaled=True, scaler=scaler, feature_name_dir=feature_name_dir
        )
        target_loader = DataLoader(
            target_data, batch_size=batch_size, num_workers=num_workers, shuffle=False
        )

        preds_output = []
        for batch in tqdm(target_loader, desc="desc2mof inference"):
            with torch.no_grad():
                desc_batch, _ = batch
                desc_batch = desc_batch.to(device)
                preds, _, _, _ = desc2mof.model.generate_beam(
                    desc_batch,
                    temperature=temperature,
                    beam_width=topk,
                    use_gnmt_length_penalty=use_gnmt_length_penalty,
                    length_penalty_alpha=length_penalty_alpha,
                )
                preds_output.append(preds)
        preds_output = torch.concat(preds_output)
        all_output = (
            preds_output.reshape(-1, preds_output.shape[-1]).detach().cpu().tolist()
        )
        mof_names = decode_token2mof(all_output)
        return all_output, mof_names, target_data

    def _run_mof2desc_and_select(
        self,
        all_output: List,
        target_data: MOFGenDataset,
        topk: int,
        wmse_target: float,
        batch_size: int = 256,
        num_workers: int = 0,
        device: str = "cuda",
    ):
        """Run mof2desc, compute wmse, and select best MOFs."""
        try:
            from .mof2desc import MOF2Desc as MOF2DescModel, Desc2MOFOutputDataset
        except ImportError:
            raise ImportError(
                "mof2desc module not found. Please migrate to src/egmof/mof2desc/ first."
            )

        mof2desc = self._mof2desc_model
        if mof2desc is None:
            raise RuntimeError("mof2desc model not loaded. Call load() first.")
        mof2desc = mof2desc.to(device)
        mof2desc.eval()

        weights = self._sk_feature_importances
        if weights is None:
            raise RuntimeError(
                "sk feature_importances not loaded. Add 'feature_importances' to scaler YAML."
            )

        valid_mask, log_list = self._is_valid(all_output, SEP_TOKEN_ID=SEP_TOKEN)
        any_valid_mask = valid_mask.reshape(-1, topk).any(axis=1)

        x_np = np.array(target_data.x.squeeze(-1))
        x_tensor = (
            torch.from_numpy(x_np)
            .unsqueeze(1)
            .repeat(1, topk, 1)
            .view(-1, self.desc2mof_feature_size)
        )
        mof2desc_dataset = Desc2MOFOutputDataset(all_output, x_tensor, scaled=False)
        mof2desc_loader = DataLoader(
            mof2desc_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
        )

        wmse_output, mof2desc_preds = [], []
        for batch in tqdm(mof2desc_loader, desc="mof2desc inference"):
            with torch.no_grad():
                target_desc, token_ids, attention_mask = batch
                token_ids = token_ids.to(device)
                attention_mask = attention_mask.to(device)
                target_desc = target_desc.to(device)
                pred_desc = mof2desc(token_ids, attention_mask)
                wmse = cal_wmse(pred_desc, target_desc, weights)
                wmse_output.append(wmse)
                mof2desc_preds.append(pred_desc)

        wmse_output = torch.concat(wmse_output).detach().cpu().numpy()
        mof2desc_preds = torch.concat(mof2desc_preds).detach().cpu().numpy()
        scaler = self._desc2mof_scaler
        mof2desc_preds_decoded = (
            scaler.decode(mof2desc_preds) if scaler else mof2desc_preds
        )

        wmse_reshaped = wmse_output.reshape(-1, topk)
        mof_names_reshaped = np.array(decode_token2mof(all_output)).reshape(-1, topk)
        valid_mask_reshaped = valid_mask.reshape(-1, topk)

        wmse_masked = np.where(valid_mask_reshaped, wmse_reshaped, np.inf)
        min_indices = wmse_masked.argmin(axis=1)
        row_idx = np.arange(wmse_reshaped.shape[0])

        best_wmse = wmse_reshaped[row_idx, min_indices]
        best_mof_names = mof_names_reshaped[row_idx, min_indices]

        feature_names = self._desc2mof_feature_names or []
        new_df = pd.DataFrame(target_data.x_origin, columns=feature_names)
        new_df["filename"] = best_mof_names
        new_df["wmse"] = best_wmse

        if self._sk_model is not None:
            prop_preds = sk_predict(
                mof2desc_preds_decoded, self._sk_model, self._sk_scaler
            )
            prop_preds_reshaped = prop_preds.reshape(-1, topk)
            best_prop_preds = prop_preds_reshaped[row_idx, min_indices]
            new_df["pred_value"] = best_prop_preds

        wmse_mask = new_df["wmse"] < wmse_target
        final_mask = wmse_mask & any_valid_mask

        return new_df[final_mask], new_df[~final_mask], log_list

    def generate(
        self,
        num_samples: int = 100,
        target: Optional[float] = None,
        output_type: Literal["cif", "token", "df"] = "df",
        topk: int = 5,
        temperature: float = 1.0,
        wmse_target: float = 0.5,
        batch_size: int = 256,
        num_workers: int = 0,
        device: str = "cuda",
        save_descriptor_path: Optional[str] = None,
    ) -> Union[pd.DataFrame, List[str]]:
        """Generate MOF structures from target property.

        Pipeline:
        1. prop2desc.sample() → descriptor tensor
        2. desc2mof inference → MOF tokens (beam search)
        3. mof2desc + wmse → select best MOFs
        4. Optional: save generated descriptors to CSV

        Args:
            num_samples: Number of MOFs to generate per target
            target: Target property value for conditional generation
            output_type: 'df' returns DataFrame, 'token' returns MOF name list
            topk: Beam width for desc2mof beam search
            temperature: Sampling temperature for desc2mof
            wmse_target: WMSE threshold for filtering generated MOFs
            batch_size: Batch size for inference
            num_workers: Num workers for DataLoader
            device: Device to run models on
            save_descriptor_path: If provided, save generated descriptors to this path

        Returns:
            DataFrame with generated MOFs (if output_type='df') or list of MOF names
        """
        if self.prop2desc is None:
            self.load()
        if self.prop2desc is None:
            raise RuntimeError(
                "prop2desc not loaded. Provide prop2desc checkpoint or config."
            )

        desc_tensor = self.prop2desc.sample(
            num_samples=num_samples,
            target=target,
        )
        if len(desc_tensor.shape) == 3:
            desc_tensor = desc_tensor.squeeze(1)

        if save_descriptor_path:
            desc_df = pd.DataFrame(desc_tensor.detach().cpu().numpy())
            desc_df.to_csv(save_descriptor_path, index=False)
            print(f"Descriptors saved to {save_descriptor_path}")

        valid_df, invalid_df, log_list = self._run_desc2mof_and_select(
            desc_tensor,
            topk=topk,
            temperature=temperature,
            wmse_target=wmse_target,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
        )

        print(f"Generated {len(valid_df)} valid MOFs out of {num_samples}")

        if output_type == "df":
            result_df: pd.DataFrame = valid_df  # type: ignore[assignment]
            return result_df
        elif output_type == "token":
            return valid_df["filename"].tolist()
        else:
            raise ValueError(f"Unknown output_type: {output_type}")

    def _run_desc2mof_and_select(
        self,
        target_desc: torch.Tensor,
        topk: int = 5,
        temperature: float = 1.0,
        wmse_target: float = 0.5,
        batch_size: int = 256,
        num_workers: int = 0,
        device: str = "cuda",
    ):
        """Internal: run full desc2mof → mof2desc → selection pipeline."""
        all_output, mof_names, target_data = self._run_desc2mof(
            target_desc,
            topk=topk,
            temperature=temperature,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
        )
        return self._run_mof2desc_and_select(
            all_output,
            target_data,
            topk=topk,
            wmse_target=wmse_target,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
        )

    def save(self):
        # """Save models (prop2desc checkpoint)."""
        # if self.prop2desc:
        #     self.prop2desc.save_checkpoint("egmof_prop2desc.ckpt")
        raise NotImplementedError("Saving is not implemented yet.")

    @classmethod
    def from_config(
        cls,
        model_ckpt_path: str | Path = None,
    ):
        pass