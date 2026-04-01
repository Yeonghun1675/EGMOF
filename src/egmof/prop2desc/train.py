from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Type

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from omegaconf import OmegaConf

from egmof.data import Datamodule
from egmof.data.dataset import CSVDataset, JsonSplitDataset, TextSplitDataset

from .model import Prop2Desc


DatasetType = Type[CSVDataset | TextSplitDataset | JsonSplitDataset]


def _dataset_cls_from_name(name: str) -> DatasetType:
    name = (name or "csv").lower()
    if name == "csv":
        return CSVDataset
    if name in {"text", "txt"}:
        return TextSplitDataset
    if name == "json":
        return JsonSplitDataset
    raise ValueError(f"Unknown dataset_cls: {name}. Use one of [csv, text, json].")


def _build_datamodule(cfg: Any, data_path: str | Path, target: str | None) -> Datamodule:
    dm_cfg = cfg.get("datamodule", {})
    dataset_cls = _dataset_cls_from_name(dm_cfg.get("dataset_cls", "csv"))
    return Datamodule(
        path=data_path,
        batch_size=int(dm_cfg.get("batch_size", 64)),
        num_workers=int(dm_cfg.get("num_workers", 4)),
        dataset_cls=dataset_cls,
        task=dm_cfg.get("task", None),
        target=target,
    )


def _build_model(cfg: Any, scaler_value: dict[str, Any] | None) -> Prop2Desc:
    model_cfg = cfg.get("model", {})
    return Prop2Desc(
        in_channels=int(model_cfg["in_channels"]),
        timestep=int(model_cfg.get("timestep", 1000)),
        lr=float(model_cfg.get("lr", 1e-4)),
        dim=int(model_cfg.get("dim", model_cfg["in_channels"])),
        dim_mults=list(model_cfg.get("dim_mults", [1, 2])),
        condition=model_cfg.get("condition", None),
        out_channels=model_cfg.get("out_channels", None),
        num_classes=int(model_cfg.get("num_classes", 0)),
        cond_dim=int(model_cfg.get("cond_dim", 0) or 0),
        scaler_mode=model_cfg.get("scaler_mode", "standard"),
        scaler_value=scaler_value,
    )


def _resolve_scaler_value(cfg: Any, datamodule: Datamodule) -> dict[str, Any] | None:
    model_cfg = cfg.get("model", {})
    scaler_value = model_cfg.get("scaler_value", None)
    if scaler_value is not None:
        return OmegaConf.to_container(scaler_value, resolve=True)

    scaler_mode = model_cfg.get("scaler_mode", "standard")
    if scaler_mode == "standard":
        return datamodule.get_mean_and_std()
    if scaler_mode == "minmax":
        return datamodule.get_min_and_max()
    raise ValueError(f"Unknown scaler_mode: {scaler_mode}")


def _build_loggers(cfg: Any, log_dir: str | Path | None) -> list[Any]:
    logger_cfg = cfg.get("logger", {})
    tensorboard_cfg = logger_cfg.get("tensorboard", {})
    wandb_cfg = logger_cfg.get("wandb", {})

    base_log_dir = str(log_dir) if log_dir is not None else "logs"
    logger_list: list[Any] = [
        TensorBoardLogger(
            save_dir=tensorboard_cfg.get("save_dir", base_log_dir),
            name=tensorboard_cfg.get("name", "tensorboard"),
            version=tensorboard_cfg.get("version", None),
        )
    ]

    enable_wandb = bool(wandb_cfg.get("enabled", True))
    if enable_wandb:
        wandb_kwargs = {
            "name": wandb_cfg.get("name", None),
            "project": wandb_cfg.get("project", "egmof"),
            "save_dir": wandb_cfg.get("save_dir", base_log_dir),
            "offline": bool(wandb_cfg.get("offline", False)),
            "log_model": wandb_cfg.get("log_model", False),
        }
        if group := wandb_cfg.get("group", None):
            wandb_kwargs["group"] = group
        if tags := wandb_cfg.get("tags", None):
            wandb_kwargs["tags"] = list(tags)

        try:
            logger_list.append(WandbLogger(**wandb_kwargs))
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "WandbLogger is enabled but `wandb` is not installed. Install wandb or set logger.wandb.enabled=false."
            ) from exc

    return logger_list


def _build_trainer(cfg: Any, log_dir: str | Path | None, ckpt_dir: str | Path | None) -> Trainer:
    trainer_cfg = OmegaConf.to_container(cfg.get("trainer", {}), resolve=True)
    trainer_cfg = dict(trainer_cfg)

    callbacks = []
    if ckpt_dir is not None:
        callbacks.append(
            ModelCheckpoint(
                dirpath=str(ckpt_dir),
                monitor="val/loss",
                mode="min",
                save_top_k=1,
                save_last=True,
            )
        )
    if log_dir is not None:
        callbacks.append(LearningRateMonitor(logging_interval="epoch"))

    existing_callbacks = trainer_cfg.pop("callbacks", None)
    if existing_callbacks:
        callbacks.extend(existing_callbacks)
    if callbacks:
        trainer_cfg["callbacks"] = callbacks

    trainer_cfg["logger"] = _build_loggers(cfg, log_dir)

    if log_dir is not None and "default_root_dir" not in trainer_cfg:
        trainer_cfg["default_root_dir"] = str(log_dir)

    return Trainer(**trainer_cfg)


def train_prop2desc(
    config_path: str | Path | Any,
    data_path: str | Path,
    target: str | None,
    log_dir: str | Path | None = None,
    ckpt_dir: str | Path | None = None,
    test_only: bool = False,
) -> Prop2Desc:
    cfg = OmegaConf.load(config_path) if isinstance(config_path, str | Path) else config_path
    seed_everything(int(cfg.get("seed", 42)), workers=True)

    datamodule = _build_datamodule(cfg, data_path=data_path, target=target)
    datamodule.setup("test" if test_only else "fit")

    if not test_only:
        datamodule.setup("fit")
    scaler_value = _resolve_scaler_value(cfg, datamodule)
    model = _build_model(cfg, scaler_value=scaler_value)

    trainer = _build_trainer(cfg, log_dir=log_dir, ckpt_dir=ckpt_dir)
    ckpt_path = cfg.get("ckpt_path", None)

    if test_only:
        datamodule.setup("test")
        trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path)
        return model

    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the prop2desc diffusion model.")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML.")
    parser.add_argument("--data_path", type=str, required=True, help="Directory containing train/val/test splits.")
    parser.add_argument("--target", type=str, default=None, help="Target property column name.")
    parser.add_argument("--log_dir", type=str, default=None, help="Optional Lightning log directory.")
    parser.add_argument("--ckpt_dir", type=str, default=None, help="Optional checkpoint directory.")
    parser.add_argument("--test_only", action="store_true", help="Run test instead of training.")
    args = parser.parse_args()

    train_prop2desc(
        config_path=Path(args.config),
        data_path=Path(args.data_path),
        target=args.target,
        log_dir=Path(args.log_dir) if args.log_dir else None,
        ckpt_dir=Path(args.ckpt_dir) if args.ckpt_dir else None,
        test_only=args.test_only,
    )


if __name__ == "__main__":
    main()
