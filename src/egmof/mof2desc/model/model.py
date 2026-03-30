import os
import math
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import lightning as pl
from transformers import get_cosine_schedule_with_warmup, AutoModel
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
)
from .dataset import PAD_TOKEN, SEP_TOKEN, MOF_ENCODE_DICT, MOF_DECODE_DICT


class MOF2Desc(pl.LightningModule):
    def __init__(self, config, scaler=None):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        self.lr = config["learning_rate"]
        self.scaler = scaler
        self.warmup_step = config["warmup_steps"]
        self.vocab_size = max(MOF_ENCODE_DICT.values()) + 1
        self.model = MOFEncoder(
            vocab_size=self.vocab_size,
            target_dim=config["desc_dim"],
            pad_token_id=PAD_TOKEN,
            max_len=config.get("max_token_len", 512),
        )
        self.pad_token = PAD_TOKEN
        self.validation_outputs = []
        self.test_outputs = []

    def forward(self, token_ids, token_type_ids):
        return self.model(token_ids, token_type_ids)

    def training_step(self, batch, batch_idx):
        y, token_ids, token_type_ids = batch
        y = y.squeeze()
        y_hat = self(token_ids, token_type_ids)
        loss = F.mse_loss(y_hat, y)
        self.log("train_loss", loss, sync_dist=True, on_epoch=True, on_step=False)
        return loss

    def on_validation_start(self):
        self.validation_outputs = []

    def validation_step(self, batch, batch_idx):
        y, token_ids, token_type_ids = batch
        batch_size = y.shape[0]
        y = y.squeeze()
        y_hat = self(token_ids, token_type_ids)
        loss = F.mse_loss(y_hat, y)
        output = {
            "val_loss": loss.item() * batch_size,
            "y_true": y,
            "y_pred": y_hat,
            "batch_size": batch_size,
        }
        self.validation_outputs.append(output)
        return output

    def on_validation_epoch_end(self):
        outputs = self.validation_outputs
        total_samples = sum(x["batch_size"] for x in outputs)
        avg_loss = sum(x["val_loss"] for x in outputs) / total_samples
        y_true = torch.concat([x["y_true"] for x in outputs]).cpu().numpy()
        y_pred = torch.concat([x["y_pred"] for x in outputs]).detach().cpu().numpy()
        y_true_origin = self.scaler.decode(y_true)
        y_pred_origin = self.scaler.decode(y_pred)
        r2 = r2_score(y_true_origin, y_pred_origin)
        mae = mean_absolute_error(y_true_origin, y_pred_origin)
        self.log(
            "val/avg_val_loss", avg_loss, sync_dist=True, on_epoch=True, on_step=False
        )
        self.log("val/avg_val_mae", mae, sync_dist=True, on_epoch=True, on_step=False)
        self.log(
            "val/avg_val_r2score", r2, sync_dist=True, on_epoch=True, on_step=False
        )

    def on_test_start(self):
        self.test_outputs = []

    def test_step(self, batch, batch_idx):
        y, token_ids, token_type_ids = batch
        batch_size = y.shape[0]
        y = y.squeeze()
        y_hat = self(token_ids, token_type_ids)
        loss = F.mse_loss(y_hat, y)
        output = {
            "test_loss": loss.item() * batch_size,
            "y_true": y,
            "y_pred": y_hat,
            "batch_size": batch_size,
        }
        self.test_outputs.append(output)
        return output

    def on_test_epoch_end(self):
        outputs = self.test_outputs
        total_samples = sum(x["batch_size"] for x in outputs)
        avg_loss = sum(x["test_loss"] for x in outputs) / total_samples
        y_true = torch.concat([x["y_true"] for x in outputs]).cpu().numpy()
        y_pred = torch.concat([x["y_pred"] for x in outputs]).detach().cpu().numpy()
        y_true_origin = np.round(self.scaler.decode(y_true), 6)
        y_pred_origin = np.round(self.scaler.decode(y_pred), 6)
        r2 = r2_score(y_true_origin, y_pred_origin)
        mae = mean_absolute_error(y_true_origin, y_pred_origin)
        self.log(
            "test/avg_test_loss", avg_loss, sync_dist=True, on_epoch=True, on_step=False
        )
        self.log("test/avg_test_mae", mae, sync_dist=True, on_epoch=True, on_step=False)
        self.log(
            "test/avg_test_r2score", r2, sync_dist=True, on_epoch=True, on_step=False
        )

        feature_name_dir = self.config.get(
            "feature_name_dir", f"{__mof2desc_dir__}/../desc2mof/data/feature_name.txt"
        )
        with open(feature_name_dir, "r") as g:
            feature_names = [line.strip() for line in g.readlines()]
        r2_raw = r2_score(y_true_origin, y_pred_origin, multioutput="raw_values")
        mae_raw = mean_absolute_error(
            y_true_origin, y_pred_origin, multioutput="raw_values"
        )
        results_df = pd.DataFrame(np.array([mae_raw, r2_raw]), columns=feature_names)
        results_df.insert(0, column="metric", value=["MAE", "r2"])
        results_df.to_csv(f"{self.config['exp_name']}_test_results.csv", index=None)

        save_true = self.config.get("save_true", False)
        if save_true:
            np.save(f"{self.config['exp_name']}_test_origin.npy", y_true_origin)
            np.save(f"{self.config['exp_name']}_test_pred.npy", y_pred_origin)

    def on_predict_epoch_start(self):
        self.pred_outputs = []

    def predict_step(self, batch, batch_idx):
        token_ids, token_type_ids = batch
        y_hat = self(token_ids, token_type_ids)
        self.pred_outputs.append(y_hat)

    def on_predict_epoch_end(self):
        all_pred = torch.concat(self.pred_outputs).detach().cpu().numpy()
        all_pred_origin = self.scaler.decode(all_pred)
        np.save(f"{self.config['exp_name']}_desc_pred.npy", all_pred_origin)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        if self.trainer.max_steps == -1:
            max_steps = self.trainer.estimated_stepping_batches
        else:
            max_steps = self.trainer.max_steps
        if isinstance(self.warmup_step, float):
            warmup_steps = int(max_steps * self.warmup_step)
        else:
            warmup_steps = self.warmup_step
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_steps
        )
        sched = {"scheduler": scheduler, "interval": "step"}
        return ([optimizer], [sched])


class MOFEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        target_dim: int = 183,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        pad_token_id: int = 0,
        max_len: int = 512,
        use_cls_token: bool = True,
        mlp_hidden: int = 256,
        mlp_dropout: float = 0.1,
    ):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.use_cls_token = use_cls_token
        self.max_len = max_len

        if use_cls_token:
            self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.normal_(self.cls, std=0.02)

        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        pos_size = max_len + (1 if use_cls_token else 0)
        self.pos_emb = nn.Embedding(pos_size, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_hidden, d_model),
        )
        self.regressor = nn.Linear(d_model, target_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        B, T = input_ids.shape
        if T > self.max_len:
            input_ids = input_ids[:, : self.max_len]
            attention_mask = attention_mask[:, : self.max_len]
            T = self.max_len

        x = self.tok_emb(input_ids)
        src_key_padding_mask = attention_mask == 0

        if self.use_cls_token:
            cls = self.cls.expand(B, 1, -1)
            x = torch.cat([cls, x], dim=1)
            cls_pad = torch.zeros(
                B, 1, dtype=torch.bool, device=src_key_padding_mask.device
            )
            src_key_padding_mask = torch.cat([cls_pad, src_key_padding_mask], dim=1)
            pos_ids = (
                torch.arange(T + 1, device=input_ids.device)
                .unsqueeze(0)
                .expand(B, T + 1)
            )
        else:
            pos_ids = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)

        x = x + self.pos_emb(pos_ids)
        h = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        h = self.norm(h)

        if self.use_cls_token:
            pooled = h[:, 0, :]
        else:
            mask = (attention_mask == 1).unsqueeze(-1)
            denom = mask.sum(dim=1).clamp_min(1)
            pooled = (h * mask).sum(dim=1) / denom

        pooled = pooled + self.mlp(pooled)
        return self.regressor(pooled)
