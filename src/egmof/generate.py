import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def cal_wmse(
    pred_desc: torch.Tensor, target_desc: torch.Tensor, weights
) -> torch.Tensor:
    """Calculate weighted MSE between predicted and target descriptors."""
    mse = F.mse_loss(pred_desc, target_desc, reduction="none")
    w = torch.tensor(weights, dtype=mse.dtype, device=mse.device)
    weighted_mse = mse * w
    wmse = weighted_mse.sum(dim=-1) / w.sum()
    return wmse


def sk_predict(desc_pred: np.ndarray, sk_model, sk_scaler) -> np.ndarray:
    """Predict property from descriptor using sklearn model."""
    desc_scaled = sk_scaler.encode(desc_pred)
    pred_scaled = sk_model.predict(desc_scaled)
    return sk_scaler.decode_target(pred_scaled)


def _build_mofgen_dataset(desc_tensor: torch.Tensor, scaler, feature_name_dir: str):
    """Build MOFGenDataset from descriptor tensor [N, D]."""
    from .desc2mof import MOFGenDataset

    desc_np = desc_tensor.detach().cpu().numpy()
    df = pd.DataFrame(desc_np, columns=None)
    return MOFGenDataset(
        df, scaled=True, scaler=scaler, feature_name_dir=feature_name_dir
    )


def _parse_mof_output(all_output: list, SEP_TOKEN_ID: int):
    """Parse MOF token output, returns (valid_mask, log_list)."""
    from .desc2mof.utils import is_valid as _is_valid

    return _is_valid(all_output, SEP_TOKEN_ID)


def run_desc2mof(
    model,
    target_desc: torch.Tensor,
    scaler,
    feature_names: list[str],
    feature_name_dir: str,
    topk: int = 5,
    temperature: float = 1.0,
    batch_size: int = 256,
    num_workers: int = 0,
    device: str = "cuda",
):
    """Run desc2mof inference: descriptor → MOF tokens (beam search)."""
    from .desc2mof import MOFGenDataset, decode_token2mof

    model.eval()

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
            preds, _, _, _ = model.generate_beam(
                desc_batch,
                temperature=temperature,
                beam_width=topk,
            )
            preds_output.append(preds)
    preds_output = torch.concat(preds_output)
    all_output = (
        preds_output.reshape(-1, preds_output.shape[-1]).detach().cpu().tolist()
    )
    mof_names = decode_token2mof(all_output)
    return all_output, mof_names, target_data


def run_mof2desc_and_select(
    model,
    all_output: list,
    target_data,
    feature_size: int,
    weights=None,
    sk_model=None,
    sk_scaler=None,
    desc2mof_scaler=None,
    topk: int = 5,
    wmse_target: float = 0.5,
    batch_size: int = 256,
    num_workers: int = 0,
    device: str = "cuda",
):
    """Run mof2desc, compute wmse, and select best MOFs."""
    from .desc2mof import SEP_TOKEN
    from .mof2desc import Desc2MOFOutputDataset
    from .desc2mof import decode_token2mof

    model = model.to(device)
    model.eval()

    valid_mask, log_list = _parse_mof_output(all_output, SEP_TOKEN_ID=SEP_TOKEN)
    any_valid_mask = valid_mask.reshape(-1, topk).any(axis=1)

    x_np = np.array(target_data.x.squeeze(-1))
    x_tensor = (
        torch.from_numpy(x_np).unsqueeze(1).repeat(1, topk, 1).view(-1, feature_size)
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
            pred_desc = model(token_ids, attention_mask)
            if weights is not None:
                wmse = cal_wmse(pred_desc, target_desc, weights)
                wmse_output.append(wmse)
            mof2desc_preds.append(pred_desc)

    mof2desc_preds = torch.concat(mof2desc_preds).detach().cpu().numpy()
    if desc2mof_scaler is not None:
        mof2desc_preds_decoded = desc2mof_scaler.decode(mof2desc_preds)
    else:
        mof2desc_preds_decoded = mof2desc_preds

    if weights is not None:
        wmse_output = torch.concat(wmse_output).detach().cpu().numpy()
        wmse_reshaped = wmse_output.reshape(-1, topk)
    else:
        wmse_reshaped = None

    mof_names_reshaped = np.array(decode_token2mof(all_output)).reshape(-1, topk)
    valid_mask_reshaped = valid_mask.reshape(-1, topk)

    if weights is not None:
        wmse_masked = np.where(valid_mask_reshaped, wmse_reshaped, np.inf)
        min_indices = wmse_masked.argmin(axis=1)
        row_idx = np.arange(wmse_reshaped.shape[0])
        best_wmse = wmse_reshaped[row_idx, min_indices]
    else:
        num_samples = len(mof_names_reshaped)
        min_indices = np.zeros(num_samples, dtype=int)
        row_idx = np.arange(num_samples)
        best_wmse = None

    best_mof_names = mof_names_reshaped[row_idx, min_indices]

    result_df = pd.DataFrame({"filename": best_mof_names})
    if best_wmse is not None:
        result_df["wmse"] = best_wmse

    if sk_model is not None and sk_scaler is not None:
        prop_preds = sk_predict(mof2desc_preds_decoded, sk_model, sk_scaler)
        prop_preds_reshaped = prop_preds.reshape(-1, topk)
        best_prop_preds = prop_preds_reshaped[row_idx, min_indices]
        result_df["pred_value"] = best_prop_preds

    if best_wmse is not None:
        wmse_mask = result_df["wmse"] < wmse_target
        final_mask = wmse_mask & any_valid_mask
    else:
        final_mask = any_valid_mask

    return result_df[final_mask], result_df[~final_mask], log_list
