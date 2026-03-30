# EGMOF: Efficient Generative Model for Metal-Organic Frameworks

EGMOF is a PyTorch Lightning-based generative model for Metal-Organic Frameworks (MOFs) that learns to generate MOF structures from target properties using a diffusion model (prop2desc) and a transformer decoder (desc2mof).

---

## Architecture Overview

```
Property → [prop2desc] → Descriptor → [desc2mof] → MOF tokens → [mof2desc] → Validation
                                              ↑
                                         (beam search)
```

- **prop2desc**: Diffusion model that generates descriptors from target property
- **desc2mof**: Transformer encoder-decoder that converts descriptors to MOF SELFIES tokens
- **mof2desc**: Validates generated MOFs by computing weighted MSE between predicted and target descriptors

---

## Quick Start

### 1. Training prop2desc

Create config files and run training:

```yaml
# config_prop2desc_model.yaml
model:
  in_channels: 183        # descriptor dimension
  timestep: 1000          # diffusion timesteps
  lr: 1e-4
  dim: 183
  dim_mults: [1, 2]
  condition: numeric      # numeric, binary, class, or None
  scaler_mode: standard   # standard or minmax

datamodule:
  batch_size: 64
  num_workers: 4
  dataset_cls: csv       # csv, json, or text

trainer:
  accelerator: gpu
  devices: 1
  max_epochs: 100
  gradient_clip_val: 1.0
```

```python
# train_prop2desc.py
from egmof.egmof import EGMOF

egmof = EGMOF(
    target="CO2_working_capacity",
    data_path="/path/to/your/data.csv",
    prop2desc_model_config_path="config_prop2desc_model.yaml",
    prop2desc_training_config_path="config_prop2desc_model.yaml",  # same file or separate
)
egmof.train()
# Checkpoint saved automatically by Lightning Trainer
```

**Expected data format** (`data.csv`):
```csv
CO2_working_capacity,feature_1,feature_2,...,feature_183
150.5,0.12,0.34,...,0.89
```

- First column: target property
- Remaining columns: descriptor features (183 dimensions)

### 2. Generation with prop2desc checkpoint

```python
from egmof.egmof import EGMOF

egmof = EGMOF(
    target="CO2_working_capacity",
    data_path="/path/to/data.csv",
    prop2desc_model_config_path="config_prop2desc_model.yaml",
    overrides={"model.ckpt_path": "/path/to/prop2desc.ckpt"},
)

# Load models
egmof.load()

# Generate MOFs for target property = 150.0
results = egmof.generate(
    num_samples=100,
    target=150.0,
    output_type="df",
    topk=5,           # beam search width
    temperature=1.0,
    wmse_target=0.5,   # WMSE threshold for filtering
)
# results: DataFrame with columns [filename, wmse, pred_value, ...]
```

### 3. Full Generation Pipeline (requires all checkpoints)

```python
egmof = EGMOF(
    target="CO2_working_capacity",
    data_path="/path/to/data.csv",
    prop2desc_model_config_path="config_prop2desc_model.yaml",
    overrides={"model.ckpt_path": "/path/to/prop2desc.ckpt"},
    # desc2mof
    desc2mof_ckpt_dir="/path/to/desc2mof.ckpt",
    desc2mof_config_path="/path/to/desc2mof_config.yaml",
    desc2mof_mean_dir="data/mean.csv",
    desc2mof_std_dir="data/std.csv",
    desc2mof_feature_name_dir="data/feature_names.txt",
    # mof2desc (for validation)
    mof2desc_ckpt_dir="/path/to/mof2desc.ckpt",
    # sklearn model (for property prediction)
    skmodel_ckpt_dir="/path/to/sk_model.joblib",
    skmodel_mean_std_dir="/path/to/scaler.json",
)

egmof.load()
results = egmof.generate(
    num_samples=100,
    target=150.0,
    output_type="df",
)
```

---

## Training desc2mof

**desc2mof training is NOT supported through EGMOF class.** Use the standalone script:

```bash
python -m egmof.desc2mof.pretrain \
    --config /path/to/desc2mof_config.yaml \
    --accelerator gpu \
    --devices 2
```

---

## Known Issues / Bugs

### 1. Duplicate method in `desc2mof/model.py` (lines 554-610)

Two `on_predict_epoch_end` methods defined - the second one (lines 576-610) overrides the first:

```python
# Line 554-574: First definition (overwritten)
def on_predict_epoch_end(self):
    ...

# Line 576-610: Second definition (active)
def on_predict_epoch_end(self):  
    ...
```

**Fix**: Delete one of the two methods.

### 2. Deprecated imports in `desc2mof/model.py` and `desc2mof/pretrain.py`

Uses deprecated `pytorch_lightning` instead of `lightning`:

```python
# Wrong
import pytorch_lightning as pl
from pytorch_lightning import Trainer

# Should be
from lightning import Trainer
```

### 3. Undefined variable in `desc2mof/pretrain.py` (lines 93, 99)

```python
# Line 93: num_device is never defined
accumulate_grad_batches = config["batch_size"] // (
    config["per_gpu_batchsize"] * num_device * config["num_nodes"]
)
```

**Fix**: Should use `args.devices` instead of undefined `num_device`.

### 4. Missing dependency in `pyproject.toml`

`egmof.py` imports `joblib` but it's not in dependencies:

```toml
# Add to dependencies in pyproject.toml
joblib
```

---

## API Reference

### EGMOF Class

```python
EGMOF(
    target: str,                              # Target property name
    data_path: str | Path,                    # Path to training data
    prop2desc_model_config_path: Optional,    # Model config YAML
    prop2desc_training_config_path: Optional, # Training config YAML
    overrides: Optional[Dict],                 # Config overrides
    # For generation only:
    desc2mof_ckpt_dir: Optional,
    desc2mof_config_path: Optional,
    mof2desc_ckpt_dir: Optional,
    skmodel_ckpt_dir: Optional,
)
```

### Methods

- `train()`: Train prop2desc model
- `load()`: Load all model checkpoints  
- `generate(num_samples, target, ...)`: Generate MOFs from target property

---

## Project Structure

```
src/egmof/
├── egmof.py              # Main orchestrator
├── data/                 # LightningDataModule + datasets
├── descriptors/          # RAC + Zeo++ calculator
├── desc2mof/             # Descriptor → MOF (Transformer)
├── mof2desc/             # MOF → Descriptor (validation)
├── prop2desc/            # Property → Descriptor (Diffusion)
└── builder/              # MOF building blocks
```
