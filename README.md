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
    target="target",              # Column name in your CSV (or "co2_working_capacity", etc.)
    data_path="/path/to/data/",   # Directory containing train.csv, val.csv, test.csv
    prop2desc_model_config_path="config_prop2desc_model.yaml",
    prop2desc_training_config_path="config_prop2desc_model.yaml",
)
egmof.train()
# Checkpoint saved automatically by Lightning Trainer
```

**Expected data format** (`train.csv`, `val.csv`, `test.csv`):
```csv
target,feature_1,feature_2,...,feature_183
150.5,0.12,0.34,...,0.89
```

- `target`: Column name for your property (e.g., `co2_working_capacity`, `target`, etc.)
- **Descriptor columns**: Must match exactly the 183 features in `src/egmof/data/descriptor_name.json`

CSV 파일들을 `data_path` 디렉토리에 넣으세요:
```
/path/to/data/
├── train.csv
├── val.csv
└── test.csv
```

### 2. Generation with prop2desc checkpoint

```python
from egmof.egmof import EGMOF

egmof = EGMOF(

    prop2desc_model_config_path="config_prop2desc_model.yaml",
    overrides={"model.ckpt_path": "/path/to/prop2desc.ckpt"},
    skmodel_mean_std_dir = 'config/h2uptake/18463.yaml'
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
# results: DataFrame with columns [filename, wmse, pred_value (optional)]
# pred_value column only exists if skmodel_ckpt_dir or skmodel_mean_std_dir is provided
```

### 3. Full Generation Pipeline

**Default paths**: All desc2mof and mof2desc paths are set to default values:
- `desc2mof_ckpt_dir`: `checkpoints/desc2mof/desc2mof_best.ckpt`
- `desc2mof_config_path`: `config/desc2mof_training_config.yaml`
- `desc2mof_mean_dir`: `src/egmof/desc2mof/data/mean_all.csv`
- `desc2mof_std_dir`: `src/egmof/desc2mof/data/std_all.csv`
- `desc2mof_feature_name_dir`: `src/egmof/desc2mof/data/feature_name.txt`
- `mof2desc_ckpt_dir`: `checkpoints/mof2desc/mof2desc_best.ckpt`
- `mof2desc_config_path`: `config/mof2desc_training_config.yaml`

If checkpoints don't exist, `egmof.load()` will raise an error with download URL.

**Pre-configured yaml files** with feature_importances:
- `config/h2uptake/`: H2 uptake models (1000, 2200, 5000, 10000, 18463)
- `config/various_dataset/`: Various gas uptake models (CO2, CH4, N2, etc.)

```python
# Minimal usage (uses all default paths)
egmof = EGMOF(
    prop2desc_model_config_path="config_prop2desc_model.yaml",
    overrides={"model.ckpt_path": "/path/to/prop2desc.ckpt"},
)

egmof.load()  # Raises error if default checkpoints not found

# Override default paths if needed
egmof = EGMOF(
    prop2desc_model_config_path="config_prop2desc_model.yaml",
    overrides={"model.ckpt_path": "/path/to/prop2desc.ckpt"},
    desc2mof_ckpt_dir="/path/to/desc2mof.ckpt",
    mof2desc_ckpt_dir="/path/to/mof2desc.ckpt",
    skmodel_ckpt_dir="/path/to/sk_model.pickle",      # Optional: for pred_value column
    skmodel_mean_std_dir="/path/to/scaler.yaml",      # Optional: for feature_importances
)

# Without sklearn model (no pred_value column)
egmof = EGMOF(
    prop2desc_model_config_path="config_prop2desc_model.yaml",
    skmodel_mean_std_dir="/path/to/scaler.yaml",  # Loads feature_importances from yaml
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

## API Reference

### EGMOF Class

```python
EGMOF(
    target: str,                              # Target property name
    data_path: str | Path,                    # Path to training data
    prop2desc_model_config_path: Optional,    # Model config YAML
    prop2desc_training_config_path: Optional, # Training config YAML
    overrides: Optional[Dict],                 # Config overrides
    # desc2mof (defaults set automatically)
    desc2mof_ckpt_dir: Optional,               # Default: checkpoints/desc2mof/desc2mof_best.ckpt
    desc2mof_config_path: Optional,            # Default: config/desc2mof_training_config.yaml
    desc2mof_mean_dir: Optional,               # Default: src/egmof/desc2mof/data/mean_all.csv
    desc2mof_std_dir: Optional,                # Default: src/egmof/desc2mof/data/std_all.csv
    desc2mof_feature_name_dir: Optional,       # Default: src/egmof/desc2mof/data/feature_name.txt
    desc2mof_feature_size: int = 183,
    # mof2desc (defaults set automatically)
    mof2desc_ckpt_dir: Optional,               # Default: checkpoints/mof2desc/mof2desc_best.ckpt
    mof2desc_config_path: Optional,             # Default: config/mof2desc_training_config.yaml
    # sklearn model (optional)
    skmodel_ckpt_dir: Optional,
    skmodel_mean_std_dir: Optional,
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
