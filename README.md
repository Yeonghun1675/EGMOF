# EGMOF: Efficient Generative Model for Metal-Organic Frameworks (v.0.0.2)

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

### 1. Create EGMOF instance

```python
from egmof import EGMOF

# Auto-loads pretrained desc2mof + mof2desc using defaults
egmof = EGMOF(
    load_pretrained_modules=True,  # loads desc2mof & mof2desc
    prop2desc_ckpt_path="path/to/prop2desc.ckpt",
    prop2desc_config_path="path/to/prop2desc_config.yaml",
    skmodel_ckpt_dir="path/to/sk_model.pkl",  # Optional: for pred_value
    accelerator="cuda",
    devices=1,
)

# Or disable auto-loading, then call setup() manually
egmof = EGMOF(load_pretrained_modules=False)
egmof.setup()  # loads desc2mof & mof2desc using defaults

# Or manually load specific models
egmof = EGMOF(load_pretrained_modules=False)
egmof.load_desc2mof(ckpt_path="...", config_path="...", ...)
egmof.load_mof2desc(ckpt_path="...", config_path="...", scaler=egmof._desc2mof_scaler)
```

### 2. Train models

**Tip:** With `load_pretrained_modules=True`, scaler is already loaded. `train_desc2mof` and `train_mof2desc` will reuse it automatically.

```python
# Train prop2desc
egmof.train_prop2desc(
    data_path="/path/to/data/",
    target_property="target",
    prop2desc_config_path="config/prop2desc_model_config.yaml",
)

# Train desc2mof (will reuse scaler if already loaded)
egmof.train_desc2mof(
    accelerator="gpu",
    devices=1,
    max_epochs=200,
)

# Train mof2desc (will reuse scaler from desc2mof)
egmof.train_mof2desc(
    accelerator="gpu",
    devices=1,
    max_epochs=500,
)
```

### 3. Generate MOFs

```python
results = egmof.generate(
    num_samples=100,
    target_value=150.0,  # target property value
)
```

---

## Default Paths

| Model | Default Path |
|-------|--------------|
| desc2mof checkpoint | `checkpoints/desc2mof/desc2mof_best.ckpt` |
| desc2mof config | `config/desc2mof_training_config.yaml` |
| desc2mof mean | `src/egmof/desc2mof/data/mean_all.csv` |
| desc2mof std | `src/egmof/desc2mof/data/std_all.csv` |
| desc2mof feature names | `src/egmof/desc2mof/data/feature_name.txt` |
| mof2desc checkpoint | `checkpoints/mof2desc/mof2desc_best.ckpt` |
| mof2desc config | `config/mof2desc_training_config.yaml` |

---

## Sklearn Model (Optional)

For `pred_value` column in generate output, load sklearn model:

```python
egmof = EGMOF(
    skmodel_ckpt_dir="path/to/sk_model.pkl",
    prop2desc_config_path="path/to/config.yaml",  # contains feature_importances & scaler
)
```

**Requirements in prop2desc_config:**
- `feature_importances`: list of weights for WMSE calculation
- `scaler_value`: {mean, std, target_mean, target_std} for descriptor normalization

If sklearn model not loaded:
- `wmse` column: still available if `feature_importances` in config
- `pred_value` column: not available

---

## Training Config Files

Example config files are in `config/`:
- `prop2desc_model_config.yaml` - prop2desc model configuration
- `prop2desc_training_config.yaml` - prop2desc training configuration
- `desc2mof_training_config.yaml` - desc2mof training configuration
- `mof2desc_training_config.yaml` - mof2desc training configuration

**Pre-configured property configs:**
- `config/h2uptake/` - H2 uptake models (1000, 2200, 5000, 10000, 18463)
- `config/various_dataset/` - Various gas uptake models (CO2, CH4, N2, etc.)

---

## Data Format

Expected CSV format for training data (`train.csv`, `val.csv`, `test.csv`):
```csv
target,feature_1,feature_2,...,feature_183
150.5,0.12,0.34,...,0.89
```

- `target`: Column name for your property
- **Descriptor columns**: Must match exactly the 183 features in `src/egmof/data/descriptor_name.json`

---

## API Reference

### EGMOF Class

```python
EGMOF(
    prop2desc_ckpt_path: Optional[str | Path] = None,
    prop2desc_config_path: Optional[str | Path] = None,
    skmodel_ckpt_dir: Optional[str | Path] = None,  # sklearn model for pred_value
    load_pretrained_modules: bool = True,
    accelerator: Literal["cpu", "cuda"] = "cpu",
    devices: int | List[int] = 1,
)
```

### Methods

| Method | Description |
|--------|-------------|
| `load_prop2desc(ckpt_path, config_path)` | Load prop2desc checkpoint |
| `load_desc2mof(ckpt_path, config_path, mean_path, std_path, feature_name_path)` | Load desc2mof checkpoint + scaler |
| `load_mof2desc(ckpt_path, config_path, scaler)` | Load mof2desc checkpoint (reuses desc2mof scaler) |
| `train_prop2desc(data_path, target_property, config_path)` | Train prop2desc model |
| `train_desc2mof(config_path, train_data_dir, ...)` | Train desc2mof model |
| `train_mof2desc(config_path, train_data_dir, ...)` | Train mof2desc model |
| `generate(num_samples, target_value, ...)` | Generate MOFs from target property |

### generate() Parameters

```python
egmof.generate(
    num_samples=100,
    target_value=150.0,
    output_type="df",       # "df" or "token"
    topk=5,                # beam search width
    temperature=1.0,
    wmse_target=0.5,      # WMSE threshold for filtering
    batch_size=256,
    num_workers=0,
    save_descriptor_path=None,
)
```

**Output columns** (DataFrame):
- `filename`: Generated MOF name
- `wmse`: Weighted MSE (if `_sk_feature_importances` available)
- `pred_value`: Predicted property (if sklearn model loaded)

---

## Project Structure

```
src/egmof/
├── egmof.py              # Main orchestrator (simple wrapper)
├── egmof_backup.py       # Original implementation
├── train.py              # train_desc2mof, train_mof2desc, dataloader creators
├── utils.py              # create_scaler, load_config, load_feature_names
├── generate.py           # run_desc2mof, run_mof2desc_and_select, cal_wmse, sk_predict
├── data/                 # LightningDataModule + datasets
├── descriptors/          # RAC + Zeo++ calculator
├── desc2mof/             # Descriptor → MOF (Transformer)
├── mof2desc/             # MOF → Descriptor (validation)
├── prop2desc/            # Property → Descriptor (Diffusion)
└── builder/              # MOF building blocks
```

### Module Details

- **`egmof.py`**: Entry point - loads models, calls training functions
- **`train.py`**: Core training logic - creates dataloaders, sets up Trainer, runs training
- **`utils.py`**: Helper functions - scaler creation, config loading, feature name loading
- **`generate.py`**: Generation helpers - run_desc2mof, run_mof2desc_and_select, cal_wmse, sk_predict
