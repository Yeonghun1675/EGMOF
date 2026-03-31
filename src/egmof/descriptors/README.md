# Descriptors

RAC (Resource-Aware Covariants) + Zeo++ descriptors calculator for MOF CIF files.

## get_all_descriptors.py

Calculate both RAC and Zeo++ descriptors from CIF files in one pipeline.

### Quick Start

```bash
python -m egmof.descriptors.get_all_descriptors
```

Default: reads all `.cif` files from `cif_opt/` and outputs to `examples/descriptors.csv`.

### Arguments

| Argument      | Description                    | Default                          |
| ------------- | ------------------------------ | -------------------------------- |
| `--cif`       | Single CIF file path           | none                             |
| `--cif_dir`   | Directory containing CIF files | `src/egmof/descriptors/cif_opt/` |
| `--zeopp_bin` | Path to Zeo++ `network` binary | auto-detect                      |
| `--output`    | Output CSV path                | `examples/descriptors.csv`       |

### Example

```bash
# All CIFs in default directory
python -m egmof.descriptors.get_all_descriptors --output all_mofs.csv

# Single CIF file
python -m egmof.descriptors.get_all_descriptors --cif /path/to/MOF.cif --output single.csv

# Custom zeo++ binary
python -m egmof.descriptors.get_all_descriptors --zeopp_bin /home/user/zeo++/network --output mofs.csv

# Custom CIF directory
python -m egmof.descriptors.get_all_descriptors --cif_dir /path/to/cifs --output custom.csv
```

### Example: Calculate from builder/cifs

Generate descriptors for MOFs created by `build_MOFs.py`:

```bash
python -m egmof.descriptors.get_all_descriptors \
    --cif_dir ./cif_opt \
    --output examples/descriptors.csv
```

Or using Python:

```python
from egmof.descriptors.get_all_descriptors import get_all_descriptors

# Use __desc_dir__ for relative paths
from egmof.descriptors.get_all_descriptors import __desc_dir__

cif_dir = str(__desc_dir__ / "./cif_opt")
output_path = str(__desc_dir__ / "examples/descriptors.csv")

df = get_all_descriptors(cif_dir=cif_dir, output_path=output_path)
print(f"Generated {len(df)} rows, {len(df.columns)} columns")
```

### Python API

```python
from egmof.descriptors import get_all_descriptors

df = get_all_descriptors(
    cif_dir="path/to/cifs",
    zeopp_bin="/path/to/network",
    output_path="descriptors.csv",
)
print(df.head())
```

### Output Columns

| Column     | Description                        |
| ---------- | ---------------------------------- |
| `*`        | RAC descriptors (from molSimplify) |
| `sa`       | Surface area (Zeo++)               |
| `cv`       | Crystal volume (Zeo++)             |
| `density`  | Density (Zeo++)                    |
| `vf`       | Pore volume fraction (Zeo++)       |
| `di`       | Largest included sphere (Zeo++)    |
| `df`       | Largest free sphere (Zeo++)        |
| `dif`      | Pore accessibility (Zeo++)         |
| `filename` | CIF filename                       |

### Zeo++ Installation

Zeo++ is automatically downloaded and compiled on first run if not found.

Manual installation (if needed):

1. Download from GitHub: https://github.com/mharanczyk/zeoplusplus
2. Compile:
   ```bash
   cd zeoplusplus-master/voro++/src && make
   cd ../..
   make
   ```
3. Place `network` binary at one of:
   - `src/egmof/descriptors/zeopp/network`
   - Anywhere in PATH (named `network`)
   - Pass explicit path via `--zeopp_bin`

### How It Works

1. **Primitive cell** — Convert CIF to primitive cell (pymatgen)
2. **RAC** — Generate MOF descriptors via molSimplify
3. **Zeo++** — Run three analyses:
   - `network -ha -sa 1.2 1.2 5000 <cif>` — surface area, crystal volume, density
   - `network -ha -vol 1.2 1.2 50000 <cif>` — pore volume fraction
   - `network -ha -res <cif>` — pore dimensions (di, df, dif)
4. **Merge** — Combine RAC + Zeo++ into single DataFrame