# descriptors

RAC (Resource-Aware Covariants) + Zeo++ descriptors calculator for MOF CIF files.

## get_all_descriptors.py

Calculate both RAC and Zeo++ descriptors from CIF files in one pipeline.

### Quick Start

```bash
python -m egmof.descriptors.get_all_descriptors
```

Default: reads all `.cif` files from `src/egmof/descriptors/cif/` and outputs to `descriptors.csv`.

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--cif` | Single CIF file path | none |
| `--cif_dir` | Directory containing CIF files | `src/egmof/descriptors/cif/` |
| `--zeopp_bin` | Path to Zeo++ `network` binary | auto-detect |
| `--output` | Output CSV path | `descriptors.csv` |

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

| Column | Description |
|--------|-------------|
| `*` | RAC descriptors (from molSimplify) |
| `sa` | Surface area (Zeo++) |
| `cv` | Crystal volume (Zeo++) |
| `density` | Density (Zeo++) |
| `vf` | Pore volume fraction (Zeo++) |
| `di` | Largest included sphere (Zeo++) |
| `df` | Largest free sphere (Zeo++) |
| `dif` | Pore accessibility (Zeo++) |
| `filename` | CIF filename |

### Zeo++ Installation

Zeo++ requires manual download (registration required):

1. Download: http://www.zeoplusplus.org/download.html
2. Compile from source:
   ```bash
   tar -xf zeo++-0.3.tar.gz
   cd zeo++-0.3/voro++/src && make
   cd ..
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
   - `network <cif> <out>.sa` — surface area, crystal volume, density
   - `network -pb 1.4 <cif> <out>.vol` — pore volume fraction
   - `network -res <cif> <out>.res` — pore dimensions (di, df, dif)
4. **Merge** — Combine RAC + Zeo++ into single DataFrame