import os
import re
import glob
import shutil
import subprocess
import warnings
from pathlib import Path
from tqdm.auto import tqdm

import numpy as np
import pandas as pd
from pymatgen.io.cif import CifParser

from molSimplify.Informatics.MOF.MOF_descriptors import get_MOF_descriptors

warnings.filterwarnings("ignore")

__desc_dir__ = Path(__file__).parent.resolve()

ZEO_GITHUB_URL = (
    "https://github.com/mharanczyk/zeoplusplus/archive/refs/heads/master.tar.gz"
)


def find_zeopp_bin() -> str | None:
    desc_dir = __desc_dir__
    local_bin = desc_dir / "zeopp" / "network"
    if local_bin.exists():
        return str(local_bin)
    if shutil.which("network"):
        return "network"
    return None


def install_zeopp(force: bool = False) -> str:
    desc_dir = __desc_dir__
    zeopp_dir = desc_dir / "zeopp"
    local_bin = zeopp_dir / "network"
    if local_bin.exists() and not force:
        return str(local_bin)

    zeopp_dir.mkdir(parents=True, exist_ok=True)

    common_paths = [
        Path("/usr/local/bin/network"),
        Path("/usr/bin/network"),
        Path.home() / "zeo++-0.3/bin/network",
        Path.home() / "zeo++/bin/network",
    ]
    for p in common_paths:
        if p.exists():
            shutil.copy(p, local_bin)
            local_bin.chmod(0o755)
            print(f"[OK] Found Zeo++ at {p}, copied to {local_bin}")
            return str(local_bin)

    print("[INFO] Downloading and compiling Zeo++ from GitHub...")
    src_tar = zeopp_dir / "zeo.tar.gz"
    src_dir = zeopp_dir / "zeoplusplus-master"

    if not src_dir.exists():
        result = subprocess.run(
            ["wget", "-q", ZEO_GITHUB_URL, "-O", str(src_tar)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to download Zeo++: {result.stderr}")

        subprocess.run(
            ["tar", "-xzf", str(src_tar), "-C", str(zeopp_dir)],
            check=True,
        )

    voro_dir = src_dir / "voro++"
    if voro_dir.exists():
        print("[INFO] Compiling voro++...")
        subprocess.run(
            ["make", "-C", str(voro_dir), "-j4"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )

    print("[INFO] Compiling Zeo++...")
    subprocess.run(
        ["make", "-C", str(src_dir), "clean"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    result = subprocess.run(
        ["make", "-C", str(src_dir), "-j4"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to compile Zeo++: {result.stderr}")

    src_bin = src_dir / "network"
    if not src_bin.exists():
        raise RuntimeError("Zeo++ binary not found after compilation")

    shutil.copy(src_bin, local_bin)
    local_bin.chmod(0o755)
    print(f"[OK] Zeo++ compiled and installed to {local_bin}")
    return str(local_bin)


def ensure_zeopp() -> str:
    bin_path = find_zeopp_bin()
    if bin_path is None:
        return install_zeopp()
    return bin_path


def cif_to_rac(cif_path: str, work_dir: str) -> pd.DataFrame:
    os.makedirs(work_dir, exist_ok=True)
    primitive_dir = os.path.join(work_dir, "primitive")
    os.makedirs(primitive_dir, exist_ok=True)

    cif_name = Path(cif_path).name
    cif_stem = cif_name.replace(".cif", "")

    s = CifParser(cif_path).get_structures()[0]
    sprim = s.get_primitive_structure()
    # Write CIF - use CifWriter with explicit file handling
    from pymatgen.io.cif import CifWriter

    cif_path_out = os.path.join(primitive_dir, cif_name)
    with open(cif_path_out, "w") as f:
        f.write(str(CifWriter(sprim)))

    full_names, full_descriptors = get_MOF_descriptors(
        os.path.join(primitive_dir, cif_name),
        3,
        path=work_dir,
        xyz_path=os.path.join(work_dir, f"{cif_stem}.xyz"),
    )

    # Check for molSimplify failure (returns [0] on error)
    if len(full_names) == 1 and full_names[0] == 0:
        warnings.warn(
            f"molSimplify failed for {cif_name} (atomic overlap or invalid structure)"
        )
        return pd.DataFrame()

    full_names.append("filename")
    full_descriptors.append(cif_name)
    featurization = dict(zip(full_names, full_descriptors))
    df = pd.DataFrame([featurization])

    # Filter out failed rows (where descriptor value is 0)
    if "0" in df.columns:
        df = df[df["0"] != 0]

    return df if len(df) > 0 else pd.DataFrame()  # type: ignore[return-value]


def cif_to_zeo(cif_path: str, zeopp_bin: str, work_dir: str) -> tuple[dict, dict, dict]:
    os.makedirs(work_dir, exist_ok=True)
    cif_name = Path(cif_path).name
    cif_stem = cif_name.replace(".cif", "")
    cif_dir = str(Path(cif_path).parent.resolve())

    sa_out = os.path.join(cif_dir, f"{cif_stem}.sa")
    vol_out = os.path.join(cif_dir, f"{cif_stem}.vol")
    res_out = os.path.join(cif_dir, f"{cif_stem}.res")

    subprocess.run(
        [zeopp_bin, "-ha", "-sa", "1.2", "1.2", "5000", cif_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    subprocess.run(
        [zeopp_bin, "-ha", "-vol", "1.2", "1.2", "50000", cif_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    subprocess.run(
        [zeopp_bin, "-ha", "-res", cif_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )

    sa = {}
    if os.path.exists(sa_out):
        with open(sa_out) as f:
            for line in f:
                if "ASA_A^2:" in line:
                    name = cif_stem
                    sa[name] = {
                        "sa": line.split("ASA_A^2:")[1].split()[0],
                        "cv": line.split("Unitcell_volume:")[1].split()[0],
                        "density": line.split("Density:")[1].split()[0],
                    }

    pv = {}
    if os.path.exists(vol_out):
        with open(vol_out) as f:
            for line in f:
                if "AV_Volume_fraction:" in line:
                    name = cif_stem
                    pv[name] = {"vf": line.split("AV_Volume_fraction:")[1].split()[0]}

    res = {}
    if os.path.exists(res_out):
        with open(res_out) as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 4:
                    name = parts[0].split("/")[-1].split(".")[0]
                    res[name] = {
                        "di": parts[1],
                        "df": parts[2],
                        "dif": parts[3],
                    }

    return sa, pv, res


def get_all_descriptors(
    cif_path: str | None = None,
    cif_dir: str | None = None,
    zeopp_bin: str | None = None,
    output_path: str | None = None,
    work_dir: str | None = None,
    rac_only: bool = False,
) -> pd.DataFrame:
    """Calculate RAC + Zeo++ descriptors for MOF CIF files.

    Args:
        cif_path: Single CIF file path
        cif_dir: Directory containing CIF files
        zeopp_bin: Path to Zeo++ network binary (auto-detected if None)
        output_path: Path to save CSV output
        work_dir: Working directory for intermediate files
        rac_only: If True, skip Zeo++ descriptors (for when Zeo++ not installed)

    Returns:
        DataFrame with RAC (+ Zeo++) descriptors
    """
    if cif_path is None and cif_dir is None:
        raise ValueError("Provide either cif_path or cif_dir")
    if cif_path is not None and cif_dir is not None:
        raise ValueError("Provide only one of cif_path or cif_dir, not both")

    if work_dir is None:
        work_dir = str(Path(__file__).parent / "work")
    os.makedirs(work_dir, exist_ok=True)

    # Find Zeo++ binary
    use_zeopp = not rac_only
    if not rac_only:
        if zeopp_bin is None:
            zeopp_bin = ensure_zeopp()
        elif (
            zeopp_bin and not os.path.exists(zeopp_bin) and not shutil.which(zeopp_bin)
        ):
            print(f"[WARNING] zeopp_bin not found: {zeopp_bin}")
            print("[INFO] Falling back to RAC-only mode")
            use_zeopp = False

    if cif_path:
        cif_files = [cif_path]
    else:
        assert cif_dir is not None
        cif_dir_path = Path(cif_dir)
        cif_files = sorted(cif_dir_path.glob("*.cif"))

    rac_dfs = []
    all_sa, all_pv, all_res = {}, {}, {}

    for cif_path_item in tqdm(cif_files, desc="Calculating descriptors"):
        cif_path_str = str(cif_path_item)
        cif_stem = Path(cif_path_str).stem
        cif_work = os.path.join(work_dir, cif_stem)
        os.makedirs(cif_work, exist_ok=True)

        rac_df = cif_to_rac(cif_path_str, cif_work)
        if len(rac_df) > 0:
            rac_dfs.append(rac_df)

        if use_zeopp:
            assert zeopp_bin is not None
            sa, pv, res = cif_to_zeo(cif_path_str, zeopp_bin, cif_work)
            all_sa.update(sa)
            all_pv.update(pv)
            all_res.update(res)

    if not rac_dfs:
        warnings.warn("No CIF files produced valid descriptors")
        return pd.DataFrame()

    final_df = pd.concat(rac_dfs, ignore_index=True)

    if use_zeopp:
        rows = []
        for i, row in tqdm(
            final_df.iterrows(), total=len(final_df), desc="Merging descriptors"
        ):
            cif_name = row.get("filename", "")
            if isinstance(cif_name, str) and cif_name.endswith(".cif"):
                cif_stem = cif_name[:-4]
            else:
                cif_stem = str(cif_name)

            merged = list(row)
            merged.extend(
                [
                    float(all_sa.get(cif_stem, {}).get("sa", np.nan)),
                    float(all_sa.get(cif_stem, {}).get("cv", np.nan)),
                    float(all_sa.get(cif_stem, {}).get("density", np.nan)),
                    float(all_pv.get(cif_stem, {}).get("vf", np.nan)),
                    float(all_res.get(cif_stem, {}).get("di", np.nan)),
                    float(all_res.get(cif_stem, {}).get("df", np.nan)),
                    float(all_res.get(cif_stem, {}).get("dif", np.nan)),
                    cif_name,
                ]
            )
            rows.append(merged)

        cols = list(final_df.columns) + [
            "sa",
            "cv",
            "density",
            "vf",
            "di",
            "df",
            "dif",
            "filename",
        ]
        result = pd.DataFrame(rows, columns=cols)  # type: ignore[arg-type]
    else:
        # RAC only - filename already added by cif_to_rac
        result = final_df

    if output_path:
        result.to_csv(output_path, index=False)
        print(f"[OK] Saved to {output_path}")

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Calculate RAC + ZEO descriptors from CIF files"
    )
    parser.add_argument("--cif", help="Single CIF file path")
    parser.add_argument("--cif_dir", help="Directory containing CIF files")
    parser.add_argument("--zeopp_bin", help="Path to zeo++ network binary")
    parser.add_argument("--output", help="Output CSV path")
    args = parser.parse_args()

    cif_dir_default = str(__desc_dir__ / "cif_opt")
    output_default = str(__desc_dir__ / "examples" / "descriptors.csv")
    cif_dir_arg = args.cif_dir or cif_dir_default
    output_arg = args.output or output_default

    result = get_all_descriptors(
        cif_path=args.cif,
        cif_dir=cif_dir_arg,
        zeopp_bin=args.zeopp_bin,
        output_path=output_arg,
    )
    print(f"[OK] {len(result)} rows, {len(result.columns)} columns")
