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

ZEO_URL = "http://www.zeoplusplus.org/download.html"


def find_zeopp_bin() -> str | None:
    desc_dir = Path(__file__).parent
    local_bin = desc_dir / "zeopp" / "network"
    if local_bin.exists():
        return str(local_bin)
    if shutil.which("network"):
        return "network"
    return None


def install_zeopp(force: bool = False) -> str:
    desc_dir = Path(__file__).parent
    local_bin = desc_dir / "zeopp" / "network"
    if local_bin.exists() and not force:
        return str(local_bin)

    print("[INFO] Zeo++ binary not found.")
    print(f"[INFO] Download from: {ZEO_URL} (registration required)")
    print("[INFO] After download, extract and place 'network' binary at:")
    print(f"       {local_bin}")
    print("[INFO] Or add 'network' to your PATH.")
    raise RuntimeError(
        "Zeo++ not installed. Download from http://www.zeoplusplus.org/download.html "
        "and place the 'network' binary."
    )


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
    sprim.to(os.path.join(primitive_dir, cif_name))

    full_names, full_descriptors = get_MOF_descriptors(
        os.path.join(primitive_dir, cif_name),
        3,
        path=work_dir,
        xyzpath=os.path.join(work_dir, f"{cif_stem}.xyz"),
    )
    full_names.append("filename")
    full_descriptors.append(cif_name)
    featurization = dict(zip(full_names, full_descriptors))
    df = pd.DataFrame([featurization])
    df = df[df["0"] != 0]
    return df.iloc[:, :-1]


def cif_to_zeo(cif_path: str, zeopp_bin: str, work_dir: str) -> tuple[dict, dict, dict]:
    os.makedirs(work_dir, exist_ok=True)
    cif_name = Path(cif_path).name
    cif_stem = cif_name.replace(".cif", "")

    sa_out = os.path.join(work_dir, f"{cif_stem}.sa")
    vol_out = os.path.join(work_dir, f"{cif_stem}.vol")
    res_out = os.path.join(work_dir, f"{cif_stem}.res")

    subprocess.run(
        [zeopp_bin, cif_path, sa_out],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    subprocess.run(
        [zeopp_bin, "-pb", "1.4", cif_path, vol_out],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    subprocess.run(
        [zeopp_bin, "-res", cif_path, res_out],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )

    sa = {}
    if os.path.exists(sa_out):
        with open(sa_out) as f:
            for line in f:
                if "Density" in line:
                    parts = line.split()
                    name = parts[1].split("/")[-1].split(".")[0]
                    sa[name] = {
                        "sa": parts[9],
                        "cv": parts[3],
                        "density": parts[5],
                    }

    pv = {}
    if os.path.exists(vol_out):
        with open(vol_out) as f:
            for line in f:
                if "AV_Volume_fraction" in line:
                    parts = line.split()
                    name = parts[1].split("/")[-1].split(".")[0]
                    pv[name] = {"vf": parts[9]}

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
) -> pd.DataFrame:
    if cif_path is None and cif_dir is None:
        raise ValueError("Provide either cif_path or cif_dir")
    if cif_path is not None and cif_dir is not None:
        raise ValueError("Provide only one of cif_path or cif_dir, not both")

    if work_dir is None:
        work_dir = str(Path(__file__).parent / "work")
    os.makedirs(work_dir, exist_ok=True)

    if zeopp_bin is None:
        zeopp_bin = ensure_zeopp()
    elif zeopp_bin and not os.path.exists(zeopp_bin) and not shutil.which(zeopp_bin):
        raise FileNotFoundError(f"zeopp_bin not found: {zeopp_bin}")

    if cif_path:
        cif_files = [cif_path]
    else:
        cif_dir = Path(cif_dir)
        cif_files = sorted(cif_dir.glob("*.cif"))

    rac_dfs = []
    all_sa, all_pv, all_res = {}, {}, {}

    for cif_path_item in tqdm(cif_files, desc="Calculating descriptors"):
        cif_stem = Path(cif_path_item).stem
        cif_work = os.path.join(work_dir, cif_stem)
        os.makedirs(cif_work, exist_ok=True)

        rac_df = cif_to_rac(cif_path_item, cif_work)
        rac_dfs.append(rac_df)

        sa, pv, res = cif_to_zeo(cif_path_item, zeopp_bin, cif_work)
        all_sa.update(sa)
        all_pv.update(pv)
        all_res.update(res)

    if not rac_dfs:
        raise ValueError("No CIF files found")

    final_df = pd.concat(rac_dfs, ignore_index=True)

    rows = []
    for i, row in tqdm(final_df.iterrows(), total=len(final_df), desc="Merging descriptors"):
        cif_name = row.get("filename", "")
        if isinstance(cif_name, str) and cif_name.endswith(".cif"):
            cif_stem = cif_name[:-4]
        else:
            cif_stem = str(cif_name)

        merged = list(row)
        merged.extend([
            float(all_sa.get(cif_stem, {}).get("sa", np.nan)),
            float(all_sa.get(cif_stem, {}).get("cv", np.nan)),
            float(all_sa.get(cif_stem, {}).get("density", np.nan)),
            float(all_pv.get(cif_stem, {}).get("vf", np.nan)),
            float(all_res.get(cif_stem, {}).get("di", np.nan)),
            float(all_res.get(cif_stem, {}).get("df", np.nan)),
            float(all_res.get(cif_stem, {}).get("dif", np.nan)),
            cif_name,
        ])
        rows.append(merged)

    cols = list(final_df.columns) + ["sa", "cv", "density", "vf", "di", "df", "dif", "filename"]
    result = pd.DataFrame(rows, columns=cols)

    if output_path:
        result.to_csv(output_path, index=False)
        print(f"[OK] Saved to {output_path}")

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Calculate RAC + ZEO descriptors from CIF files")
    parser.add_argument("--cif", help="Single CIF file path")
    parser.add_argument("--cif_dir", help="Directory containing CIF files")
    parser.add_argument("--zeopp_bin", help="Path to zeo++ network binary")
    parser.add_argument("--output", help="Output CSV path")
    args = parser.parse_args()

    cif_dir_default = str(Path(__file__).parent / "cif")
    cif_dir_arg = args.cif_dir or cif_dir_default

    result = get_all_descriptors(
        cif_path=args.cif,
        cif_dir=cif_dir_arg,
        zeopp_bin=args.zeopp_bin,
        output_path=args.output,
    )
    print(f"[OK] {len(result)} rows, {len(result.columns)} columns")
