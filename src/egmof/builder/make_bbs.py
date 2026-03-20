import os
import shutil
import argparse
from pathlib import Path

from tqdm.auto import tqdm

from egmof import __root_dir__
from egmof.builder.selfies2bb import decode_selfies_to_xyz_opt


XTB_VERSION = "6.7.1"
XTB_URL = f"https://github.com/grimme-lab/xtb/releases/download/v{XTB_VERSION}/xtb-{XTB_VERSION}-linux-x86_64.tar.xz"


def find_xtb_bin() -> str | None:
    builder_dir = Path(__file__).parent
    local_bin = builder_dir / "xtb-dist" / "bin" / "xtb"
    if local_bin.exists():
        return str(local_bin)
    if shutil.which("xtb"):
        return "xtb"
    return None


def ensure_xtb_installed(force: bool = False) -> str:
    builder_dir = Path(__file__).parent
    local_bin = builder_dir / "xtb-dist" / "bin" / "xtb"
    if local_bin.exists() and not force:
        return str(local_bin)
    print(f"[INFO] xTB not found. Downloading from {XTB_URL}")
    archive = Path("/tmp") / f"xtb-{XTB_VERSION}-linux-x86_64.tar.xz"
    try:
        import urllib.request
        urllib.request.urlretrieve(XTB_URL, archive)
        import tarfile
        with tarfile.open(archive, "r:xz") as tar:
            tar.extractall(builder_dir)
        archive.unlink(missing_ok=True)
        print(f"[INFO] xTB installed to {local_bin}")
        return str(local_bin)
    except Exception as e:
        raise RuntimeError(
            f"Failed to download/install xTB. "
            f"Install manually: wget {XTB_URL} && tar -xf xtb-{XTB_VERSION}-linux-x86_64.tar.xz"
        ) from e


def make_bbs(
    selfies_list,
    run_dir: str | None = None,
    engine: str = "xtb",
    max_workers: int = 1,
) -> tuple[list, dict]:
    if run_dir is None:
        run_dir = str(Path(__file__).parent / "new_bbs")
    os.makedirs(run_dir, exist_ok=True)

    xtb_bin = None
    if engine == "xtb":
        xtb_bin = find_xtb_bin() or ensure_xtb_installed()
    else:
        xtb_bin = "xtb"

    e_counter, n_counter = 0, 0
    failed_selfies = []
    selfies_to_name = {}

    for ith, selfies_str in enumerate(tqdm(selfies_list, desc="Generating BBs")):
        cn = selfies_str.count("[Lr]")

        if cn == 2:
            bb_type, num = "E", e_counter + 1
            out_name = f"Custom_E{num}.xyz"
        elif cn > 2:
            bb_type, num = "N", n_counter + 1
            out_name = f"Custom_N{num}.xyz"
        else:
            failed_selfies.append(selfies_str)
            print(f"[SKIP] ith={ith} unexpected [Lr] count={cn}")
            continue

        save_path = os.path.join(run_dir, out_name)

        if os.path.exists(save_path):
            selfies_to_name[selfies_str] = out_name[:-4]
            if bb_type == "E":
                e_counter += 1
            else:
                n_counter += 1
            continue

        try:
            _ = decode_selfies_to_xyz_opt(
                selfies_str,
                run_dir=run_dir,
                out_xyz_name=out_name,
                xtb_bin=xtb_bin,
                engine=engine,
            )
            selfies_to_name[selfies_str] = out_name[:-4]
            if bb_type == "E":
                e_counter += 1
            else:
                n_counter += 1
        except Exception as err:
            failed_selfies.append(selfies_str)
            print(f"[FAIL] ith={ith} name={out_name} err={err}")

    return failed_selfies, selfies_to_name


def main():
    parser = argparse.ArgumentParser(description="Batch generate building blocks from SELFIES")
    parser.add_argument("input", help="Pickle file with list of SELFIES strings")
    parser.add_argument("--run_dir", default=None, help="Output directory")
    parser.add_argument("--engine", default="xtb", choices=["xtb", "mmff", "uff"], help="Optimization engine")
    parser.add_argument("--no_auto_install", action="store_true", help="Skip auto-download of xtb")
    args = parser.parse_args()

    import pickle
    with open(args.input, "rb") as f:
        selfies_list = pickle.load(f)

    print(f"[INFO] Loaded {len(selfies_list)} SELFIES strings")
    print(f"[INFO] Engine: {args.engine}")

    failed, mapping = make_bbs(selfies_list, run_dir=args.run_dir, engine=args.engine)

    print(f"\n[RESULT] Generated: {len(mapping)} | Failed: {len(failed)}")
    if failed:
        print(f"[WARN] {len(failed)} SELFIES failed")
        failed_path = os.path.join(args.run_dir or str(Path(__file__).parent / "new_bbs"), "failed_selfies.pkl")
        with open(failed_path, "wb") as f:
            pickle.dump(failed, f)
        print(f"[INFO] Failed list saved to: {failed_path}")


if __name__ == "__main__":
    main()
