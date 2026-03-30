import os
import shutil
import argparse
import re
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


def get_counters(run_dir: str) -> tuple[int, int]:
    """Parse existing Custom_E*.xyz / Custom_N*.xyz files to find next counter values."""
    e_counter, n_counter = 0, 0
    if not os.path.isdir(run_dir):
        return e_counter, n_counter
    for fname in os.listdir(run_dir):
        m = re.match(r"Custom_E(\d+)\.xyz", fname)
        if m:
            e_counter = max(e_counter, int(m.group(1)))
        m = re.match(r"Custom_N(\d+)\.xyz", fname)
        if m:
            n_counter = max(n_counter, int(m.group(1)))
    return e_counter, n_counter


def make_bb(
    selfies_str: str,
    run_dir: str,
    engine: str = "xtb",
    xtb_bin: str | None = None,
) -> tuple[bool, str]:
    cn = selfies_str.count("[Lr]")
    if cn < 2:
        print(f"[SKIP] [Lr] count={cn} (expected >=2)")
        return False, ""

    e_counter, n_counter = get_counters(run_dir)

    if cn == 2:
        bb_type, num = "E", e_counter + 1
    else:
        bb_type, num = "N", n_counter + 1

    out_name = f"Custom_{bb_type}{num}.xyz"
    save_path = os.path.join(run_dir, out_name)

    if os.path.exists(save_path):
        print(f"[SKIP] Already exists: {out_name}")
        return True, out_name[:-4]

    if xtb_bin is None:
        if engine == "xtb":
            xtb_bin = find_xtb_bin() or ensure_xtb_installed()
        else:
            xtb_bin = "xtb"

    try:
        _ = decode_selfies_to_xyz_opt(
            selfies_str,
            run_dir=run_dir,
            out_xyz_name=out_name,
            xtb_bin=xtb_bin,
            engine=engine,
        )
        return True, out_name[:-4]
    except Exception as err:
        print(f"[FAIL] {out_name} — {err}")
        return False, ""


def main():
    parser = argparse.ArgumentParser(description="Generate building block XYZ from SELFIES")
    parser.add_argument("selfies", help="SELFIES string")
    parser.add_argument("--run_dir", default=None, help="Output directory (default: builder/new_bbs)")
    parser.add_argument("--engine", default="xtb", choices=["xtb", "mmff", "uff"], help="Optimization engine")
    args = parser.parse_args()

    if args.run_dir is None:
        args.run_dir = os.path.join(__root_dir__, "builder", "new_bbs")
    os.makedirs(args.run_dir, exist_ok=True)

    print(f"[INFO] SELFIES: {args.selfies}")
    print(f"[INFO] run_dir: {args.run_dir}")
    print(f"[INFO] engine: {args.engine}")

    success, bb_name = make_bb(args.selfies, args.run_dir, args.engine)

    if success:
        print(f"\n[OK] {bb_name}.xyz")
    else:
        print(f"\n[FAIL] Generation failed")


if __name__ == "__main__":
    main()
