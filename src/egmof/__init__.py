import os

__root_dir__ = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from .egmof import EGMOF
from .utils import (
    download_desc2mof,
    download_mof2desc,
    download_prop2desc,
    download_rf,
    download_all,
)

__all__ = [
    "EGMOF",
    "__root_dir__",
    "download_desc2mof",
    "download_mof2desc",
    "download_prop2desc",
    "download_rf",
    "download_all",
]
