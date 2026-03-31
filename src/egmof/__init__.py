import os

__root_dir__ = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from .egmof import EGMOF

__all__ = ["EGMOF", "__root_dir__"]
