import os

from .egmof import EGMOF

__root_dir__ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

__all__ = ["EGMOF", "__root_dir__"]