from __future__ import annotations

import os

__egmof_dir__ = os.path.dirname(os.path.abspath(__file__))
__src_dir__ = os.path.dirname(__egmof_dir__)
__root_dir__ = os.path.dirname(__src_dir__)
__desc2mof_dir__ = os.path.join(__egmof_dir__, "desc2mof")

DEFAULT_DESC2MOF_CKPT = os.path.join(
    __root_dir__, "checkpoints", "desc2mof", "desc2mof_best.ckpt"
)
DEFAULT_MOF2DESC_CKPT = os.path.join(
    __root_dir__, "checkpoints", "mof2desc", "mof2desc_best.ckpt"
)
DEFAULT_PROP2DESC_CONFIG = os.path.join(
    __root_dir__, "config", "prop2desc_training_config.yaml"
)
DEFAULT_DESC2MOF_CONFIG = os.path.join(
    __root_dir__, "config", "desc2mof_training_config.yaml"
)
DEFAULT_MOF2DESC_CONFIG = os.path.join(
    __root_dir__, "config", "mof2desc_training_config.yaml"
)
DEFAULT_DESC2MOF_MEAN = os.path.join(__desc2mof_dir__, "data", "mean_all.csv")
DEFAULT_DESC2MOF_STD = os.path.join(__desc2mof_dir__, "data", "std_all.csv")
DEFAULT_DESC2MOF_FEATURE_NAME = os.path.join(
    __desc2mof_dir__, "data", "feature_name.txt"
)
