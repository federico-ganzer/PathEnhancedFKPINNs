"""Small helpers."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed python, numpy and torch (CPU + CUDA). `deterministic` also disables cuDNN autotune.

    For path sampling that must not disturb the global RNG, use a separate
    `torch.Generator().manual_seed(s)` and pass it as `generator=`.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
