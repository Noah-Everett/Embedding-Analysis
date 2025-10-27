import os
import random
from typing import Optional

import numpy as np
import torch


def get_device() -> torch.device:
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available() and getattr(torch.backends.mps, 'is_built', lambda: True)():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def next_run_path(base_dir: str = "runs", prefix: str = "run_", ensure_exists: bool = True) -> str:
    os.makedirs(base_dir, exist_ok=True)
    n = -1
    run_path = ""
    while os.path.exists(run_path) or n < 0:
        n += 1
        run_path = os.path.join(base_dir, f"{prefix}{n}")
    if ensure_exists:
        os.makedirs(run_path, exist_ok=True)
    return run_path
