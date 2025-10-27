from typing import Callable, Tuple

import numpy as np
import torch
from torch.utils.data import IterableDataset, TensorDataset


class OnlineFunctionDataset(IterableDataset):
    """
    Iterable dataset that samples inputs uniformly in [-R, R]^in_dim each epoch
    and computes targets via a provided function.

    target_fn: Callable[[torch.Tensor], torch.Tensor]
      - Expects input shape [N, in_dim], returns output shape [N, out_dim]
    """

    def __init__(
        self,
        n: int,
        in_dim: int,
        target_fn: Callable[[torch.Tensor], torch.Tensor],
        data_range: float = 3.0,
        shuffle: bool = True,
        dtype: np.dtype = np.float32,
    ) -> None:
        super().__init__()
        self.n = int(n)
        self.in_dim = int(in_dim)
        self.target_fn = target_fn
        self.data_range = float(data_range)
        self.shuffle = bool(shuffle)
        self.dtype = dtype

    def __iter__(self):
        X = np.random.uniform(-self.data_range, self.data_range, size=(self.n, self.in_dim)).astype(self.dtype)
        xb = torch.from_numpy(X)
        with torch.no_grad():
            yb = self.target_fn(xb)
            if not isinstance(yb, torch.Tensor):
                yb = torch.as_tensor(yb)
            if yb.dim() == 1:
                yb = yb.unsqueeze(-1)
            yb = yb.to(dtype=xb.dtype)

        idx = np.arange(self.n)
        if self.shuffle:
            np.random.shuffle(idx)
        for i in idx:
            yield xb[i], yb[i]


def make_fixed_val_dataset(
    n_val: int,
    in_dim: int,
    target_fn: Callable[[torch.Tensor], torch.Tensor],
    data_range: float = 3.0,
    dtype: np.dtype = np.float32,
) -> TensorDataset:
    X = np.random.uniform(-data_range, data_range, size=(int(n_val), int(in_dim))).astype(dtype)
    xb = torch.from_numpy(X)
    with torch.no_grad():
        yb = target_fn(xb)
        if not isinstance(yb, torch.Tensor):
            yb = torch.as_tensor(yb)
        if yb.dim() == 1:
            yb = yb.unsqueeze(-1)
        yb = yb.to(dtype=xb.dtype)
    return TensorDataset(xb, yb)
