from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import numpy as np


@dataclass
class ProbeData:
    x: np.ndarray   # [N, in_dim]
    z: np.ndarray   # [N, embed_dim]
    pred: np.ndarray  # [N, out_dim]
    y: Optional[np.ndarray] = None  # optional ground truth

    @property
    def in_dim(self) -> int:
        return int(self.x.shape[1])

    @property
    def embed_dim(self) -> int:
        return int(self.z.shape[1])

    @property
    def out_dim(self) -> int:
        return int(self.pred.shape[1]) if self.pred.ndim == 2 else 1

    @property
    def n(self) -> int:
        return int(self.x.shape[0])


def load_probe(npz_path: str) -> ProbeData:
    data = np.load(npz_path)
    x = data["x"]
    z = data["z"]
    pred = data["pred"]
    y = data["y"] if "y" in data.files else None
    # Normalize shapes
    if pred.ndim == 1:
        pred = pred[:, None]
    if y is not None and y.ndim == 1:
        y = y[:, None]
    return ProbeData(x=x, z=z, pred=pred, y=y)


def compute_norms(arr: np.ndarray) -> np.ndarray:
    return np.linalg.norm(arr, axis=1)


def pca_2d(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (proj, explained_var_ratio)
    - proj: [N,2] projection
    - explained_var_ratio: [2] variance explained by components
    """
    X = arr - arr.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    comps = Vt[:2]
    proj = X @ comps.T
    var = (S**2) / (len(X) - 1)
    total = var.sum()
    evr = var[:2] / total if total > 0 else np.array([0.0, 0.0])
    return proj, evr


def corr(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation, safe for constant arrays."""
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    if x.size != y.size or x.size == 0:
        return np.nan
    x_std = x.std()
    y_std = y.std()
    if x_std == 0 or y_std == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def basic_metrics(probe: ProbeData) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    xr = compute_norms(probe.x)
    zr = compute_norms(probe.z)
    metrics["corr_norm_x_z"] = corr(xr, zr)
    if probe.y is not None:
        # L2 error per-sample, averaged
        err = np.linalg.norm(probe.pred - probe.y, axis=1)
        metrics["mean_l2_error"] = float(np.mean(err))
        metrics["median_l2_error"] = float(np.median(err))
    return metrics
