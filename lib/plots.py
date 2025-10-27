from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


def scatter_2d(points: np.ndarray, *, c: Optional[np.ndarray] = None, title: str = "", xlabel: str = "x", ylabel: str = "y", cmap: str = "viridis", s: int = 5):
    fig, ax = plt.subplots(figsize=(6, 6))
    if c is None:
        ax.scatter(points[:, 0], points[:, 1], s=s)
    else:
        sc = ax.scatter(points[:, 0], points[:, 1], c=c, s=s, cmap=cmap)
        plt.colorbar(sc, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.axis('equal')
    return fig, ax


def hist(values: np.ndarray, *, bins: int = 50, title: str = "", xlabel: str = "value"):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(values, bins=bins)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    return fig, ax


def explained_variance_bar(evr: np.ndarray, *, title: str = "Explained variance (PCA)"):
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(["PC1", "PC2"], evr[:2])
    ax.set_ylim(0, 1)
    ax.set_ylabel("ratio")
    ax.set_title(title)
    return fig, ax
