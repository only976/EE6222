from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.datasets import fetch_olivetti_faces, fetch_openml

DatasetName = Literal["mnist", "olivetti_faces"]


@dataclass(frozen=True)
class DatasetBundle:
    name: str
    X: np.ndarray
    y: np.ndarray
    n_classes: int
    input_dim: int


def _as_float64_2d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim != 2:
        X = X.reshape(X.shape[0], -1)
    if X.dtype != np.float64:
        X = X.astype(np.float64, copy=False)
    return X


def load_dataset(
    name: DatasetName,
    *,
    limit: int | None = None,
    random_state: int = 0,
) -> DatasetBundle:
    """Load a high-dimensional labeled dataset and return a normalized 2D feature matrix.

    Notes
    -----
    - This function only loads data and does *not* split it.
    - Any sub-sampling is random but deterministic via `random_state`.
    """

    if name == "mnist":
        # 70k samples, 784 dims. Labels are strings on OpenML.
        mnist = fetch_openml("mnist_784", version=1, as_frame=False)
        X = _as_float64_2d(mnist.data)
        y = np.asarray(mnist.target)
        # Convert labels to int if possible
        try:
            y = y.astype(np.int64)
        except Exception:
            pass
        # Scale pixel range to [0, 1] for more stable optimization.
        # (Standardization will happen later, fit on training only.)
        if X.max() > 1.0:
            X = X / 255.0
        n_classes = int(len(np.unique(y)))

    elif name == "olivetti_faces":
        faces = fetch_olivetti_faces(shuffle=True, random_state=random_state)
        X = _as_float64_2d(faces.data)  # already normalized to [0, 1]
        y = np.asarray(faces.target, dtype=np.int64)
        n_classes = int(len(np.unique(y)))

    else:
        raise ValueError(f"Unknown dataset: {name}")

    if limit is not None and limit < X.shape[0]:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(X.shape[0], size=limit, replace=False)
        X = X[idx]
        y = y[idx]

    return DatasetBundle(
        name=name,
        X=X,
        y=y,
        n_classes=n_classes,
        input_dim=int(X.shape[1]),
    )
