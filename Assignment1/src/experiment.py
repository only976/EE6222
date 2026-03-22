from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .datasets import DatasetBundle
from .mahalanobis import MinimumMahalanobisClassifier

ReducerName = Literal["none", "pca", "lda", "pca_lda", "pca75_lda", "pca90_lda", "pca95_lda", "pca99_lda"]
ClassifierName = Literal["knn", "mahalanobis", "logreg", "rf"]


@dataclass(frozen=True)
class ExperimentConfig:
    reducer: ReducerName
    classifier: ClassifierName
    test_size: float = 0.3
    random_state: int = 0
    standardize: bool = True
    knn_k: int = 1
    pca_solver: str = "randomized"
    rf_n_estimators: int = 200


def _make_classifier(cfg: ExperimentConfig):
    if cfg.classifier == "knn":
        return KNeighborsClassifier(n_neighbors=cfg.knn_k)
    if cfg.classifier == "mahalanobis":
        return MinimumMahalanobisClassifier()
    if cfg.classifier == "logreg":
        return OneVsRestClassifier(
            LogisticRegression(
                solver="liblinear",
                max_iter=5000,
            )
        )
    if cfg.classifier == "rf":
        return RandomForestClassifier(
            n_estimators=int(cfg.rf_n_estimators),
            random_state=cfg.random_state,
            n_jobs=-1,
        )
    raise ValueError(f"Unknown classifier: {cfg.classifier}")


def _make_pipeline(cfg: ExperimentConfig, *, n_components: int | None, n_classes: int):
    return _make_pipeline_with_pca_pre_dim(
        cfg,
        n_components=n_components,
        n_classes=n_classes,
        pca_pre_dim=None,
    )


def _make_pipeline_with_pca_pre_dim(
    cfg: ExperimentConfig,
    *,
    n_components: int | None,
    n_classes: int,
    pca_pre_dim: int | None,
):
    steps: list[tuple[str, object]] = []
    if cfg.standardize:
        steps.append(("scaler", StandardScaler()))

    if cfg.reducer == "none":
        pass

    elif cfg.reducer == "pca":
        if n_components is None:
            raise ValueError("n_components required for PCA")
        steps.append(
            (
                "pca",
                PCA(
                    n_components=n_components,
                    svd_solver=cfg.pca_solver,
                    random_state=cfg.random_state,
                ),
            )
        )

    elif cfg.reducer == "lda":
        if n_components is None:
            raise ValueError("n_components required for LDA")
        # LDA dimension <= C-1 enforced by caller.
        steps.append(("lda", LinearDiscriminantAnalysis(n_components=n_components)))

    elif cfg.reducer == "pca_lda":
        if n_components is None:
            raise ValueError("n_components required for PCA+LDA")
        # Convention here: n_components refers to the *final* LDA dimension.
        # We choose a PCA pre-dimension high enough but still efficient.
        lda_dim = n_components
        pca_dim = int(pca_pre_dim) if pca_pre_dim is not None else max(lda_dim * 3, min(200, 500))
        # PCA dim cannot exceed input dim; handled during fit by sklearn.
        steps.append(
            (
                "pca",
                PCA(
                    n_components=pca_dim,
                    svd_solver=cfg.pca_solver,
                    random_state=cfg.random_state,
                ),
            )
        )
        steps.append(("lda", LinearDiscriminantAnalysis(n_components=lda_dim)))

    else:
        raise ValueError(f"Unknown reducer: {cfg.reducer}")

    steps.append(("clf", _make_classifier(cfg)))
    return Pipeline(steps)


def default_dims(dataset: DatasetBundle, reducer: ReducerName) -> list[int]:
    if reducer == "none":
        return [dataset.input_dim]

    if reducer == "pca":
        # A compact sweep that usually shows the trend clearly.
        candidates = [1, 2, 5, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200, 300]
        return [d for d in candidates if d < dataset.input_dim]

    if reducer in {"lda", "pca_lda", "pca75_lda", "pca90_lda", "pca95_lda", "pca99_lda"}:
        max_d = max(1, dataset.n_classes - 1)
        # For many classes (MNIST: 10), this gives 1..9. For faces: 1..39.
        if max_d <= 15:
            return list(range(1, max_d + 1))
        # Otherwise, sample more sparsely at higher dims.
        base = list(range(1, 11))
        tail = [15, 20, 25, 30, 35, max_d]
        dims = sorted(set([d for d in base + tail if 1 <= d <= max_d]))
        return dims

    raise ValueError(f"Unknown reducer: {reducer}")


def run_dimension_sweep(
    dataset: DatasetBundle,
    cfg: ExperimentConfig,
    dims: Iterable[int] | None = None,
) -> pd.DataFrame:
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.X,
        dataset.y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=dataset.y,
    )

    if dims is None:
        dims_list = default_dims(dataset, cfg.reducer)
    else:
        dims_list = list(dims)

    if cfg.reducer in {"pca75_lda", "pca90_lda", "pca95_lda", "pca99_lda"}:
        if cfg.reducer == "pca75_lda":
            variance = 0.75
        elif cfg.reducer == "pca90_lda":
            variance = 0.90
        elif cfg.reducer == "pca95_lda":
            variance = 0.95
        else:
            variance = 0.99
        return _run_pca_variance_lda_sweep(
            dataset_name=dataset.name,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            cfg=cfg,
            dims_list=dims_list,
            pca_variance=variance,
            n_classes=dataset.n_classes,
        )

    rows: list[dict] = []
    max_components = int(min(X_train.shape[0], X_train.shape[1]))
    for d in dims_list:
        pca_pre_dim: int | None = None

        if cfg.reducer == "none":
            n_comp = None

        elif cfg.reducer == "pca":
            n_comp = min(int(d), max_components)
            n_comp = max(1, n_comp)

        elif cfg.reducer == "lda":
            n_comp = min(int(d), dataset.n_classes - 1)
            n_comp = max(1, n_comp)

        elif cfg.reducer == "pca_lda":
            lda_dim = min(int(d), dataset.n_classes - 1)
            lda_dim = max(1, lda_dim)
            lda_dim = min(lda_dim, max_components)
            # Choose PCA pre-dimension capped by training-set limits.
            pca_pre_dim = max(lda_dim, min(max(lda_dim * 3, 200), max_components))
            n_comp = lda_dim

        else:
            raise ValueError(f"Unknown reducer: {cfg.reducer}")

        pipe = _make_pipeline_with_pca_pre_dim(
            cfg,
            n_components=n_comp,
            n_classes=dataset.n_classes,
            pca_pre_dim=pca_pre_dim,
        )
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))

        rows.append(
            {
                "dataset": dataset.name,
                "reducer": cfg.reducer,
                "classifier": cfg.classifier,
                "dim": int(n_comp if cfg.reducer != "none" else dataset.input_dim),
                "accuracy": acc,
                "error_rate": 1.0 - acc,
                "pca_dim": int(pca_pre_dim) if pca_pre_dim is not None else "",
                "n_train": int(X_train.shape[0]),
                "n_test": int(X_test.shape[0]),
                "random_state": int(cfg.random_state),
                "test_size": float(cfg.test_size),
            }
        )

    return pd.DataFrame(rows).sort_values("dim")


def _run_pca_variance_lda_sweep(
    *,
    dataset_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cfg: ExperimentConfig,
    dims_list: list[int],
    pca_variance: float,
    n_classes: int,
) -> pd.DataFrame:
    # Fit scaler on training only
    if cfg.standardize:
        scaler = StandardScaler().fit(X_train)
        X_train_s = scaler.transform(X_train)
        X_test_s = scaler.transform(X_test)
    else:
        X_train_s = X_train
        X_test_s = X_test

    # PCA to keep a target explained variance ratio (fit on training only)
    # sklearn requires svd_solver='full' for float n_components.
    pca = PCA(n_components=float(pca_variance), svd_solver="full", random_state=cfg.random_state)
    X_train_p = pca.fit_transform(X_train_s)
    X_test_p = pca.transform(X_test_s)
    pca_dim = int(X_train_p.shape[1])

    max_lda_dim = int(min(n_classes - 1, pca_dim))
    rows: list[dict] = []
    for d in dims_list:
        lda_dim = max(1, min(int(d), max_lda_dim))
        lda = LinearDiscriminantAnalysis(n_components=lda_dim)
        X_train_l = lda.fit_transform(X_train_p, y_train)
        X_test_l = lda.transform(X_test_p)

        clf = _make_classifier(cfg)
        clf.fit(X_train_l, y_train)
        y_pred = clf.predict(X_test_l)
        acc = float(accuracy_score(y_test, y_pred))

        rows.append(
            {
                "dataset": dataset_name,
                "reducer": cfg.reducer,
                "classifier": cfg.classifier,
                "dim": int(lda_dim),
                "accuracy": acc,
                "error_rate": 1.0 - acc,
                "pca_dim": int(pca_dim),
                "pca_variance": float(pca_variance),
                "n_train": int(X_train.shape[0]),
                "n_test": int(X_test.shape[0]),
                "random_state": int(cfg.random_state),
                "test_size": float(cfg.test_size),
            }
        )

    return pd.DataFrame(rows).sort_values("dim")


def save_results(df: pd.DataFrame, out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
