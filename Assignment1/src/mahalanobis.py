from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.covariance import LedoitWolf


class MinimumMahalanobisClassifier(BaseEstimator, ClassifierMixin):
    """Minimum Mahalanobis distance classifier with shrinkage covariance.

    Fits class means and a (pooled) covariance estimator on training data.
    Predicts the class with minimal Mahalanobis distance.

    Using a shrinkage estimator (Ledoit-Wolf) keeps covariance invertible
    even when dimensionality is high.
    """

    def __init__(self):
        self.classes_ = None
        self.means_ = None
        self.precision_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same length")

        self.classes_ = np.unique(y)
        means = []
        for c in self.classes_:
            means.append(X[y == c].mean(axis=0))
        self.means_ = np.vstack(means)

        # Pooled covariance across all samples (after any transforms).
        cov = LedoitWolf().fit(X)
        self.precision_ = cov.precision_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.classes_ is None or self.means_ is None or self.precision_ is None:
            raise ValueError("Model is not fitted")
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be 2D")

        # Compute squared Mahalanobis distances to each class mean.
        # dist^2(x, mu) = (x-mu)^T P (x-mu)
        # Vectorized: for each class k, compute diag((X - mu_k) P (X - mu_k)^T)
        dists = np.empty((X.shape[0], self.means_.shape[0]), dtype=np.float64)
        P = self.precision_
        for k, mu in enumerate(self.means_):
            diff = X - mu
            dists[:, k] = np.einsum("ij,jk,ik->i", diff, P, diff)

        pred_idx = np.argmin(dists, axis=1)
        return self.classes_[pred_idx]
