"""Applicability domain assessment via Tanimoto-to-train + KDE on PCA.

A compound is flagged as **in-domain** if its nearest-neighbour Tanimoto
similarity to any training compound exceeds a data-driven threshold AND
its position in the descriptor PCA space lies within the KDE support of
the training distribution.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity


class ApplicabilityDomain:
    """Combined Tanimoto + PCA-KDE applicability domain model."""

    def __init__(
        self,
        tanimoto_radius: int = 2,
        tanimoto_nbits: int = 2048,
        pca_components: int = 5,
        kde_bandwidth: float = 0.5,
    ) -> None:
        self.tanimoto_radius = tanimoto_radius
        self.tanimoto_nbits = tanimoto_nbits
        self.pca_components = pca_components
        self.kde_bandwidth = kde_bandwidth
        self._train_fps: np.ndarray | None = None
        self._pca: PCA | None = None
        self._kde: KernelDensity | None = None
        self._log_density_threshold: float = -float("inf")

    def fit(
        self,
        train_fps: np.ndarray,
        train_descriptors: np.ndarray,
    ) -> "ApplicabilityDomain":
        """Fit the AD model on training fingerprints and descriptors."""
        self._train_fps = train_fps.astype(np.uint8)

        n_components = min(self.pca_components, train_descriptors.shape[1], train_descriptors.shape[0])
        self._pca = PCA(n_components=n_components)
        train_pca = self._pca.fit_transform(train_descriptors.astype(np.float64))

        self._kde = KernelDensity(bandwidth=self.kde_bandwidth, kernel="gaussian")
        self._kde.fit(train_pca)

        log_densities = self._kde.score_samples(train_pca)
        self._log_density_threshold = float(np.percentile(log_densities, 1))
        return self

    def _tanimoto_nn(self, query_fps: np.ndarray) -> np.ndarray:
        """Max Tanimoto similarity of each query to the training set."""
        assert self._train_fps is not None
        query = query_fps.astype(np.float32)
        train = self._train_fps.astype(np.float32)
        intersection = query @ train.T
        query_bits = query.sum(axis=1, keepdims=True)
        train_bits = train.sum(axis=1, keepdims=True).T
        union = query_bits + train_bits - intersection
        tanimoto = intersection / np.maximum(union, 1e-12)
        return tanimoto.max(axis=1)

    def score(
        self,
        query_fps: np.ndarray,
        query_descriptors: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Return per-compound AD scores.

        Returns dict with:
        - ``tanimoto_nn``: max Tanimoto to training set [0, 1]
        - ``log_density``: KDE log-density in PCA space
        - ``in_domain``: bool mask (True = inside AD)
        """
        assert self._pca is not None and self._kde is not None

        tani = self._tanimoto_nn(query_fps)

        query_pca = self._pca.transform(query_descriptors.astype(np.float64))
        log_dens = self._kde.score_samples(query_pca)

        tani_threshold = 0.3
        in_domain = (tani >= tani_threshold) & (log_dens >= self._log_density_threshold)

        return {
            "tanimoto_nn": tani,
            "log_density": log_dens,
            "in_domain": in_domain,
        }
