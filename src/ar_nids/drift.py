from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def rbf_kernel(x: np.ndarray, y: np.ndarray, gamma: float | None = None) -> np.ndarray:
    gamma = gamma if gamma is not None else 1.0 / max(x.shape[1], 1)
    distances = np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1)
    return np.exp(-gamma * distances)


def maximum_mean_discrepancy(
    reference: np.ndarray,
    candidate: np.ndarray,
    gamma: float | None = None,
) -> float:
    k_xx = rbf_kernel(reference, reference, gamma=gamma)
    k_yy = rbf_kernel(candidate, candidate, gamma=gamma)
    k_xy = rbf_kernel(reference, candidate, gamma=gamma)
    return float(k_xx.mean() + k_yy.mean() - 2 * k_xy.mean())


@dataclass(slots=True)
class DriftDetector:
    reference: np.ndarray
    threshold: float = 0.15

    def score(self, candidate: np.ndarray) -> float:
        return maximum_mean_discrepancy(self.reference, candidate)

    def is_drifted(self, candidate: np.ndarray) -> bool:
        return self.score(candidate) >= self.threshold
