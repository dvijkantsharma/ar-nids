from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import shap
except ImportError:  # pragma: no cover - optional dependency
    shap = None


@dataclass(slots=True)
class ExplanationResult:
    method: str
    top_features: list[tuple[str, float]]


def explain_prediction(
    model: Any,
    packet_window: np.ndarray,
    feature_names: list[str],
    max_features: int = 5,
) -> ExplanationResult:
    flattened = packet_window.mean(axis=0)
    if shap is None:
        ranking = np.argsort(np.abs(flattened))[::-1][:max_features]
        return ExplanationResult(
            method="heuristic_mean_activation",
            top_features=[(feature_names[index], float(flattened[index])) for index in ranking],
        )

    explainer = shap.Explainer(lambda samples: model.predict(samples, verbose=0))
    shap_values = explainer(packet_window[:1])
    scores = np.abs(np.asarray(shap_values.values)).mean(axis=tuple(range(shap_values.values.ndim - 1)))
    ranking = np.argsort(scores)[::-1][:max_features]
    return ExplanationResult(
        method="shap",
        top_features=[(feature_names[index], float(scores[index])) for index in ranking],
    )
