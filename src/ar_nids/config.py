from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class AdversarialConfig:
    enabled: bool = True
    epsilon: float = 0.1
    alpha: float = 0.01
    steps: int = 10
    smoothing_sigma: float = 0.05


@dataclass(slots=True)
class ServingConfig:
    confidence_threshold: float = 0.6
    mmd_threshold: float = 0.15
    explanation_budget_ms: int = 200


@dataclass(slots=True)
class MLflowConfig:
    experiment_name: str = "ar-nids"


@dataclass(slots=True)
class ARNIDSConfig:
    project_name: str = "ar-nids"
    labels: list[str] = field(
        default_factory=lambda: ["Benign", "DoS", "Probe", "R2L", "U2R", "Unknown"]
    )
    feature_count: int = 80
    packet_window_size: int = 50
    sequence_length: int = 100
    pca_components: int = 40
    rolling_window_seconds: int = 300
    batch_size: int = 128
    epochs: int = 20
    learning_rate: float = 1e-3
    dropout: float = 0.3
    recurrent_dropout: float = 0.2
    l2_weight_decay: float = 1e-4
    adversarial: AdversarialConfig = field(default_factory=AdversarialConfig)
    serving: ServingConfig = field(default_factory=ServingConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)

    @property
    def num_classes(self) -> int:
        return len(self.labels)


def _merge_dataclass(dataclass_type: type[Any], payload: dict[str, Any]) -> Any:
    return dataclass_type(**payload) if payload else dataclass_type()


def load_config(path: str | Path) -> ARNIDSConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return ARNIDSConfig(
        project_name=raw.get("project_name", "ar-nids"),
        labels=raw.get("labels", ARNIDSConfig().labels),
        feature_count=raw.get("feature_count", 80),
        packet_window_size=raw.get("packet_window_size", 50),
        sequence_length=raw.get("sequence_length", 100),
        pca_components=raw.get("pca_components", 40),
        rolling_window_seconds=raw.get("rolling_window_seconds", 300),
        batch_size=raw.get("batch_size", 128),
        epochs=raw.get("epochs", 20),
        learning_rate=raw.get("learning_rate", 1e-3),
        dropout=raw.get("dropout", 0.3),
        recurrent_dropout=raw.get("recurrent_dropout", 0.2),
        l2_weight_decay=raw.get("l2_weight_decay", 1e-4),
        adversarial=_merge_dataclass(AdversarialConfig, raw.get("adversarial", {})),
        serving=_merge_dataclass(ServingConfig, raw.get("serving", {})),
        mlflow=_merge_dataclass(MLflowConfig, raw.get("mlflow", {})),
    )
