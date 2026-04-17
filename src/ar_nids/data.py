from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ARNIDSConfig


@dataclass(slots=True)
class DatasetBundle:
    frame: pd.DataFrame
    label_column: str = "label"
    flow_id_column: str = "flow_id"
    timestamp_column: str = "timestamp"


def load_csv_dataset(path: str | Path) -> DatasetBundle:
    frame = pd.read_csv(path)
    return DatasetBundle(frame=frame)


def make_synthetic_dataset(
    config: ARNIDSConfig,
    flows: int = 512,
    seed: int = 7,
) -> DatasetBundle:
    rng = np.random.default_rng(seed)
    base = rng.normal(loc=0.0, scale=1.0, size=(flows, config.feature_count))
    labels = rng.integers(low=0, high=config.num_classes, size=flows)
    attack_bias = np.linspace(0.05, 0.4, config.num_classes)
    for idx, label in enumerate(labels):
        base[idx, :10] += attack_bias[label]
        base[idx, 10:20] += (label / max(config.num_classes - 1, 1)) * 0.3

    columns = [f"f_{i:02d}" for i in range(config.feature_count)]
    frame = pd.DataFrame(base, columns=columns)
    frame["flow_id"] = np.arange(flows)
    frame["timestamp"] = pd.date_range("2026-01-01", periods=flows, freq="s")
    frame["label"] = [config.labels[value] for value in labels]
    return DatasetBundle(frame=frame)
