from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .config import ARNIDSConfig


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column.startswith("f_")]


@dataclass(slots=True)
class OnlineNormalizer:
    feature_names: list[str]
    mean_: np.ndarray = field(init=False)
    var_: np.ndarray = field(init=False)
    count_: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.mean_ = np.zeros(len(self.feature_names), dtype=float)
        self.var_ = np.ones(len(self.feature_names), dtype=float)

    def update(self, values: np.ndarray) -> None:
        values = np.atleast_2d(values)
        for row in values:
            self.count_ += 1
            delta = row - self.mean_
            self.mean_ += delta / self.count_
            delta2 = row - self.mean_
            self.var_ += delta * delta2

    def transform(self, values: np.ndarray) -> np.ndarray:
        denom = np.sqrt(np.maximum(self.var_ / max(self.count_, 1), 1e-6))
        return (values - self.mean_) / denom


@dataclass(slots=True)
class FeatureArtifacts:
    scaler: StandardScaler
    pca: PCA
    label_encoder: LabelEncoder
    feature_names: list[str]


@dataclass(slots=True)
class PreparedDataset:
    packet_windows: np.ndarray
    sequences: np.ndarray
    labels: np.ndarray
    artifacts: FeatureArtifacts


def _pad_windows(values: np.ndarray, target_length: int) -> np.ndarray:
    if len(values) >= target_length:
        return values[:target_length]
    padded = np.zeros((target_length, values.shape[1]), dtype=values.dtype)
    padded[: len(values)] = values
    if len(values) > 0:
        padded[len(values) :] = values[-1]
    return padded


def _pad_feature_width(values: np.ndarray, target_width: int) -> np.ndarray:
    if values.shape[1] >= target_width:
        return values[:, :target_width]
    padded = np.zeros((values.shape[0], target_width), dtype=values.dtype)
    padded[:, : values.shape[1]] = values
    return padded


def build_sequences(
    embeddings: np.ndarray,
    sequence_length: int,
) -> np.ndarray:
    sequences = []
    for idx in range(len(embeddings)):
        start = max(0, idx - sequence_length + 1)
        sequence = embeddings[start : idx + 1]
        sequences.append(_pad_windows(sequence, sequence_length))
    return np.asarray(sequences, dtype=np.float32)


def prepare_training_data(
    frame: pd.DataFrame,
    config: ARNIDSConfig,
    label_column: str = "label",
) -> PreparedDataset:
    names = feature_columns(frame)
    features = frame[names].to_numpy(dtype=np.float32)
    labels = frame[label_column].astype(str).to_numpy()

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    pca_components = min(config.pca_components, scaled.shape[1], scaled.shape[0])
    pca = PCA(n_components=max(pca_components, 1))
    reduced = pca.fit_transform(scaled).astype(np.float32)
    reduced = _pad_feature_width(reduced, config.pca_components)

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels).astype(np.int64)

    packet_windows = np.stack(
        [_pad_windows(np.repeat(row[None, :], config.packet_window_size, axis=0), config.packet_window_size) for row in scaled],
        axis=0,
    ).astype(np.float32)
    sequences = build_sequences(reduced, config.sequence_length)

    artifacts = FeatureArtifacts(
        scaler=scaler,
        pca=pca,
        label_encoder=label_encoder,
        feature_names=names,
    )
    return PreparedDataset(
        packet_windows=packet_windows,
        sequences=sequences,
        labels=encoded_labels,
        artifacts=artifacts,
    )


def transform_inference_frame(
    frame: pd.DataFrame,
    config: ARNIDSConfig,
    artifacts: FeatureArtifacts,
) -> tuple[np.ndarray, np.ndarray]:
    values = frame[artifacts.feature_names].to_numpy(dtype=np.float32)
    scaled = artifacts.scaler.transform(values).astype(np.float32)
    reduced = artifacts.pca.transform(scaled).astype(np.float32)
    reduced = _pad_feature_width(reduced, config.pca_components)
    packet_windows = np.stack(
        [_pad_windows(np.repeat(row[None, :], config.packet_window_size, axis=0), config.packet_window_size) for row in scaled],
        axis=0,
    ).astype(np.float32)
    sequences = build_sequences(reduced, config.sequence_length)
    return packet_windows, sequences


def flow_feature_template(config: ARNIDSConfig) -> dict[str, float]:
    return {f"f_{index:02d}": 0.0 for index in range(config.feature_count)}


def batched(iterable: Iterable[np.ndarray], batch_size: int) -> Iterable[list[np.ndarray]]:
    batch: list[np.ndarray] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
