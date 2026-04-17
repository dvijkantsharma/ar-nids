from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from .adversarial import adversarial_training_step, randomized_smoothing_certify
from .config import ARNIDSConfig
from .data import DatasetBundle, make_synthetic_dataset
from .drift import DriftDetector
from .feature_engineering import PreparedDataset, prepare_training_data
from .model import ModelBundle, build_classifier, require_tensorflow


@dataclass(slots=True)
class TrainingArtifacts:
    model_bundle: ModelBundle
    prepared: PreparedDataset
    drift_detector: DriftDetector
    metrics: dict[str, float]


def _split_inputs(prepared: PreparedDataset) -> tuple[dict[str, np.ndarray], ...]:
    x_train_packet, x_test_packet, x_train_seq, x_test_seq, y_train, y_test = train_test_split(
        prepared.packet_windows,
        prepared.sequences,
        prepared.labels,
        test_size=0.2,
        random_state=42,
        stratify=prepared.labels,
    )
    train_inputs = {"packet_window": x_train_packet, "temporal_sequence": x_train_seq}
    test_inputs = {"packet_window": x_test_packet, "temporal_sequence": x_test_seq}
    return train_inputs, test_inputs, y_train, y_test


def train(
    config: ARNIDSConfig,
    dataset: DatasetBundle | None = None,
    output_dir: str | Path = "artifacts",
) -> TrainingArtifacts:
    tf = require_tensorflow()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dataset = dataset or make_synthetic_dataset(config)
    prepared = prepare_training_data(dataset.frame, config, label_column=dataset.label_column)
    train_inputs, test_inputs, y_train, y_test = _split_inputs(prepared)

    model_bundle = build_classifier(config)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    ]

    mlflow.set_experiment(config.mlflow.experiment_name)
    with mlflow.start_run(run_name="baseline-train"):
        mlflow.log_params(
            {
                "feature_count": config.feature_count,
                "packet_window_size": config.packet_window_size,
                "sequence_length": config.sequence_length,
                "epsilon": config.adversarial.epsilon,
                "adversarial_enabled": config.adversarial.enabled,
            }
        )

        history = model_bundle.classifier.fit(
            train_inputs,
            y_train,
            validation_split=0.2,
            epochs=config.epochs,
            batch_size=config.batch_size,
            callbacks=callbacks,
            verbose=0,
        )
        if config.adversarial.enabled:
            batch_slice = slice(0, min(len(y_train), config.batch_size))
            adversarial_loss = adversarial_training_step(
                model_bundle.classifier,
                {
                    "packet_window": train_inputs["packet_window"][batch_slice],
                    "temporal_sequence": train_inputs["temporal_sequence"][batch_slice],
                },
                y_train[batch_slice],
                config,
            )
            mlflow.log_metric("adversarial_batch_loss", adversarial_loss)

        predictions = model_bundle.classifier.predict(test_inputs, verbose=0).argmax(axis=1)
        metrics = {
            "precision_macro": float(precision_score(y_test, predictions, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_test, predictions, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
            "val_loss_min": float(min(history.history["val_loss"])),
        }
        robustness = randomized_smoothing_certify(
            model_bundle.classifier,
            {
                "packet_window": test_inputs["packet_window"][:8],
                "temporal_sequence": test_inputs["temporal_sequence"][:8],
            },
            sigma=config.adversarial.smoothing_sigma,
        )
        metrics["certified_radius"] = robustness.certified_radius
        metrics["confidence_penalty"] = robustness.confidence_penalty
        mlflow.log_metrics(metrics)
        report = classification_report(y_test, predictions, zero_division=0)
        (output_path / "classification_report.txt").write_text(report, encoding="utf-8")

    model_bundle.classifier.save(output_path / "classifier.keras")
    joblib.dump(prepared.artifacts, output_path / "feature_artifacts.joblib")
    drift_detector = DriftDetector(
        reference=prepared.sequences[: min(len(prepared.sequences), 256)].reshape(len(prepared.sequences[: min(len(prepared.sequences), 256)]), -1),
        threshold=config.serving.mmd_threshold,
    )
    joblib.dump(drift_detector, output_path / "drift_detector.joblib")
    return TrainingArtifacts(
        model_bundle=model_bundle,
        prepared=prepared,
        drift_detector=drift_detector,
        metrics=metrics,
    )
