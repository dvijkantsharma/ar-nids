from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Histogram, make_asgi_app
from pydantic import BaseModel, Field

from .adversarial import randomized_smoothing_certify
from .config import ARNIDSConfig, load_config
from .drift import DriftDetector
from .explainability import explain_prediction
from .feature_engineering import FeatureArtifacts, flow_feature_template, transform_inference_frame
from .model import require_tensorflow


REQUEST_LATENCY = Histogram("ar_nids_inference_latency_seconds", "Inference latency")
PREDICTION_COUNTER = Counter("ar_nids_predictions_total", "Prediction count", ["label"])


class FlowRecord(BaseModel):
    features: dict[str, float] = Field(default_factory=dict)


class PredictRequest(BaseModel):
    records: list[FlowRecord]


@dataclass(slots=True)
class RuntimeBundle:
    config: ARNIDSConfig
    model: Any
    artifacts: FeatureArtifacts
    drift_detector: DriftDetector


def load_runtime(
    config_path: str | Path = "configs/default.yaml",
    artifacts_dir: str | Path = "artifacts",
) -> RuntimeBundle:
    config = load_config(config_path)
    tf = require_tensorflow()
    artifacts_path = Path(artifacts_dir)
    model = tf.keras.models.load_model(artifacts_path / "classifier.keras")
    artifacts = joblib.load(artifacts_path / "feature_artifacts.joblib")
    drift_detector = joblib.load(artifacts_path / "drift_detector.joblib")
    return RuntimeBundle(config=config, model=model, artifacts=artifacts, drift_detector=drift_detector)


def create_app(runtime: RuntimeBundle | None = None) -> FastAPI:
    app = FastAPI(title="AR-NIDS", version="0.1.0")
    app.mount("/metrics", make_asgi_app())
    runtime = runtime

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/predict")
    def predict(request: PredictRequest) -> dict[str, Any]:
        nonlocal runtime
        if runtime is None:
            raise HTTPException(status_code=503, detail="Model runtime is not loaded.")

        default_row = flow_feature_template(runtime.config)
        rows = []
        for record in request.records:
            row = default_row.copy()
            row.update(record.features)
            rows.append(row)
        frame = pd.DataFrame(rows)

        with REQUEST_LATENCY.time():
            packet_windows, sequences = transform_inference_frame(frame, runtime.config, runtime.artifacts)
            probabilities = runtime.model.predict(
                {"packet_window": packet_windows, "temporal_sequence": sequences},
                verbose=0,
            )
            predicted_indices = probabilities.argmax(axis=1)
            predicted_labels = runtime.artifacts.label_encoder.inverse_transform(predicted_indices)
            smoothing = randomized_smoothing_certify(
                runtime.model,
                {"packet_window": packet_windows, "temporal_sequence": sequences},
                sigma=runtime.config.adversarial.smoothing_sigma,
                samples=8,
            )
            drift_score = runtime.drift_detector.score(sequences.reshape(len(sequences), -1))
            explanations = [
                explain_prediction(
                    runtime.model,
                    packet_window=packet_windows[index],
                    feature_names=runtime.artifacts.feature_names,
                )
                for index in range(len(packet_windows))
            ]

        results = []
        for index, label in enumerate(predicted_labels):
            PREDICTION_COUNTER.labels(label=label).inc()
            confidence = float(np.max(probabilities[index]))
            action = "escalate_to_soc" if confidence >= runtime.config.serving.confidence_threshold else "monitor"
            results.append(
                {
                    "label": label,
                    "confidence": confidence,
                    "action": action,
                    "certified_radius": smoothing.certified_radius,
                    "drift_score": drift_score,
                    "drift_detected": drift_score >= runtime.config.serving.mmd_threshold,
                    "explanation_method": explanations[index].method,
                    "top_features": explanations[index].top_features,
                }
            )
        return {"results": results}

    return app
