# AR-NIDS

Adversarial-Resilient Network Intrusion Detection System (AR-NIDS) is a production-oriented Python scaffold for real-time network threat detection with adversarial hardening. It implements the core architecture from the PRD: feature engineering, a CNN encoder, a bidirectional LSTM classifier, drift monitoring, explainability hooks, and a FastAPI inference service.

## Highlights

- CNN + BiLSTM detection stack aligned to the PRD architecture
- Adversarial robustness helpers for FGSM, PGD, and randomized smoothing
- Flow-level preprocessing with scaling, PCA, and inference-time feature shaping
- MMD-based drift detection for production monitoring
- FastAPI + Prometheus serving surface for online inference
- Docker and Kubernetes deployment scaffolding
- Synthetic-data path for local smoke testing before integrating real datasets

## Repository status

This repository is a strong starter implementation, not a finished enterprise deployment. The current codebase is designed to accelerate development toward:

- dataset integration for NSL-KDD, CICIDS2017, and UNSW-NB15
- full TensorFlow training with MLflow-backed experiment tracking
- low-latency inference tuning and hardening
- SOC-facing explanation, feedback, and alert workflows

## Architecture

The codebase maps directly to the PRD's main modules:

1. Data ingestion and dataset loading
2. Feature engineering and preprocessing
3. CNN encoder for spatial packet-window representations
4. BiLSTM classifier for temporal flow behavior
5. Adversarial defense and drift-monitoring utilities
6. FastAPI service for model serving and metrics exposure

## Quick start

Install the base package plus development dependencies:

```bash
python -m pip install -e .[dev]
python -m pytest
```

Install the training extras when you want TensorFlow-based training and SHAP support:

```bash
python -m pip install -e .[train]
```

Run the training entrypoint:

```bash
ar-nids-train --config configs/default.yaml
```

Run the inference service:

```bash
ar-nids-serve --host 0.0.0.0 --port 8000
```

## Project structure

```text
configs/          Runtime and training configuration
data/             Local sample data and future dataset mounts
deploy/           Container and Kubernetes deployment assets
src/ar_nids/      Application code
tests/            Unit tests and smoke checks
```

## Core modules

- `config.py`: typed configuration loading
- `data.py`: CSV and synthetic dataset loaders
- `feature_engineering.py`: normalization, PCA, and sequence shaping
- `model.py`: CNN encoder and BiLSTM classifier builders
- `adversarial.py`: FGSM, PGD, and robustness helpers
- `drift.py`: maximum mean discrepancy drift detection
- `explainability.py`: SHAP and fallback explanation logic
- `pipeline.py`: training orchestration and artifact persistence
- `serving.py`: FastAPI application and prediction flow
- `cli.py`: train and serve entrypoints

## Next steps

- connect real intrusion-detection datasets
- install `.[train]` and run end-to-end model training
- add CI, coverage enforcement, and container publishing
- wire analyst feedback and model retraining workflows

## License

This project is released under the MIT License.
