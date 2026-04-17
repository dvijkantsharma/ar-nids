from __future__ import annotations

import argparse

import uvicorn

from .config import load_config
from .data import load_csv_dataset
from .pipeline import train
from .serving import create_app, load_runtime


def train_entrypoint() -> None:
    parser = argparse.ArgumentParser(description="Train the AR-NIDS model.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data", default=None)
    parser.add_argument("--output-dir", default="artifacts")
    args = parser.parse_args()

    config = load_config(args.config)
    dataset = load_csv_dataset(args.data) if args.data else None
    train(config=config, dataset=dataset, output_dir=args.output_dir)


def serve_entrypoint() -> None:
    parser = argparse.ArgumentParser(description="Serve the AR-NIDS inference API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--artifacts-dir", default="artifacts")
    args = parser.parse_args()

    runtime = load_runtime(args.config, args.artifacts_dir)
    app = create_app(runtime)
    uvicorn.run(app, host=args.host, port=args.port)
