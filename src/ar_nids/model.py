from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import ARNIDSConfig

try:
    import tensorflow as tf
except ImportError:  # pragma: no cover - exercised indirectly
    tf = None


def require_tensorflow() -> Any:
    if tf is None:
        raise ImportError(
            "TensorFlow is required for model building. Install with `pip install -e .[train]`."
        )
    return tf


@dataclass(slots=True)
class ModelBundle:
    cnn_encoder: Any
    classifier: Any


def build_cnn_encoder(config: ARNIDSConfig) -> Any:
    tf_lib = require_tensorflow()
    inputs = tf_lib.keras.Input(
        shape=(config.packet_window_size, config.feature_count), name="packet_window"
    )
    x = inputs
    for filters in (64, 128, 256):
        x = tf_lib.keras.layers.Conv1D(filters, 3, padding="same", activation="relu")(x)
    x = tf_lib.keras.layers.GlobalMaxPooling1D()(x)
    x = tf_lib.keras.layers.Dense(256, activation="relu")(x)
    x = tf_lib.keras.layers.Dropout(config.dropout)(x)
    x = tf_lib.keras.layers.Dense(128, activation="relu", name="spatial_embedding")(x)
    return tf_lib.keras.Model(inputs=inputs, outputs=x, name="cnn_encoder")


def build_classifier(config: ARNIDSConfig, cnn_encoder: Any | None = None) -> ModelBundle:
    tf_lib = require_tensorflow()
    encoder = cnn_encoder or build_cnn_encoder(config)

    packet_input = tf_lib.keras.Input(
        shape=(config.packet_window_size, config.feature_count),
        name="packet_window",
    )
    sequence_input = tf_lib.keras.Input(
        shape=(config.sequence_length, min(config.pca_components, config.feature_count)),
        name="temporal_sequence",
    )

    packet_embedding = encoder(packet_input)
    repeated_embedding = tf_lib.keras.layers.RepeatVector(config.sequence_length)(packet_embedding)
    merged = tf_lib.keras.layers.Concatenate(axis=-1)([repeated_embedding, sequence_input])
    x = tf_lib.keras.layers.Bidirectional(
        tf_lib.keras.layers.LSTM(
            128,
            return_sequences=True,
            recurrent_dropout=config.recurrent_dropout,
            kernel_regularizer=tf_lib.keras.regularizers.l2(config.l2_weight_decay),
        )
    )(merged)
    x = tf_lib.keras.layers.Bidirectional(
        tf_lib.keras.layers.LSTM(
            64,
            recurrent_dropout=config.recurrent_dropout,
            kernel_regularizer=tf_lib.keras.regularizers.l2(config.l2_weight_decay),
        )
    )(x)
    x = tf_lib.keras.layers.Dense(64, activation="relu")(x)
    x = tf_lib.keras.layers.Dense(32, activation="relu")(x)
    outputs = tf_lib.keras.layers.Dense(config.num_classes, activation="softmax")(x)

    classifier = tf_lib.keras.Model(
        inputs={"packet_window": packet_input, "temporal_sequence": sequence_input},
        outputs=outputs,
        name="cnn_bilstm_classifier",
    )
    optimizer = tf_lib.keras.optimizers.Adam(learning_rate=config.learning_rate)
    classifier.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return ModelBundle(cnn_encoder=encoder, classifier=classifier)
