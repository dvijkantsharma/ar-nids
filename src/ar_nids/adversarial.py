from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import ARNIDSConfig
from .model import require_tensorflow


@dataclass(slots=True)
class RobustnessReport:
    confidence_penalty: float
    certified_radius: float
    attack_success_gap: float


def fgsm_attack(model: Any, inputs: dict[str, Any], labels: Any, epsilon: float) -> dict[str, Any]:
    tf = require_tensorflow()
    packet_window = tf.convert_to_tensor(inputs["packet_window"])
    temporal_sequence = tf.convert_to_tensor(inputs["temporal_sequence"])
    labels_tensor = tf.convert_to_tensor(labels)

    with tf.GradientTape() as tape:
        tape.watch(packet_window)
        predictions = model(
            {"packet_window": packet_window, "temporal_sequence": temporal_sequence}, training=False
        )
        loss = tf.keras.losses.sparse_categorical_crossentropy(labels_tensor, predictions)
    gradient = tape.gradient(loss, packet_window)
    adversarial_window = packet_window + epsilon * tf.sign(gradient)
    return {"packet_window": adversarial_window, "temporal_sequence": temporal_sequence}


def pgd_attack(
    model: Any,
    inputs: dict[str, Any],
    labels: Any,
    epsilon: float,
    alpha: float,
    steps: int,
) -> dict[str, Any]:
    tf = require_tensorflow()
    original = tf.convert_to_tensor(inputs["packet_window"])
    perturbed = tf.identity(original)
    temporal_sequence = tf.convert_to_tensor(inputs["temporal_sequence"])
    labels_tensor = tf.convert_to_tensor(labels)

    for _ in range(steps):
        with tf.GradientTape() as tape:
            tape.watch(perturbed)
            predictions = model(
                {"packet_window": perturbed, "temporal_sequence": temporal_sequence}, training=False
            )
            loss = tf.keras.losses.sparse_categorical_crossentropy(labels_tensor, predictions)
        gradient = tape.gradient(loss, perturbed)
        perturbed = perturbed + alpha * tf.sign(gradient)
        perturbation = tf.clip_by_value(perturbed - original, -epsilon, epsilon)
        perturbed = original + perturbation
    return {"packet_window": perturbed, "temporal_sequence": temporal_sequence}


def adversarial_training_step(
    model: Any,
    inputs: dict[str, Any],
    labels: Any,
    config: ARNIDSConfig,
) -> float:
    tf = require_tensorflow()
    adversarial_inputs = pgd_attack(
        model,
        inputs,
        labels,
        epsilon=config.adversarial.epsilon,
        alpha=config.adversarial.alpha,
        steps=config.adversarial.steps,
    )
    combined_packet = tf.concat([inputs["packet_window"], adversarial_inputs["packet_window"]], axis=0)
    combined_sequence = tf.concat(
        [inputs["temporal_sequence"], adversarial_inputs["temporal_sequence"]], axis=0
    )
    combined_labels = tf.concat([labels, labels], axis=0)
    metrics = model.train_on_batch(
        {"packet_window": combined_packet, "temporal_sequence": combined_sequence},
        combined_labels,
        return_dict=True,
    )
    return float(metrics["loss"])


def randomized_smoothing_certify(
    model: Any,
    inputs: dict[str, Any],
    sigma: float,
    samples: int = 16,
) -> RobustnessReport:
    tf = require_tensorflow()
    packet_window = tf.convert_to_tensor(inputs["packet_window"])
    sequence = tf.convert_to_tensor(inputs["temporal_sequence"])
    probabilities = []
    for _ in range(samples):
        noise = tf.random.normal(tf.shape(packet_window), stddev=sigma)
        prediction = model(
            {"packet_window": packet_window + noise, "temporal_sequence": sequence}, training=False
        )
        probabilities.append(prediction.numpy())
    stacked = np.asarray(probabilities)
    mean_prob = stacked.mean(axis=0)
    top1 = np.max(mean_prob, axis=1)
    top2 = np.partition(mean_prob, -2, axis=1)[:, -2]
    radius = sigma * np.maximum(top1 - top2, 0.0)
    return RobustnessReport(
        confidence_penalty=float(np.mean(1.0 - top1)),
        certified_radius=float(np.mean(radius)),
        attack_success_gap=float(np.mean(top1 - top2)),
    )
