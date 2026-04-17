import numpy as np

from ar_nids.drift import DriftDetector


def test_drift_detector_scores_shift_higher_than_baseline():
    rng = np.random.default_rng(42)
    reference = rng.normal(0, 1, size=(64, 5))
    baseline = rng.normal(0, 1, size=(64, 5))
    shifted = rng.normal(1.5, 1, size=(64, 5))
    detector = DriftDetector(reference=reference, threshold=0.1)
    assert detector.score(shifted) > detector.score(baseline)
