"""Loads model.txt and turns each feature vector X_t into class probabilities and a signal.

Uses the raw LightGBM Booster rather than the sklearn wrapper: for a single row
the wrapper's input validation costs more than the trees themselves. Probability
columns follow the trainer's sorted labels: DOWN (-1), FLAT (0), UP (1).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import lightgbm as lgb
import numpy as np

from ..features.calculator import MODEL_FEATURE_INDEX, MODEL_FEATURES

Signal = Literal["LONG", "SHORT", "FLAT"]
NUM_CLASSES: Final = 3


def signal_for(p_down: float, p_up: float, threshold: float) -> Signal:
    """The one place probabilities become a signal, shared by live inference and replay."""
    if p_up > threshold:
        return "LONG"
    if p_down > threshold:
        return "SHORT"
    return "FLAT"


@dataclass(frozen=True, slots=True)
class Prediction:
    p_down: float
    p_flat: float
    p_up: float
    signal: Signal
    latency_us: float


class SignalPredictor:
    def __init__(self, model_path: Path, threshold: float = 0.70) -> None:
        if not 0.5 <= threshold < 1.0:
            raise ValueError("threshold must be in [0.5, 1.0) so LONG and SHORT are exclusive")
        if not model_path.is_file():
            raise FileNotFoundError(f"Model not found: {model_path}. Train one with src.pipeline.trainer")
        self._booster = lgb.Booster(model_file=str(model_path))
        self.threshold = threshold
        self.model_path = model_path

        names = tuple(self._booster.feature_name())
        if names != MODEL_FEATURES:
            raise ValueError(
                f"{model_path} was trained on features {names}, but the calculator produces "
                f"{MODEL_FEATURES}. Retrain the model."
            )
        num_class = self._booster.num_model_per_iteration()
        if num_class != NUM_CLASSES:
            raise ValueError(f"{model_path} has {num_class} classes, expected {NUM_CLASSES}")

        self._row = np.empty((1, len(MODEL_FEATURES)), dtype=np.float64)
        self.predict_proba(np.zeros(MODEL_FEATURE_INDEX.max() + 1))  # warm up

    def predict_proba(self, features: np.ndarray) -> Prediction:
        """``features`` is the full calculator vector aligned with FEATURE_NAMES."""
        start = time.perf_counter_ns()
        np.take(features, MODEL_FEATURE_INDEX, out=self._row[0])
        # One thread: thread-pool startup dominates single-row prediction latency.
        # (Must be passed per call; params given at load time are ignored for model files.)
        p_down, p_flat, p_up = self._booster.predict(self._row, num_threads=1)[0]
        signal = signal_for(p_down, p_up, self.threshold)
        latency_us = (time.perf_counter_ns() - start) / 1e3
        return Prediction(float(p_down), float(p_flat), float(p_up), signal, latency_us)

    def predict_batch(self, model_features: np.ndarray) -> np.ndarray:
        """Probabilities (n, 3: DOWN, FLAT, UP) for rows already in MODEL_FEATURES order."""
        return self._booster.predict(model_features)
