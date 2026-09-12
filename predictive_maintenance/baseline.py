"""Synthetic, interpretable anomaly-detection baseline.

This module intentionally uses only the Python standard library.  It is a
teaching baseline, not a predictive-maintenance system.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Sequence
from statistics import mean, pstdev
from typing import Any

FEATURES = ("temperature_c", "vibration_mm_s", "pressure_bar")


class AnomalyDetector:
    """Score condition rows by their largest normalised feature deviation.

    ``fit`` calibrates means and standard deviations from rows labelled normal
    (``label == 0``). A row is flagged when one or more feature z-scores exceed
    ``z_threshold``. This is deliberately transparent and deterministic.
    """

    def __init__(self, z_threshold: float = 3.0) -> None:
        if z_threshold <= 0:
            raise ValueError("z_threshold must be positive")
        self.z_threshold = z_threshold
        self._centres: dict[str, float] | None = None
        self._scales: dict[str, float] | None = None

    def fit(self, rows: Iterable[dict[str, Any]]) -> "AnomalyDetector":
        validated = _validate_rows(rows, require_label=True)
        normal_rows = [row for row in validated if row["label"] == 0]
        if len(normal_rows) < 2:
            raise ValueError("fit requires at least two rows labelled 0 (normal)")

        self._centres = {feature: mean(row[feature] for row in normal_rows) for feature in FEATURES}
        self._scales = {
            feature: max(pstdev(row[feature] for row in normal_rows), 1e-9)
            for feature in FEATURES
        }
        return self

    def predict(self, rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        self._check_fitted()
        validated = _validate_rows(rows, require_label=False)
        predictions: list[dict[str, Any]] = []
        for row in validated:
            z_scores = {
                feature: abs((row[feature] - self._centres[feature]) / self._scales[feature])
                for feature in FEATURES
            }
            contributors = [feature for feature in FEATURES if z_scores[feature] >= self.z_threshold]
            predictions.append(
                {
                    "machine_id": row["machine_id"],
                    "score": round(max(z_scores.values()), 4),
                    "is_anomaly": bool(contributors),
                    "contributors": contributors,
                }
            )
        return predictions

    def evaluate(self, rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
        validated = _validate_rows(rows, require_label=True)
        predictions = self.predict(validated)
        counts = {"true_positive": 0, "false_positive": 0, "true_negative": 0, "false_negative": 0}
        for row, prediction in zip(validated, predictions, strict=True):
            if row["label"] == 1 and prediction["is_anomaly"]:
                counts["true_positive"] += 1
            elif row["label"] == 0 and prediction["is_anomaly"]:
                counts["false_positive"] += 1
            elif row["label"] == 0:
                counts["true_negative"] += 1
            else:
                counts["false_negative"] += 1
        precision = _safe_ratio(counts["true_positive"], counts["true_positive"] + counts["false_positive"])
        recall = _safe_ratio(counts["true_positive"], counts["true_positive"] + counts["false_negative"])
        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(_safe_ratio(2 * precision * recall, precision + recall), 4),
            "confusion_matrix": counts,
        }

    def validate_holdout(
        self, rows: Iterable[dict[str, Any]], train_fraction: float = 0.7
    ) -> dict[str, Any]:
        """Fit on the first deterministic partition and score the remaining rows.

        This is only a reproducible synthetic demonstration. Records are ordered
        exactly as supplied; callers should use a domain-appropriate split for
        any real study.
        """
        validated = _validate_rows(rows, require_label=True)
        if not 0 < train_fraction < 1:
            raise ValueError("train_fraction must be between 0 and 1")
        split_index = round(len(validated) * train_fraction)
        if split_index < 2 or split_index >= len(validated):
            raise ValueError("train_fraction must leave at least two training rows and one validation row")
        training_rows = validated[:split_index]
        validation_rows = validated[split_index:]
        self.fit(training_rows)
        return {
            "training_samples": len(training_rows),
            "validation_samples": len(validation_rows),
            "metrics": self.evaluate(validation_rows),
        }

    def _check_fitted(self) -> None:
        if self._centres is None or self._scales is None:
            raise RuntimeError("fit must be called before predict or evaluate")


def generate_synthetic_condition_data(
    samples: int = 240,
    anomaly_rate: float = 0.1,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Generate synthetic machine-condition records with labelled excursions.

    Anomalies are constructed as clear multi-sensor deviations; this makes the
    dataset useful for a deterministic example but unlike field sensor data.
    """
    if samples < 2:
        raise ValueError("samples must be at least 2")
    if not 0 < anomaly_rate < 1:
        raise ValueError("anomaly_rate must be between 0 and 1")
    rng = random.Random(seed)
    anomaly_count = max(1, min(samples - 1, round(samples * anomaly_rate)))
    anomaly_indices = set(rng.sample(range(samples), anomaly_count))
    rows: list[dict[str, Any]] = []
    for index in range(samples):
        is_anomaly = index in anomaly_indices
        row = {
            "machine_id": f"SYN-{index % 6 + 1:03d}",
            "temperature_c": round(rng.gauss(65.0, 2.0), 3),
            "vibration_mm_s": round(rng.gauss(2.2, 0.18), 3),
            "pressure_bar": round(rng.gauss(6.0, 0.16), 3),
            "label": int(is_anomaly),
        }
        if is_anomaly:
            direction = -1 if rng.random() < 0.2 else 1
            row["temperature_c"] = round(row["temperature_c"] + direction * rng.uniform(12.0, 18.0), 3)
            row["vibration_mm_s"] = round(row["vibration_mm_s"] + rng.uniform(1.8, 3.0), 3)
            row["pressure_bar"] = round(row["pressure_bar"] - rng.uniform(1.0, 1.8), 3)
        rows.append(row)
    return rows


def _validate_rows(rows: Iterable[dict[str, Any]], require_label: bool) -> list[dict[str, Any]]:
    validated = list(rows)
    if not validated:
        raise ValueError("at least one row is required")
    required = {"machine_id", *FEATURES}
    if require_label:
        required.add("label")
    for row in validated:
        missing = required - row.keys()
        if missing:
            raise ValueError(f"row is missing required fields: {', '.join(sorted(missing))}")
        if not isinstance(row["machine_id"], str) or not row["machine_id"].strip():
            raise ValueError("machine_id must be a non-empty string")
        for feature in FEATURES:
            if isinstance(row[feature], bool) or not isinstance(row[feature], (int, float)) or not math.isfinite(row[feature]):
                raise ValueError(f"{feature} must be a finite number")
        if require_label and row["label"] not in (0, 1):
            raise ValueError("label must be 0 (normal) or 1 (synthetic anomaly)")
    return validated


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0
