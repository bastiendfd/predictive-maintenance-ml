"""Command-line demonstration for the synthetic baseline."""

from __future__ import annotations

import argparse
import json
import math

from .baseline import AnomalyDetector, generate_synthetic_condition_data

TRAIN_FRACTION = 0.7


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an interpretable anomaly-detection baseline on synthetic data."
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=240,
        help="Synthetic record count (minimum: 4 for the 70%% holdout split).",
    )
    parser.add_argument("--anomaly-rate", type=float, default=0.1, help="Fraction of generated anomalies.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--z-threshold", type=float, default=3.0, help="Finite positive z-score anomaly threshold.")
    args = parser.parse_args()
    _validate_cli_args(parser, args)

    rows = generate_synthetic_condition_data(args.samples, args.anomaly_rate, args.seed)
    detector = AnomalyDetector(args.z_threshold)
    validation = detector.validate_holdout(rows, train_fraction=TRAIN_FRACTION)
    report = {
        "dataset": "synthetic",
        "samples": len(rows),
        "anomaly_rate": args.anomaly_rate,
        "seed": args.seed,
        "z_threshold": args.z_threshold,
        "validation": validation,
        "example_alerts": [prediction for prediction in detector.predict(rows) if prediction["is_anomaly"]][:3],
    }
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


def _validate_cli_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Reject combinations that cannot complete the synthetic holdout demo."""
    if args.samples < 4:
        parser.error("samples must be at least 4 for the 70% holdout split")
    if not math.isfinite(args.anomaly_rate) or not 0 < args.anomaly_rate < 1:
        parser.error("anomaly-rate must be a finite number between 0 and 1")
    if not math.isfinite(args.z_threshold) or args.z_threshold <= 0:
        parser.error("z-threshold must be a finite positive number")

    split_index = round(args.samples * TRAIN_FRACTION)
    anomaly_count = max(1, min(args.samples - 1, round(args.samples * args.anomaly_rate)))
    if anomaly_count > split_index - 2:
        parser.error(
            "samples and anomaly-rate combination leaves fewer than two normal training rows "
            "in the 70% holdout split"
        )


if __name__ == "__main__":
    main()
