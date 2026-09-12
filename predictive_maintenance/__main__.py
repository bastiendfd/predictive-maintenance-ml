"""Command-line demonstration for the synthetic baseline."""

from __future__ import annotations

import argparse
import json

from .baseline import AnomalyDetector, generate_synthetic_condition_data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an interpretable anomaly-detection baseline on synthetic data."
    )
    parser.add_argument("--samples", type=int, default=240, help="Synthetic record count (minimum: 2).")
    parser.add_argument("--anomaly-rate", type=float, default=0.1, help="Fraction of generated anomalies.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--z-threshold", type=float, default=3.0, help="Z-score anomaly threshold.")
    args = parser.parse_args()

    rows = generate_synthetic_condition_data(args.samples, args.anomaly_rate, args.seed)
    detector = AnomalyDetector(args.z_threshold)
    validation = detector.validate_holdout(rows)
    report = {
        "dataset": "synthetic",
        "samples": len(rows),
        "anomaly_rate": args.anomaly_rate,
        "seed": args.seed,
        "z_threshold": args.z_threshold,
        "validation": validation,
        "example_alerts": [prediction for prediction in detector.predict(rows) if prediction["is_anomaly"]][:3],
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
