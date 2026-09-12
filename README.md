# Predictive Maintenance ML — Synthetic Anomaly Baseline

A small, deterministic **educational baseline** for machine-condition anomaly detection. It generates artificial sensor records, calibrates an interpretable z-score detector on synthetic records labelled normal, and prints metrics plus the sensor features that triggered each alert.

> **Not production software.** This repository contains no real maintenance, operational, customer, or asset data. It must not be used to make maintenance, safety, warranty, or operational decisions.

## What it does

- Generates reproducible synthetic records for `temperature_c`, `vibration_mm_s`, and `pressure_bar`.
- Uses a transparent per-feature z-score rule rather than a black-box model.
- Produces an anomaly score (largest absolute z-score), binary alert, and contributing features.
- Reports precision, recall, F1, and a labelled confusion matrix against the synthetic labels.

## Quick start

Requires Python 3.11+; the runtime has no third-party dependencies.

```bash
python -m pip install -r requirements.txt
python -m predictive_maintenance --samples 240 --anomaly-rate 0.10 --seed 42
python -m pytest -q
```

The command emits JSON. With the same options and Python version, data generation and results are deterministic.

## Data model

Each record is a JSON-like mapping:

| Field | Type | Meaning |
| --- | --- | --- |
| `machine_id` | string | Synthetic identifier, e.g. `SYN-001`; not a real asset ID. |
| `temperature_c` | number | Synthetic temperature in degrees Celsius. |
| `vibration_mm_s` | number | Synthetic vibration magnitude in mm/s. |
| `pressure_bar` | number | Synthetic pressure in bar. |
| `label` | `0` or `1` | Synthetic ground-truth marker: `0` normal, `1` injected anomaly. |

`AnomalyDetector.fit()` requires labelled rows and calibrates its means/scales from label-0 rows. `predict()` returns `machine_id`, `score`, `is_anomaly`, and `contributors`. `evaluate()` requires labels and returns binary metrics.

## Method and interpretation

For each feature, the detector calculates the absolute z-score from the synthetic-normal centre and standard deviation. A record is flagged if any feature meets the configurable threshold (default `3.0`). `contributors` identifies each feature over that threshold; `score` is the maximum absolute z-score.

This design is deliberately inspectable, but it is **not** a validated statistical process-control implementation.

## Limitations and responsible use

- The generator intentionally injects clear, correlated multi-sensor excursions. Its strong results do **not** represent field performance.
- Metrics come from a deterministic 70/30 ordered holdout split of the generated data; this is a demonstration, not validation evidence.
- Real sensors drift, have missing data, may be correlated, and require unit checks, calibration, monitoring, incident review, and domain-defined thresholds.
- The baseline needs known-normal labelled examples; it is not an unsupervised production anomaly detector.
- It has no temporal modelling, failure-risk prediction, uncertainty calibration, alert routing, access control, model registry, or governance workflow.
- Never train it on or upload confidential, personal, safety-critical, or operational data without an approved data-governance process.

## Development

```bash
python -m pytest -q
python -m predictive_maintenance --samples 80 --seed 3
```

See [SECURITY.md](SECURITY.md) for vulnerability reporting. Licensed under [MIT](LICENSE).
