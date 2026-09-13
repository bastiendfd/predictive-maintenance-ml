import json
import math
import subprocess
import sys

import pytest

from predictive_maintenance.baseline import (
    AnomalyDetector,
    generate_synthetic_condition_data,
)


def test_generator_is_deterministic_for_same_seed():
    first = generate_synthetic_condition_data(samples=20, anomaly_rate=0.2, seed=7)
    second = generate_synthetic_condition_data(samples=20, anomaly_rate=0.2, seed=7)

    assert first == second
    assert len(first) == 20
    assert {row["label"] for row in first} == {0, 1}


def test_detector_flags_synthetic_anomalies_and_explains_contributors():
    rows = generate_synthetic_condition_data(samples=200, anomaly_rate=0.15, seed=11)
    detector = AnomalyDetector().fit(rows)
    predictions = detector.predict(rows)

    assert len(predictions) == len(rows)
    assert all("score" in prediction and "is_anomaly" in prediction for prediction in predictions)
    assert any(prediction["is_anomaly"] for prediction in predictions)
    assert all(
        prediction["contributors"]
        for prediction in predictions
        if prediction["is_anomaly"]
    )


def test_evaluation_reports_interpretable_binary_metrics():
    rows = generate_synthetic_condition_data(samples=300, anomaly_rate=0.2, seed=4)
    detector = AnomalyDetector().fit(rows)

    metrics = detector.evaluate(rows)

    assert metrics["confusion_matrix"].keys() == {"true_positive", "false_positive", "true_negative", "false_negative"}
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0


def test_holdout_validation_is_deterministic_and_separates_fit_rows():
    rows = generate_synthetic_condition_data(samples=200, anomaly_rate=0.15, seed=8)

    first = AnomalyDetector().validate_holdout(rows, train_fraction=0.6)
    second = AnomalyDetector().validate_holdout(rows, train_fraction=0.6)

    assert first == second
    assert first["training_samples"] == 120
    assert first["validation_samples"] == 80
    assert first["metrics"]["recall"] >= 0.8


def test_detector_rejects_invalid_training_rows():
    invalid = [{"machine_id": "M-001", "temperature_c": 50.0, "vibration_mm_s": 1.0, "pressure_bar": 5.0}]

    try:
        AnomalyDetector().fit(invalid)
    except ValueError as error:
        assert "label" in str(error)
    else:
        raise AssertionError("Expected fit() to reject a row without a label")


@pytest.mark.parametrize("threshold", [math.nan, math.inf, -math.inf, True, False])
def test_detector_rejects_non_finite_or_boolean_thresholds(threshold):
    with pytest.raises(ValueError, match="finite positive number"):
        AnomalyDetector(threshold)


def test_module_cli_emits_json_metrics():
    completed = subprocess.run(
        [sys.executable, "-m", "predictive_maintenance", "--samples", "80", "--seed", "3"],
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(completed.stdout)
    assert report["dataset"] == "synthetic"
    assert report["samples"] == 80
    assert report["validation"]["training_samples"] == 56
    assert report["validation"]["validation_samples"] == 24
    assert report["validation"]["metrics"]["recall"] >= 0.0


@pytest.mark.parametrize(
    "arguments, expected_error",
    [
        (["--z-threshold", "nan"], "z-threshold must be a finite positive number"),
        (["--z-threshold", "inf"], "z-threshold must be a finite positive number"),
        (["--samples", "2"], "samples must be at least 4"),
        (["--samples", "3"], "samples must be at least 4"),
        (["--samples", "4", "--anomaly-rate", "0.5"], "fewer than two normal training rows"),
        (["--anomaly-rate", "nan"], "anomaly-rate must be a finite number between 0 and 1"),
    ],
)
def test_module_cli_rejects_invalid_inputs_without_traceback(arguments, expected_error):
    completed = subprocess.run(
        [sys.executable, "-m", "predictive_maintenance", *arguments],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert expected_error in completed.stderr
    assert "Traceback" not in completed.stderr
    assert completed.stdout == ""
