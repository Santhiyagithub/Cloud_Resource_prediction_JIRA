"""
tests/test_model.py
-------------------
Unit tests for the trained Cloud Resource Prediction model.
Feature schema: Alibaba Cluster Trace 2017 (machine_usage)
Run with:  pytest tests/ -v
"""

import os
import pytest
import joblib
import pandas as pd
import numpy as np

MODEL_PATH = os.path.join("models", "model.joblib")
FEATURES   = ["cpu_util", "mem_util", "net_in", "net_out",
              "cpu_lag1", "cpu_lag2", "mem_lag1"]

@pytest.fixture(scope="module")
def model():
    assert os.path.exists(MODEL_PATH), (
        f"Model not found at '{MODEL_PATH}'. Run 'python train.py' first."
    )
    return joblib.load(MODEL_PATH)

def make_input(cpu=50, mem=60, ni=30, no=25, l1=50, l2=50, ml1=60):
    return pd.DataFrame([[cpu, mem, ni, no, l1, l2, ml1]], columns=FEATURES)

# TC-01: Normal input
def test_model_predicts_normal_input(model):
    """Normal mid-range utilization values return a valid CPU %."""
    pred = model.predict(make_input())[0]
    assert 0 <= pred <= 100, f"Out of range: {pred}"

# TC-02: Low utilization
def test_model_predicts_low_input(model):
    """Low utilization values produce a valid prediction."""
    pred = model.predict(make_input(cpu=10, mem=15, ni=5, no=3, l1=12, l2=11, ml1=16))[0]
    assert isinstance(pred, (float, np.floating))
    assert 0 <= pred <= 100

# TC-03: High utilization (spike)
def test_model_predicts_high_input(model):
    """High/spike inputs return a valid prediction within 0-100."""
    pred = model.predict(make_input(cpu=95, mem=90, ni=85, no=80, l1=93, l2=91, ml1=89))[0]
    assert 0 <= pred <= 100, f"High-load prediction out of range: {pred}"

# TC-04: Boundary values
def test_model_predicts_boundary_values(model):
    """Extreme boundary inputs (0 and 100) don't crash the model."""
    for val in [0, 100]:
        pred = model.predict(make_input(cpu=val, mem=val, ni=val, no=val,
                                        l1=val, l2=val, ml1=val))[0]
        assert isinstance(pred, (float, np.floating))

# TC-05: Output is not NaN
def test_prediction_not_nan(model):
    """Prediction must be a real number, not NaN or None."""
    pred = model.predict(make_input())[0]
    assert not np.isnan(pred), "Prediction is NaN"
    assert pred is not None

# TC-06: Batch prediction
def test_model_predicts_batch(model):
    """Model handles a batch of multiple rows correctly."""
    rows = [
        [20, 30, 15, 10, 22, 24, 31],
        [60, 65, 45, 38, 58, 56, 63],
        [88, 80, 72, 65, 85, 82, 79],
    ]
    X = pd.DataFrame(rows, columns=FEATURES)
    preds = model.predict(X)
    assert len(preds) == 3
    for p in preds:
        assert 0 <= p <= 100

# TC-07: Correct feature schema
def test_model_feature_schema(model):
    """Model accepts the correct 7-column Alibaba-schema feature set."""
    X = make_input(cpu=55, mem=60, ni=35, no=28, l1=54, l2=53, ml1=59)
    try:
        model.predict(X)
    except Exception as e:
        pytest.fail(f"Unexpected exception with correct schema: {e}")
