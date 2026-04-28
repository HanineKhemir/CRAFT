"""Unit tests for ICBHI metrics."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from mvst_bts.utils.metrics import compute_icbhi_metrics


def test_perfect_predictions():
    y = [0, 1, 2, 3, 0, 1, 2, 3]
    m = compute_icbhi_metrics(y, y)
    assert m["sensitivity"] == 100.0
    assert m["specificity"] == 100.0
    assert m["icbhi_score"] == 100.0


def test_all_predicted_normal():
    y_true = [0, 1, 2, 3]
    y_pred = [0, 0, 0, 0]
    m = compute_icbhi_metrics(y_true, y_pred)
    # Normal recall = 1.0, all others = 0 → sensitivity = 25%
    assert m["recall_normal"]  == 100.0
    assert m["recall_crackle"] == 0.0
    assert m["recall_wheeze"]  == 0.0
    assert m["recall_both"]    == 0.0
    assert abs(m["sensitivity"] - 25.0) < 0.1


def test_icbhi_score_formula():
    y_true = [0, 0, 1, 1, 2, 2, 3, 3]
    y_pred = [0, 1, 1, 0, 2, 3, 3, 2]
    m = compute_icbhi_metrics(y_true, y_pred)
    expected = (m["sensitivity"] + m["specificity"]) / 2
    assert abs(m["icbhi_score"] - expected) < 0.01


def test_returns_all_keys():
    m = compute_icbhi_metrics([0, 1, 2, 3], [0, 1, 2, 3])
    for key in ["sensitivity", "specificity", "icbhi_score",
                "recall_normal", "recall_crackle", "recall_wheeze", "recall_both"]:
        assert key in m, f"Missing key: {key}"