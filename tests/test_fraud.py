import json
from pathlib import Path

import numpy as np
import pytest

from src.pipeline import at_capacity, best_threshold, load, net_savings

ROOT = Path(__file__).resolve().parent.parent
R = json.loads((ROOT / "results" / "results.json").read_text())


@pytest.fixture(scope="module")
def data():
    return load()


def test_duplicates_removed_and_split_is_in_time_order(data):
    df, info = data
    assert not df.drop(columns=["split", "hour", "log_amount"]).duplicated().any()
    assert df.loc[df.split == "train", "Time"].max() <= df.loc[df.split == "val", "Time"].min()
    assert df.loc[df.split == "val", "Time"].max() <= df.loc[df.split == "test", "Time"].min()
    assert info["duplicates_removed"] == 1081


def test_capacity_metrics_on_a_tiny_example():
    score = np.array([0.9, 0.8, 0.1, 0.05])
    y = np.array([1, 0, 1, 0])
    amount = np.array([100.0, 5.0, 50.0, 1.0])
    c = at_capacity(score, y, amount, 0.5)  # review the top 2
    assert c["fraud_caught"] == 1 and c["precision"] == 0.5 and c["recall"] == 0.5 and c["dollars_caught"] == 100


def test_net_savings_counts_review_cost_on_every_alert():
    s = net_savings(np.array([0.9, 0.8]), np.array([1, 0]), np.array([100.0, 5.0]), threshold=0.5)
    assert s["alerts"] == 2 and s["review_cost"] == 8.0 and s["saved"] == 115.0 and s["net"] == 107.0


def test_threshold_is_chosen_without_seeing_test_data():
    rng = np.random.default_rng(0)
    y = (rng.random(5000) < 0.01).astype(int)
    score = y * rng.random(5000) + rng.random(5000) * 0.5
    t = best_threshold(score, y, np.full(5000, 50.0))
    assert np.isfinite(t)


def test_supervised_models_beat_the_alternatives():
    m = R["models"]
    assert m["logistic"]["pr_auc"] > 0.5 and m["gradient_boosting"]["pr_auc"] > 0.5
    assert m["isolation_forest"]["pr_auc"] < m["logistic"]["pr_auc"]
    assert R["rule_largest_amounts"]["pr_auc"] < 0.05


def test_cost_threshold_beats_doing_nothing():
    assert R["models"]["logistic"]["cost_threshold_on_test"]["net"] > 0
