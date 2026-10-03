from __future__ import annotations

import numpy as np
import pytest

from diabetes_xai.evaluation import bootstrap_ci, choose_threshold, metrics_at
from diabetes_xai.models import OptimizedSoftVoting, build_model


def test_metrics_known_values():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.6, 0.4, 0.9])
    m = metrics_at(y, p, 0.5)
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (1, 1, 1, 1)
    assert m["accuracy"] == 0.5 and m["roc_auc"] == 0.75
    assert m["specificity"] == 0.5 and m["recall"] == 0.5


@pytest.mark.parametrize("policy", ["fixed", "f1", "youden", "accuracy"])
def test_threshold_policies_return_valid_thresholds(policy):
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = np.clip(y * 0.3 + rng.random(500) * 0.7, 0, 1)
    t = choose_threshold(y, p, policy)
    assert 0 < t < 1
    if policy == "fixed":
        assert t == 0.5


def test_unknown_threshold_policy():
    with pytest.raises(ValueError):
        choose_threshold([0, 1], [0.2, 0.8], "magic")


def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 800)
    p = np.clip(0.35 * y + rng.random(800) * 0.65, 0, 1)
    point = metrics_at(y, p)["roc_auc"]
    lo, hi = bootstrap_ci(y, p, n_boot=200)["roc_auc"]
    assert lo <= point <= hi and hi - lo < 0.15


@pytest.mark.parametrize(
    "model_type",
    [
        "logistic_regression",
        "decision_tree",
        "random_forest",
        "extra_trees",
        "hist_gb",
        "xgboost",
        "lightgbm",
        "catboost",
        "linear_svm_calibrated",
    ],
)
def test_every_model_type_fits_and_predicts(model_type):
    rng = np.random.default_rng(2)
    X = rng.normal(size=(200, 4))
    y = (X[:, 0] + rng.normal(0, 0.5, 200) > 0).astype(int)
    params = (
        {"n_estimators": 20} if model_type in {"random_forest", "extra_trees", "xgboost", "lightgbm"} else {}
    )
    if model_type == "catboost":
        params = {"iterations": 20}
    m = build_model({"type": model_type, "params": params}, seed=0).fit(X, y)
    p = m.predict_proba(X)[:, 1]
    assert p.shape == (200,) and ((p >= 0) & (p <= 1)).all()


def test_stacking_and_voting_reference_registry():
    registry = {
        "a": {"type": "logistic_regression"},
        "b": {"type": "decision_tree", "params": {"max_depth": 2}},
    }
    stack = build_model(
        {"type": "stacking", "estimators": ["a", "b"], "cv": 3, "passthrough": True, "scale_meta": True},
        0,
        registry,
    )
    vote = build_model({"type": "voting", "estimators": ["a", "b"], "weights": [2, 1]}, 0, registry)
    rng = np.random.default_rng(3)
    X = rng.normal(size=(150, 3))
    y = (X[:, 0] > 0).astype(int)
    for m in (stack, vote):
        assert m.fit(X, y).predict_proba(X).shape == (150, 2)


def test_optimized_soft_voting_weights_sum_to_one():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(300, 3))
    y = (X[:, 0] + rng.normal(0, 0.3, 300) > 0).astype(int)
    good = build_model({"type": "logistic_regression"}, 0).fit(X, y)
    noise = build_model({"type": "decision_tree", "params": {"max_depth": 1}}, 0).fit(X[:, 2:], y)

    class Wrap:  # model that only sees the useless column
        def predict_proba(self, Z):
            return noise.predict_proba(Z[:, 2:])

    voter = OptimizedSoftVoting({"good": good, "noise": Wrap()}).fit_weights(X, y)
    w = voter.weights_
    assert sum(w.values()) == pytest.approx(1.0) and w["good"] > w["noise"]
