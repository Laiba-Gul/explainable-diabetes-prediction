from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from diabetes_xai.inference import ModelBundle


def test_holdout_run_writes_tables_figures_and_bundle(trained):
    res = trained["results"] / "test_brfss"
    for table in [
        "validation_results",
        "test_results",
        "train_results",
        "feature_selection_ranking",
        "verification_reported_vs_reproduced",
    ]:
        assert (res / "tables" / f"{table}.csv").exists(), table
        assert (res / "tables" / f"{table}.md").exists()
    for fig in ["roc_test", "pr_test", "calibration_test", "confusion_test", "auc_ci_test"]:
        assert (res / "figures" / f"{fig}.png").exists(), fig
    frozen = json.loads((res / "frozen.json").read_text())
    assert len(frozen["selected_features"]) == 8
    test = pd.read_csv(res / "tables" / "test_results.csv")
    assert {"roc_auc_ci_low", "roc_auc_ci_high"} <= set(test.columns)
    assert (test["roc_auc_ci_low"] <= test["roc_auc"]).all()


def test_selected_model_is_a_tree_model_when_serving(trained):
    meta = json.loads((trained["artifacts"] / "metadata.json").read_text())
    assert meta["model_type"] in {"lightgbm", "decision_tree"}
    assert meta["threshold_policy"] == "f1"
    assert set(meta["validation_metrics"]) >= {"roc_auc", "f1"}


def test_bundle_loads_and_predicts_without_retraining(trained, patient):
    bundle = ModelBundle.load(trained["artifacts"])
    out = bundle.predict(patient)[0]
    assert set(out) == {"predicted_class", "label", "probability", "threshold", "model_version"}
    assert out["predicted_class"] == int(out["probability"] >= out["threshold"])
    many = bundle.predict([patient, {**patient, "GenHlth": 5, "HighBP": 1, "BMI": 45, "Age": 12}])
    assert many[1]["probability"] > many[0]["probability"]


def test_bundle_explain_returns_top_factors(trained, patient):
    bundle = ModelBundle.load(trained["artifacts"])
    factors = bundle.explain(patient, top_k=3)[0]
    assert len(factors) == 3
    assert all({"feature", "shap_value", "effect"} <= set(f) for f in factors)


def test_bundle_rejects_missing_features(trained, patient):
    bundle = ModelBundle.load(trained["artifacts"])
    bad = {k: v for k, v in patient.items() if k != "BMI"}
    with pytest.raises(ValueError, match="Missing input features"):
        bundle.predict(bad)


def test_missing_bundle_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ModelBundle.load(tmp_path)


def test_report_aggregates_results(trained):
    from diabetes_xai.report import build_report

    text = build_report(trained["results"])
    assert "Best model per experiment" in text
    assert (trained["results"] / "summary" / "all_test_results.csv").exists()


def test_cli_predict(trained, patient, tmp_path):
    from diabetes_xai.cli import main

    inp = tmp_path / "in.csv"
    pd.DataFrame([patient]).to_csv(inp, index=False)
    out = tmp_path / "out.csv"
    assert (
        main(["predict", str(trained["artifacts"]), "--input", str(inp), "--output", str(out), "--explain"])
        == 0
    )
    pred = pd.read_csv(out)
    assert len(pred) == 1 and 0 <= pred.loc[0, "probability"] <= 1


def test_pima_cv_protocols_show_leakage_effect(data_dir, tmp_path):
    from diabetes_xai.cv_experiment import LEAK_FULL, SAFE, run_cv

    cfg = {
        "experiment": "test_pima",
        "dataset": "synthetic_pima",
        "protocol": "repeated_cv",
        "seed": 0,
        "cv": {"n_splits": 3, "n_repeats": 1},
        "models": {"Logistic Regression": {"type": "logistic_regression", "params": {"max_iter": 500}}},
    }
    run_cv(cfg, data_dir, tmp_path)
    summary = pd.read_csv(tmp_path / "test_pima" / "tables" / "cv_summary.csv")
    assert set(summary["protocol"]) >= {SAFE, LEAK_FULL}
    assert np.isfinite(summary["accuracy_mean"]).all()
