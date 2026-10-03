"""Hermetic fixtures: small synthetic BRFSS-like and PIMA-like datasets.

The synthetic files are registered as datasets with their own checksums, so
the real download/verify/load code path is exercised without network access.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from diabetes_xai import datasets as ds


def _brfss_like(n=2400, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    d = {c: rng.integers(0, 2, n) for c in ds.BRFSS_BINARY}
    d["GenHlth"] = rng.integers(1, 6, n)
    d["Age"] = rng.integers(1, 14, n)
    d["Education"] = rng.integers(1, 7, n)
    d["Income"] = rng.integers(1, 9, n)
    d["BMI"] = rng.normal(28, 6, n).clip(12, 80).round()
    d["MentHlth"] = rng.integers(0, 31, n)
    d["PhysHlth"] = rng.integers(0, 31, n)
    df = pd.DataFrame(d)
    logit = (
        -3.2
        + 0.6 * df["GenHlth"]
        + 0.7 * df["HighBP"]
        + 0.06 * (df["BMI"] - 28)
        + 0.12 * df["Age"]
        + rng.normal(0, 0.5, n)
    )
    df["Diabetes_binary"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return df[ds.BRFSS_COLUMNS]


def _pima_like(n=300, seed=1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "Pregnancies": rng.integers(0, 12, n),
            "Glucose": rng.normal(120, 30, n).clip(50, 199).round(),
            "BloodPressure": rng.normal(70, 12, n).clip(30, 120).round(),
            "SkinThickness": rng.normal(25, 10, n).clip(0, 60).round(),
            "Insulin": rng.normal(80, 60, n).clip(0, 400).round(),
            "BMI": rng.normal(32, 6, n).clip(18, 60).round(1),
            "DiabetesPedigreeFunction": rng.uniform(0.08, 2.4, n).round(3),
            "Age": rng.integers(21, 80, n),
        }
    )
    for col in ["Insulin", "SkinThickness"]:
        df.loc[rng.random(n) < 0.3, col] = 0
    risk = 0.04 * (df["Glucose"] - 120) + 0.08 * (df["BMI"] - 32) + rng.normal(0, 1, n)
    df["Outcome"] = (risk > 0.6).astype(int)
    return df[ds.PIMA_COLUMNS]


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("data")
    brfss = _brfss_like()
    brfss.to_csv(d / "synthetic_brfss.csv", index=False)
    pima = _pima_like()
    pima.to_csv(d / "synthetic_pima.csv", index=False, header=False)
    ds.DATASETS["synthetic_brfss"] = dataclasses.replace(
        ds.DATASETS["brfss"],
        name="synthetic_brfss",
        filename="synthetic_brfss.csv",
        url="file://unused",
        sha256=ds.sha256_of(d / "synthetic_brfss.csv"),
        n_rows=len(brfss),
    )
    ds.DATASETS["synthetic_pima"] = dataclasses.replace(
        ds.DATASETS["pima"],
        name="synthetic_pima",
        filename="synthetic_pima.csv",
        url="file://unused",
        sha256=ds.sha256_of(d / "synthetic_pima.csv"),
        n_rows=len(pima),
    )
    return d


@pytest.fixture(scope="session")
def holdout_cfg() -> dict:
    return {
        "experiment": "test_brfss",
        "dataset": "synthetic_brfss",
        "seed": 42,
        "feature_selection": {"method": "consensus", "k": 8},
        "balance": {"method": "smote_nc", "sampling_strategy": 1.0, "min_ratio": 1.2},
        "threshold_policy": "fixed",
        "bootstrap": 50,
        "serve": {"only_tree_models": True, "threshold_policy": "f1"},
        "models": {
            "Logistic Regression": {"type": "logistic_regression", "params": {"max_iter": 500}},
            "LightGBM": {"type": "lightgbm", "params": {"n_estimators": 60, "num_leaves": 15}},
            "Decision Tree": {"type": "decision_tree", "params": {"max_depth": 4}},
        },
        "weighted_voting": {"Voting": {"estimators": ["LightGBM", "Logistic Regression"]}},
        "reported": {"test": {"LightGBM": {"roc_auc": 0.5}}},
    }


@pytest.fixture(scope="session")
def trained(tmp_path_factory, data_dir, holdout_cfg):
    from diabetes_xai.experiment import run_holdout

    results = tmp_path_factory.mktemp("results")
    artifacts = tmp_path_factory.mktemp("artifacts")
    summary = run_holdout(holdout_cfg, data_dir, results, artifacts)
    return {"summary": summary, "results": results, "artifacts": artifacts / "test_brfss"}


@pytest.fixture()
def patient() -> dict:
    return {
        "HighBP": 1,
        "HighChol": 1,
        "CholCheck": 1,
        "BMI": 31,
        "Smoker": 0,
        "Stroke": 0,
        "HeartDiseaseorAttack": 0,
        "PhysActivity": 1,
        "Fruits": 1,
        "Veggies": 1,
        "HvyAlcoholConsump": 0,
        "AnyHealthcare": 1,
        "NoDocbcCost": 0,
        "GenHlth": 3,
        "MentHlth": 0,
        "PhysHlth": 5,
        "DiffWalk": 0,
        "Sex": 0,
        "Age": 9,
        "Education": 5,
        "Income": 6,
    }
