"""Repeated cross-validation for small datasets (PIMA, 768 rows) + leakage audit.

Compares the *same models* under three protocols:

* ``leakage-safe``  - every learned step (median imputation, SMOTE-Tomek,
  scaling) is fitted inside each training fold only;
* ``leaky: class-median imputation`` - missing values filled with the median
  of each *outcome class* computed on the full dataset (uses the label);
* ``leaky (notebook protocol)`` - class-median imputation **and** SMOTE-Tomek
  applied to the full dataset before splitting, as in the original notebooks.

The difference between protocols quantifies how much of the originally
reported 91-96% accuracy came from leakage.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import RobustScaler

from diabetes_xai import plots
from diabetes_xai.datasets import audit, get_spec, load
from diabetes_xai.evaluation import metrics_at
from diabetes_xai.features import PIMA_ZERO_AS_MISSING, DataFrameImputer, PimaClinicalFeatures, ZeroToNaN
from diabetes_xai.models import build_model
from diabetes_xai.reporting import write_table

log = logging.getLogger(__name__)
SAFE = "leakage-safe"
LEAK_IMPUTE = "leaky: class-median imputation"
LEAK_FULL = "leaky (notebook protocol)"
CV_METRICS = ["accuracy", "roc_auc", "f1", "recall", "precision", "mcc", "specificity"]


def _safe_pipeline(model, seed, balance=True):
    steps = [
        ("zero_to_nan", ZeroToNaN(PIMA_ZERO_AS_MISSING)),
        ("impute", DataFrameImputer()),
        ("clinical", PimaClinicalFeatures()),
    ]
    if balance:
        steps.append(("smote_tomek", SMOTETomek(random_state=seed)))
    steps += [("scale", RobustScaler()), ("model", model)]
    return ImbPipeline(steps)


def _class_median_impute(X, y):  # noqa: N803  - reproduces the notebook (uses the label!)
    X = ZeroToNaN(PIMA_ZERO_AS_MISSING).fit_transform(X)  # noqa: N806
    for col in PIMA_ZERO_AS_MISSING:
        X[col] = X[col].fillna(X.groupby(y)[col].transform("median"))
    return PimaClinicalFeatures().fit_transform(X)


def run_cv(cfg: dict, data_dir: Path, results_dir: Path) -> dict:
    t0 = time.time()
    name, seed = cfg["experiment"], int(cfg.get("seed", 42))
    spec = get_spec(cfg["dataset"])
    out = results_dir / name
    tables, figs = out / "tables", out / "figures"
    df = load(spec, data_dir)
    X = df[spec.features].copy()  # noqa: N806
    y = df[spec.target].astype(int).reset_index(drop=True)

    cv_cfg = cfg.get("cv", {})
    rskf = RepeatedStratifiedKFold(
        n_splits=cv_cfg.get("n_splits", 5), n_repeats=cv_cfg.get("n_repeats", 5), random_state=seed
    )
    registry = cfg["models"]
    models = {n: build_model(s, seed, registry) for n, s in registry.items() if not s.get("skip")}

    # Leaky data versions (computed once on the FULL dataset, exactly the notebook mistake)
    X_leak_imp = _class_median_impute(X, y)  # noqa: N806
    X_leak_full, y_leak_full = SMOTETomek(random_state=seed).fit_resample(X_leak_imp, y)  # noqa: N806
    X_leak_full, y_leak_full = X_leak_full.reset_index(drop=True), pd.Series(y_leak_full)  # noqa: N806
    rskf_full = RepeatedStratifiedKFold(
        n_splits=cv_cfg.get("n_splits", 5), n_repeats=cv_cfg.get("n_repeats", 5), random_state=seed
    )

    fold_rows, oof = [], {n: np.zeros(len(y)) for n in models}
    protocols = cfg.get("protocols", [SAFE, LEAK_IMPUTE, LEAK_FULL])
    for protocol in protocols:
        if protocol == LEAK_FULL:
            data_x, data_y, splitter = X_leak_full, y_leak_full, rskf_full
        else:
            data_x, data_y, splitter = (X if protocol == SAFE else X_leak_imp), y, rskf
        for fold, (tr, te) in enumerate(splitter.split(data_x, data_y)):
            for mname, model in models.items():
                if protocol == SAFE:
                    est = _safe_pipeline(clone(model), seed)
                else:  # leaky data already imputed/engineered; balance + scale inside the fold
                    est = ImbPipeline(
                        ([("smote_tomek", SMOTETomek(random_state=seed))] if protocol == LEAK_IMPUTE else [])
                        + [("scale", RobustScaler()), ("model", clone(model))]
                    )
                est.fit(data_x.iloc[tr], data_y.iloc[tr])
                proba = est.predict_proba(data_x.iloc[te])[:, 1]
                m = metrics_at(data_y.iloc[te], proba, 0.5)
                fold_rows.append(
                    {"protocol": protocol, "model": mname, "fold": fold, **{k: m[k] for k in CV_METRICS}}
                )
                if protocol == SAFE and fold < cv_cfg.get("n_splits", 5):  # first repeat -> OOF preds
                    oof[mname][te] = proba
            log.info("[%s] %s fold %d done (%.0fs)", name, protocol, fold, time.time() - t0)

    folds = pd.DataFrame(fold_rows)
    write_table(folds.round(4), tables / "cv_fold_metrics")
    agg = folds.groupby(["protocol", "model"])[CV_METRICS].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg = agg.reset_index()
    protocol_order = {p: i for i, p in enumerate(protocols)}
    agg = agg.sort_values(
        ["protocol", "roc_auc_mean"], key=lambda s: s.map(protocol_order) if s.name == "protocol" else -s
    ).reset_index(drop=True)
    write_table(agg.round(4), tables / "cv_summary")

    safe = agg[agg["protocol"] == SAFE].copy()
    write_table(safe.round(4), tables / "cv_summary_leakage_safe")
    pivot = agg.pivot(index="model", columns="protocol", values="accuracy_mean")
    if SAFE in pivot and LEAK_FULL in pivot:
        pivot["inflation_pp"] = 100 * (pivot[LEAK_FULL] - pivot[SAFE])
    write_table(pivot.reset_index().round(4), tables / "leakage_accuracy_inflation")

    title = cfg.get("title", name)
    plots.protocol_comparison(agg, "accuracy", f"Accuracy by protocol ({title})", figs / "leakage_accuracy")
    plots.protocol_comparison(agg, "roc_auc", f"ROC-AUC by protocol ({title})", figs / "leakage_roc_auc")
    plots.roc_curves(y, oof, f"Out-of-fold ROC, leakage-safe ({title})", figs / "roc_oof_safe")
    plots.pr_curves(y, oof, f"Out-of-fold precision-recall, leakage-safe ({title})", figs / "pr_oof_safe")

    _verification(cfg, agg, tables)
    summary = {
        "experiment": name,
        "dataset_audit": audit(df, spec),
        "seconds": round(time.time() - t0, 1),
        "best_safe_model": safe.iloc[0]["model"] if len(safe) else None,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def _verification(cfg, agg, tables):
    rows = []
    for item in cfg.get("reported", []):
        sub = agg[(agg["model"] == item["model"])]
        for _, r in sub.iterrows():
            rows.append(
                {
                    "claim": item["claim"],
                    "model": item["model"],
                    "protocol": r["protocol"],
                    "reported_accuracy": item.get("accuracy"),
                    "reproduced_accuracy_mean": round(r["accuracy_mean"], 4),
                    "reproduced_accuracy_std": round(r["accuracy_std"], 4),
                    "reported_auc": item.get("roc_auc"),
                    "reproduced_auc_mean": round(r["roc_auc_mean"], 4),
                }
            )
    if rows:
        write_table(pd.DataFrame(rows), tables / "verification_reported_vs_reproduced")
