"""Hold-out experiment runner (train / validation / locked test).

Protocol (identical for every model in an experiment):

1. stratified 70/15/15 split *before* any learned step;
2. train-only feature selection (optional consensus selector);
3. train-only class balancing (optional SMOTE-NC);
4. scaler fitted on original training rows;
5. models fitted on training data, compared on validation data;
6. ensembles weighted / thresholds chosen on validation data only;
7. the model registry is frozen, then the test set is scored exactly once.
"""

from __future__ import annotations

import json
import logging
import platform
import time
from collections import OrderedDict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from diabetes_xai import __version__, plots
from diabetes_xai.datasets import audit, get_spec, load
from diabetes_xai.evaluation import bootstrap_ci, choose_threshold, metrics_at, results_frame
from diabetes_xai.features import ColumnSelector, ConsensusFeatureSelector, PartialScaler, ScaledSMOTENC
from diabetes_xai.inference import ModelBundle
from diabetes_xai.models import TREE_TYPES, OptimizedSoftVoting, build_model
from diabetes_xai.reporting import write_table

log = logging.getLogger(__name__)


def split_70_15_15(X, y, seed):  # noqa: N803
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=seed)  # noqa: N806
    X_va, X_te, y_va, y_te = train_test_split(
        X_tmp,
        y_tmp,
        test_size=0.50,
        stratify=y_tmp,  # noqa: N806
        random_state=seed,
    )
    assert set(X_tr.index).isdisjoint(X_va.index) and set(X_tr.index).isdisjoint(X_te.index)
    assert set(X_va.index).isdisjoint(X_te.index)
    return X_tr, X_va, X_te, y_tr, y_va, y_te


def run_holdout(cfg: dict, data_dir: Path, results_dir: Path, artifacts_dir: Path) -> dict:
    t0 = time.time()
    name = cfg["experiment"]
    seed = int(cfg.get("seed", 42))
    spec = get_spec(cfg["dataset"])
    out = results_dir / name
    tables, figs = out / "tables", out / "figures"

    df = load(spec, data_dir)
    data_audit = audit(df, spec)
    if cfg.get("drop_duplicates", False):
        df = df.drop_duplicates().reset_index(drop=True)
    X = df[spec.features].copy()  # noqa: N806
    y = df[spec.target].astype(int)
    X_tr, X_va, X_te, y_tr, y_va, y_te = split_70_15_15(X, y, seed)  # noqa: N806
    log.info("[%s] split train=%d val=%d test=%d", name, len(X_tr), len(X_va), len(X_te))

    # ---- feature selection (train only)
    fs_cfg = cfg.get("feature_selection", {"method": "all"})
    scale_cols = [c for c in spec.ordinal + spec.continuous] if spec.categorical else list(spec.features)
    if fs_cfg.get("method") == "consensus":
        selector = ConsensusFeatureSelector(
            k=fs_cfg.get("k", 15),
            categorical=spec.categorical,
            continuous=spec.continuous,
            scale=scale_cols,
            random_state=seed,
        ).fit(X_tr, y_tr)
        selected = [f for f in selector.selected_features_]
        write_table(selector.ranking_table_.round(6), tables / "feature_selection_ranking")
        plots.feature_ranking(
            selector.ranking_table_,
            len(selected),
            f"Train-only consensus feature ranking ({spec.name})",
            figs / "feature_selection_ranking",
        )
    else:
        selected = list(spec.features)
    select = ColumnSelector(selected)

    # ---- class balancing (train only) + scaler fitted on original training rows
    X_tr_sel = select.transform(X_tr)  # noqa: N806
    bal_cfg = cfg.get("balance", {"method": "none"})
    imbalance = float(y_tr.value_counts().max() / y_tr.value_counts().min())
    apply_balance = bal_cfg.get("method") == "smote_nc" and imbalance >= bal_cfg.get("min_ratio", 1.2)
    if apply_balance:
        X_fit, y_fit = ScaledSMOTENC(  # noqa: N806
            categorical=[c for c in selected if c in spec.categorical],
            continuous=[c for c in selected if c in spec.continuous],
            sampling_strategy=bal_cfg.get("sampling_strategy", 1.0),
            random_state=seed,
        ).fit_resample(X_tr_sel.reset_index(drop=True), y_tr.reset_index(drop=True))
    else:
        X_fit, y_fit = X_tr_sel, y_tr  # noqa: N806
    scaler = PartialScaler([c for c in selected if c in scale_cols]).fit(X_tr_sel)
    preprocessor = Pipeline([("select", select), ("scale", scaler)])
    Xm_fit = scaler.transform(X_fit)  # noqa: N806
    Xm_tr, Xm_va, Xm_te = (preprocessor.transform(d) for d in (X_tr, X_va, X_te))  # noqa: N806
    log.info(
        "[%s] features=%d balance=%s (ratio %.2f) -> fit rows %d",
        name,
        len(selected),
        apply_balance,
        imbalance,
        len(Xm_fit),
    )

    # ---- fit models
    registry = cfg["models"]
    fitted: OrderedDict[str, object] = OrderedDict()
    families, fit_seconds = {}, {}
    for model_name, mspec in registry.items():
        if mspec.get("skip"):
            continue
        start = time.time()
        model = build_model(mspec, seed, registry)
        model.fit(Xm_fit, np.asarray(y_fit))
        fitted[model_name] = model
        families[model_name] = mspec.get(
            "family",
            "Ensemble"
            if mspec["type"]
            in {
                "random_forest",
                "extra_trees",
                "hist_gb",
                "xgboost",
                "lightgbm",
                "catboost",
                "stacking",
                "voting",
            }
            else "Machine Learning",
        )
        fit_seconds[model_name] = round(time.time() - start, 1)
        log.info("[%s] fitted %-28s %6.1fs", name, model_name, fit_seconds[model_name])

    for ens_name, espec in cfg.get("weighted_voting", {}).items():
        voter = OptimizedSoftVoting(OrderedDict((n, fitted[n]) for n in espec["estimators"]))
        voter.fit_weights(Xm_va, y_va)
        fitted[ens_name] = voter
        families[ens_name] = "Ensemble (validation-weighted)"
        fit_seconds[ens_name] = 0.0
        write_table(
            pd.DataFrame({"model": list(voter.weights_), "weight": list(voter.weights_.values())}).round(4),
            tables / f"{_slug(ens_name)}_weights",
        )

    # ---- validation: thresholds + model selection (test still unused)
    policy = cfg.get("threshold_policy", "fixed")
    thresholds, val_rows, val_proba = {}, [], {}
    for model_name, model in fitted.items():
        p_va = model.predict_proba(Xm_va)[:, 1]
        val_proba[model_name] = p_va
        thresholds[model_name] = choose_threshold(y_va, p_va, policy)
        val_rows.append(
            {
                "model": model_name,
                "family": families[model_name],
                "split": "validation",
                **metrics_at(y_va, p_va, thresholds[model_name]),
                "fit_seconds": fit_seconds[model_name],
            }
        )
    val_table = results_frame(val_rows).sort_values("roc_auc", ascending=False)
    write_table(val_table.round(4), tables / "validation_results")
    sel_metric = cfg.get("selection_metric", "roc_auc")
    eligible = val_table
    if cfg.get("serve", {}).get("only_tree_models", False):
        eligible = val_table[val_table["model"].map(lambda m: registry.get(m, {}).get("type") in TREE_TYPES)]
    best = eligible.sort_values(sel_metric, ascending=(sel_metric in {"log_loss", "brier"})).iloc[0]["model"]
    frozen = {
        "selected_model": best,
        "selection_metric": sel_metric,
        "threshold_policy": policy,
        "thresholds": thresholds,
        "selected_features": selected,
    }
    (out / "frozen.json").parent.mkdir(parents=True, exist_ok=True)
    (out / "frozen.json").write_text(json.dumps(frozen, indent=2))

    # ---- FINAL TEST GATE: everything above is frozen
    n_boot = int(cfg.get("bootstrap", 500))
    test_rows, test_proba = [], {}
    for model_name, model in fitted.items():
        p_te = model.predict_proba(Xm_te)[:, 1]
        test_proba[model_name] = p_te
        m = metrics_at(y_te, p_te, thresholds[model_name])
        ci = bootstrap_ci(y_te, p_te, thresholds[model_name], n_boot=n_boot, seed=seed)
        for k, (lo, hi) in ci.items():
            m[f"{k}_ci_low"], m[f"{k}_ci_high"] = lo, hi
        test_rows.append({"model": model_name, "family": families[model_name], "split": "test", **m})
    test_table = results_frame(test_rows).sort_values("roc_auc", ascending=False)
    write_table(test_table.round(4), tables / "test_results")
    # training-set metrics (original, un-resampled rows) to expose overfitting
    train_rows = [
        {"model": n, "split": "train", **metrics_at(y_tr, m.predict_proba(Xm_tr)[:, 1], thresholds[n])}
        for n, m in fitted.items()
    ]
    write_table(results_frame(train_rows).round(4), tables / "train_results")

    # ---- figures from real predictions
    title = cfg.get("title", name)
    plots.roc_curves(y_te, test_proba, f"ROC curves - test set ({title})", figs / "roc_test", top=8)
    plots.pr_curves(y_te, test_proba, f"Precision-recall - test set ({title})", figs / "pr_test", top=8)
    top4 = list(test_table["model"].head(4))
    plots.calibration(
        y_te, {k: test_proba[k] for k in top4}, f"Calibration - test set ({title})", figs / "calibration_test"
    )
    plots.confusion_grid(
        y_te, test_proba, thresholds, f"Confusion matrices - test set ({title})", figs / "confusion_test"
    )
    plots.metric_bars(
        test_table,
        "roc_auc",
        "roc_auc_ci_low",
        "roc_auc_ci_high",
        f"Test ROC-AUC with 95% bootstrap CI ({title})",
        figs / "auc_ci_test",
    )

    # ---- explainability for the selected model (computed after freezing)
    explain_info = {}
    if registry.get(best, {}).get("type") in TREE_TYPES:
        explain_info = _explain(fitted[best], Xm_tr, Xm_te, best, seed, tables, figs, title)

    # ---- serving threshold (validation only) for the exported model
    serve_cfg = cfg.get("serve", {})
    serve_policy = serve_cfg.get("threshold_policy", policy)
    serve_threshold = choose_threshold(y_va, val_proba[best], serve_policy)
    serve_test = metrics_at(y_te, test_proba[best], serve_threshold)

    # ---- export the selected model as a self-contained bundle
    metadata = {
        "model_version": cfg.get("model_version", __version__),
        "experiment": name,
        "dataset": spec.name,
        "dataset_sha256": spec.sha256,
        "model_name": best,
        "model_type": registry.get(best, {}).get("type", "ensemble"),
        "features": list(spec.features),
        "selected_features": selected,
        "threshold": serve_threshold,
        "threshold_policy": serve_policy,
        "class_labels": {"0": "no diabetes", "1": "diabetes / prediabetes"}
        if spec.name.startswith("brfss")
        else {"0": "non-diabetic", "1": "diabetic"},
        "validation_metrics": _clean(val_table.set_index("model").loc[best].to_dict()),
        "test_metrics": _clean(test_table.set_index("model").loc[best].to_dict()),
        "test_metrics_at_serving_threshold": _clean(serve_test),
        "split_sizes": {"train": len(X_tr), "validation": len(X_va), "test": len(X_te)},
        "balancing": "SMOTE-NC (train only)" if apply_balance else "none",
        "random_state": seed,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "diabetes_xai": __version__,
        },
        **explain_info,
    }
    if cfg.get("export_model", True):
        ModelBundle(preprocessor, fitted[best], list(spec.features), serve_threshold, metadata).save(
            artifacts_dir / name
        )
    summary = {
        "experiment": name,
        "dataset_audit": data_audit,
        "selected_model": best,
        "seconds": round(time.time() - t0, 1),
    }
    (out / "summary.json").write_text(json.dumps({**summary, "metadata": metadata}, indent=2, default=str))
    _verification(cfg, test_table, val_table, tables)
    log.info("[%s] done in %.1fs; selected %s", name, time.time() - t0, best)
    return summary


def _explain(model, X_bg, X_te, model_name, seed, tables, figs, title):  # noqa: N803
    import shap

    sample = X_te.sample(n=min(2000, len(X_te)), random_state=seed)
    explainer = shap.TreeExplainer(model)
    values = np.asarray(explainer.shap_values(sample))
    if values.ndim == 3:
        values = values[:, :, -1]
    mean_abs = pd.Series(np.abs(values).mean(axis=0), index=sample.columns).sort_values(ascending=False)
    write_table(
        mean_abs.rename("mean_abs_shap").rename_axis("feature").reset_index().round(5),
        tables / "shap_importance",
    )
    plots.shap_bar(mean_abs, f"SHAP global importance - {model_name}", figs / "shap_importance")
    plots.shap_summary(values, sample, f"SHAP summary - {model_name} ({title})", figs / "shap_summary")
    return {"shap_top_features": mean_abs.head(10).round(5).to_dict()}


def _verification(cfg, test_table, val_table, tables):
    """Compare reported notebook numbers with this reproducible run."""
    reported = cfg.get("reported", {})
    rows = []
    for model_name, metrics in reported.get("test", {}).items():
        if model_name not in set(test_table["model"]):
            continue
        ours = test_table.set_index("model").loc[model_name]
        for metric, value in metrics.items():
            rows.append(
                {
                    "model": model_name,
                    "metric": metric,
                    "reported": value,
                    "reproduced": round(float(ours[metric]), 4),
                    "abs_diff": round(abs(float(ours[metric]) - value), 4),
                }
            )
    for model_name, metrics in reported.get("validation", {}).items():
        if model_name not in set(val_table["model"]):
            continue
        ours = val_table.set_index("model").loc[model_name]
        for metric, value in metrics.items():
            rows.append(
                {
                    "model": f"{model_name} (validation)",
                    "metric": metric,
                    "reported": value,
                    "reproduced": round(float(ours[metric]), 4),
                    "abs_diff": round(abs(float(ours[metric]) - value), 4),
                }
            )
    if rows:
        write_table(pd.DataFrame(rows), tables / "verification_reported_vs_reproduced")


def _clean(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, (np.floating, float)):
            out[k] = round(float(v), 4)
        elif isinstance(v, (np.integer, int)):
            out[k] = int(v)
        else:
            out[k] = v
    return out


def _slug(s: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in s).strip("_")
