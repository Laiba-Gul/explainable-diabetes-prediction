"""Aggregate every experiment in ``results/`` into cross-experiment tables and figures."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from diabetes_xai import plots
from diabetes_xai.reporting import to_markdown, write_table

COLS = [
    "model",
    "roc_auc",
    "roc_auc_ci_low",
    "roc_auc_ci_high",
    "pr_auc",
    "f1",
    "mcc",
    "recall",
    "specificity",
    "precision",
    "accuracy",
    "threshold",
]


def build_report(results_dir: Path | str = "results") -> str:
    results_dir = Path(results_dir)
    out = results_dir / "summary"
    frames, verif = [], []
    for summary_file in sorted(results_dir.glob("*/summary.json")):
        exp_dir = summary_file.parent
        if exp_dir.name == "summary":
            continue
        test = exp_dir / "tables" / "test_results.csv"
        if test.exists():
            t = pd.read_csv(test)
            t.insert(0, "experiment", exp_dir.name)
            frames.append(t)
        v = exp_dir / "tables" / "verification_reported_vs_reproduced.csv"
        if v.exists():
            vt = pd.read_csv(v)
            vt.insert(0, "experiment", exp_dir.name)
            verif.append(vt)

    md = []
    if frames:
        all_test = pd.concat(frames, ignore_index=True)
        keep = ["experiment"] + [c for c in COLS if c in all_test.columns]
        all_test = all_test[keep].sort_values(["experiment", "roc_auc"], ascending=[True, False])
        write_table(all_test.round(4), out / "all_test_results")
        best = all_test.groupby("experiment", as_index=False).head(1)
        write_table(best.round(4), out / "best_model_per_experiment")
        md.append("## Best model per experiment (test set)\n\n" + to_markdown(best.round(4)))
        _cross_plot(all_test, out / "figures" / "test_auc_all_experiments")
    if verif:
        v = pd.concat(verif, ignore_index=True)
        write_table(v, out / "verification_all")
        if "abs_diff" in v:
            agg = (
                v.dropna(subset=["abs_diff"])
                .groupby("experiment")["abs_diff"]
                .agg(["count", "max", "mean"])
                .reset_index()
            )
            agg.columns = ["experiment", "numbers_checked", "max_abs_diff", "mean_abs_diff"]
            write_table(agg.round(4), out / "verification_overview")
            md.append("## Verification of reported results\n\n" + to_markdown(agg.round(4)))
    pima = results_dir / "pima_cv" / "tables" / "leakage_accuracy_inflation.csv"
    if pima.exists():
        md.append(
            "## PIMA leakage audit (mean accuracy, 5x5 CV)\n\n" + to_markdown(pd.read_csv(pima).round(4))
        )
    text = "\n".join(md)
    out.mkdir(parents=True, exist_ok=True)
    (out / "SUMMARY.md").write_text(text)
    (out / "index.json").write_text(
        json.dumps(
            {"experiments": sorted({f["experiment"].iloc[0] for f in frames})} if frames else {}, indent=2
        )
    )
    return text


def _cross_plot(df: pd.DataFrame, path: Path):
    import matplotlib.pyplot as plt

    exps = df["experiment"].unique().tolist()
    fig, axes = plt.subplots(
        1,
        len(exps),
        figsize=(4.6 * len(exps), 0.33 * df.groupby("experiment").size().max() + 1.6),
        squeeze=False,
    )
    for ax, exp in zip(axes[0], exps, strict=False):
        t = df[df["experiment"] == exp].sort_values("roc_auc")
        y = np.arange(len(t))
        ax.errorbar(
            t["roc_auc"],
            y,
            xerr=[t["roc_auc"] - t["roc_auc_ci_low"], t["roc_auc_ci_high"] - t["roc_auc"]],
            fmt="o",
            color="#2a6f97",
            ecolor="#9fb3c8",
            capsize=2,
            ms=4,
        )
        ax.set_yticks(y, t["model"])
        ax.set_title(exp)
        ax.set_xlabel("Test ROC-AUC (95% bootstrap CI)")
    fig.tight_layout()
    plots.save(fig, path)
