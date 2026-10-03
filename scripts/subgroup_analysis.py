"""Subgroup performance of a trained bundle on its held-out test split.

Usage: python scripts/subgroup_analysis.py artifacts/brfss_serve configs/serve.yaml
Writes results/<experiment>/tables/subgroup_metrics.{csv,md}.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

from diabetes_xai.datasets import get_spec, load
from diabetes_xai.evaluation import metrics_at
from diabetes_xai.experiment import split_70_15_15
from diabetes_xai.inference import ModelBundle
from diabetes_xai.reporting import write_table

AGE_BANDS = {range(1, 5): "18-39", range(5, 9): "40-59", range(9, 14): "60+"}


def main(model_dir: str, config: str, data_dir: str = "data", results_dir: str = "results") -> None:
    cfg = yaml.safe_load(open(config))
    spec = get_spec(cfg["dataset"])
    df = load(spec, Path(data_dir))
    X, y = df[spec.features], df[spec.target].astype(int)
    _, _, X_te, _, _, y_te = split_70_15_15(X, y, int(cfg.get("seed", 42)))
    bundle = ModelBundle.load(model_dir)
    proba = bundle.predict_proba(X_te)
    groups = {
        "Sex": X_te["Sex"].map({0: "female", 1: "male"}),
        "Age band": X_te["Age"].map(lambda a: next(v for r, v in AGE_BANDS.items() if a in r)),
        "Income": X_te["Income"].map(lambda i: "<$25k" if i <= 4 else ">=$25k"),
    }
    rows = [
        {
            "group": "All",
            "level": "all",
            "n": len(y_te),
            "prevalence": y_te.mean(),
            **metrics_at(y_te, proba, bundle.threshold),
        }
    ]
    for gname, g in groups.items():
        for level in sorted(g.unique()):
            mask = (g == level).to_numpy()
            rows.append(
                {
                    "group": gname,
                    "level": level,
                    "n": int(mask.sum()),
                    "prevalence": float(y_te[mask].mean()),
                    **metrics_at(y_te[mask], proba[mask], bundle.threshold),
                }
            )
    cols = ["group", "level", "n", "prevalence", "roc_auc", "recall", "specificity", "precision", "f1"]
    table = pd.DataFrame(rows)[cols].round(4)
    write_table(table, Path(results_dir) / cfg["experiment"] / "tables" / "subgroup_metrics")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main(*sys.argv[1:])
