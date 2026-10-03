"""Command-line interface: ``diabetes-xai <command>``.

Examples
--------
diabetes-xai download brfss
diabetes-xai train configs/brfss_ensemble.yaml
diabetes-xai report
diabetes-xai predict artifacts/brfss_ensemble --input patients.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd
import yaml


def _paths(args) -> tuple[Path, Path, Path]:
    data = Path(args.data_dir or os.environ.get("DIABETES_XAI_DATA", "data"))
    results = Path(args.results_dir or os.environ.get("DIABETES_XAI_RESULTS", "results"))
    artifacts = Path(args.artifacts_dir or os.environ.get("DIABETES_XAI_ARTIFACTS", "artifacts"))
    return data, results, artifacts


def load_config(path: Path | str) -> dict:
    with open(path) as handle:
        return yaml.safe_load(handle)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="diabetes-xai", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir")
    parser.add_argument("--results-dir")
    parser.add_argument("--artifacts-dir")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download", help="download + checksum-verify datasets")
    p.add_argument("datasets", nargs="*", default=["brfss", "brfss_5050", "pima"])

    p = sub.add_parser("train", help="run one or more experiment configs")
    p.add_argument("configs", nargs="+")

    sub.add_parser("report", help="aggregate results/ into comparison tables + figures")

    p = sub.add_parser("predict", help="score a CSV with a trained bundle (no retraining)")
    p.add_argument("model_dir")
    p.add_argument("--input", required=True, help="CSV with the raw feature columns")
    p.add_argument("--output", help="CSV to write (default: stdout)")
    p.add_argument("--explain", action="store_true", help="add top SHAP contributions")

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    data_dir, results_dir, artifacts_dir = _paths(args)

    if args.command == "download":
        from diabetes_xai.datasets import download, get_spec

        for name in args.datasets:
            print(download(get_spec(name), data_dir))
        return 0

    if args.command == "train":
        from diabetes_xai.cv_experiment import run_cv
        from diabetes_xai.experiment import run_holdout

        for cfg_path in args.configs:
            cfg = load_config(cfg_path)
            runner = run_cv if cfg.get("protocol") == "repeated_cv" else run_holdout
            kwargs = {} if runner is run_cv else {"artifacts_dir": artifacts_dir}
            summary = runner(cfg, data_dir, results_dir, **kwargs)
            print(json.dumps(summary, indent=2, default=str))
        return 0

    if args.command == "report":
        from diabetes_xai.report import build_report

        print(build_report(results_dir))
        return 0

    if args.command == "predict":
        from diabetes_xai.inference import ModelBundle

        bundle = ModelBundle.load(args.model_dir)
        frame = pd.read_csv(args.input)
        preds = pd.DataFrame(bundle.predict(frame))
        if args.explain:
            preds["top_factors"] = [json.dumps(e) for e in bundle.explain(frame, top_k=3)]
        if args.output:
            preds.to_csv(args.output, index=False)
        else:
            preds.to_csv(sys.stdout, index=False)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
