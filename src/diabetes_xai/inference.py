"""Load a trained model bundle and predict/explain without retraining.

A bundle is one ``model.joblib`` file holding the fitted preprocessing steps,
the fitted classifier, the decision threshold and metadata. It is written by
``diabetes-xai train`` and read by the CLI ``predict`` command and the API.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

BUNDLE_FILE = "model.joblib"
METADATA_FILE = "metadata.json"


@dataclass
class ModelBundle:
    preprocessor: Any  # fitted transformer chain (DataFrame -> model features)
    model: Any  # fitted classifier with predict_proba
    features: list[str]  # raw input columns expected (canonical order)
    threshold: float
    metadata: dict = field(default_factory=dict)
    _explainer: Any = field(default=None, repr=False)

    # ----------------------------------------------------------------- io
    def save(self, directory: Path | str) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "preprocessor": self.preprocessor,
                "model": self.model,
                "features": self.features,
                "threshold": self.threshold,
                "metadata": self.metadata,
            },
            directory / BUNDLE_FILE,
            compress=3,
        )
        (directory / METADATA_FILE).write_text(json.dumps(self.metadata, indent=2, default=str))
        return directory / BUNDLE_FILE

    @classmethod
    def load(cls, directory: Path | str) -> ModelBundle:
        path = Path(directory)
        path = path if path.suffix == ".joblib" else path / BUNDLE_FILE
        if not path.exists():
            raise FileNotFoundError(f"Model bundle not found: {path}")
        data = joblib.load(path)
        return cls(**data)

    @property
    def version(self) -> str:
        return str(self.metadata.get("model_version", "unknown"))

    # ------------------------------------------------------------ predict
    def _frame(self, rows: pd.DataFrame | list[dict] | dict) -> pd.DataFrame:
        if isinstance(rows, dict):
            rows = [rows]
        frame = pd.DataFrame(rows)
        missing = [c for c in self.features if c not in frame.columns]
        if missing:
            raise ValueError(f"Missing input features: {missing}")
        return frame[self.features]

    def transform(self, rows) -> pd.DataFrame:
        return self.preprocessor.transform(self._frame(rows))

    def predict_proba(self, rows) -> np.ndarray:
        return np.asarray(self.model.predict_proba(self.transform(rows)))[:, 1]

    def predict(self, rows) -> list[dict]:
        proba = self.predict_proba(rows)
        labels = self.metadata.get("class_labels", {"0": "no diabetes", "1": "diabetes"})
        out = []
        for p in proba:
            cls_ = int(p >= self.threshold)
            out.append(
                {
                    "predicted_class": cls_,
                    "label": labels[str(cls_)],
                    "probability": round(float(p), 4),
                    "threshold": self.threshold,
                    "model_version": self.version,
                }
            )
        return out

    # ------------------------------------------------------------ explain
    def explain(self, rows, top_k: int = 5) -> list[list[dict]]:
        """Per-row SHAP contributions (log-odds) for tree models."""
        import shap

        if self._explainer is None:
            self._explainer = shap.TreeExplainer(self.model)
        X = self.transform(rows)  # noqa: N806
        values = np.asarray(self._explainer.shap_values(X))
        if values.ndim == 3:
            values = values[:, :, -1]
        raw = self._frame(rows)
        explanations = []
        for i in range(len(X)):
            order = np.argsort(-np.abs(values[i]))[:top_k]
            explanations.append(
                [
                    {
                        "feature": X.columns[j],
                        "value": float(raw.iloc[i][X.columns[j]]) if X.columns[j] in raw else None,
                        "shap_value": round(float(values[i, j]), 4),
                        "effect": "increases risk" if values[i, j] > 0 else "decreases risk",
                    }
                    for j in order
                ]
            )
        return explanations
