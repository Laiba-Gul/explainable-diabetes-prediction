"""Model factory and ensemble helpers.

Models are declared in YAML configs as ``{type: ..., params: {...}}`` so every
experiment is fully described by its config file.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

TREE_TYPES = {"decision_tree", "random_forest", "extra_trees", "hist_gb", "xgboost", "lightgbm", "catboost"}


def _base(model_type: str, params: dict, seed: int):
    p = dict(params or {})
    if model_type == "logistic_regression":
        return LogisticRegression(random_state=seed, **p)
    if model_type == "linear_svm_calibrated":
        c = p.pop("C", 1.0)
        return CalibratedClassifierCV(LinearSVC(C=c, random_state=seed), method="sigmoid", cv=3, **p)
    if model_type == "decision_tree":
        return DecisionTreeClassifier(random_state=seed, **p)
    if model_type == "random_forest":
        return RandomForestClassifier(random_state=seed, n_jobs=p.pop("n_jobs", -1), **p)
    if model_type == "extra_trees":
        return ExtraTreesClassifier(random_state=seed, n_jobs=p.pop("n_jobs", -1), **p)
    if model_type == "hist_gb":
        return HistGradientBoostingClassifier(random_state=seed, **p)
    if model_type == "xgboost":
        from xgboost import XGBClassifier

        p.setdefault("eval_metric", "logloss")
        p.setdefault("tree_method", "hist")
        return XGBClassifier(random_state=seed, n_jobs=p.pop("n_jobs", -1), **p)
    if model_type == "lightgbm":
        from lightgbm import LGBMClassifier

        p.setdefault("verbosity", -1)
        return LGBMClassifier(random_state=seed, n_jobs=p.pop("n_jobs", -1), **p)
    if model_type == "catboost":
        from catboost import CatBoostClassifier

        p.setdefault("verbose", False)
        p.setdefault("allow_writing_files", False)
        return CatBoostClassifier(random_seed=seed, **p)
    raise ValueError(f"Unknown model type '{model_type}'")


def build_model(spec: dict, seed: int, registry: dict | None = None):
    """Create an (unfitted) estimator from a config entry.

    ``stacking`` and ``voting`` entries reference other models by name.
    """
    model_type = spec["type"]
    params = spec.get("params", {})
    if model_type == "stacking":
        estimators = [(n, build_model(registry[n], seed)) for n in spec["estimators"]]
        final = spec.get("final_estimator", {"type": "logistic_regression", "params": {"max_iter": 2000}})
        final_est = build_model(final, seed)
        if spec.get("scale_meta", False):
            final_est = Pipeline([("scaler", StandardScaler()), ("meta", final_est)])
        return StackingClassifier(
            estimators=estimators,
            final_estimator=final_est,
            cv=StratifiedKFold(spec.get("cv", 5), shuffle=True, random_state=seed),
            stack_method="predict_proba",
            passthrough=spec.get("passthrough", False),
            n_jobs=spec.get("n_jobs", 1),
        )
    if model_type == "voting":
        estimators = [(n, build_model(registry[n], seed)) for n in spec["estimators"]]
        return VotingClassifier(estimators=estimators, voting="soft", weights=spec.get("weights"))
    return _base(model_type, params, seed)


class OptimizedSoftVoting(BaseEstimator, ClassifierMixin):
    """Weighted average of already-fitted models; weights minimise VALIDATION log loss.

    Weights are parameterised with a softmax so they are non-negative and sum
    to one. Fitting uses validation predictions only (never test data).
    """

    def __init__(self, estimators: OrderedDict[str, object]):
        self.estimators = estimators

    def fit_weights(self, X_val, y_val):  # noqa: N803
        names = list(self.estimators)
        matrix = np.column_stack([self.estimators[n].predict_proba(X_val)[:, 1] for n in names])
        y_val = np.asarray(y_val).astype(int)

        def softmax(raw):
            e = np.exp(raw - raw.max())
            return e / e.sum()

        def loss(raw):
            return log_loss(y_val, np.clip(matrix @ softmax(raw), 1e-7, 1 - 1e-7))

        result = minimize(loss, x0=np.zeros(len(names)), method="L-BFGS-B")
        self.weights_ = OrderedDict(zip(names, softmax(result.x), strict=True))
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):  # noqa: N803
        p = sum(w * self.estimators[n].predict_proba(X)[:, 1] for n, w in self.weights_.items())
        return np.column_stack([1 - p, p])

    def predict(self, X):  # noqa: N803
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
