"""Leakage-safe feature engineering, feature selection and class balancing.

Everything here is a scikit-learn / imbalanced-learn compatible object, so it
is fitted on training rows only when placed inside a pipeline (or inside each
cross-validation fold).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectorMixin, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import StandardScaler

PIMA_ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


# --------------------------------------------------------------------------- PIMA
class ZeroToNaN(BaseEstimator, TransformerMixin):
    """In PIMA a 0 in these columns means "not measured"; mark it as missing."""

    def __init__(self, columns: list[str] | None = None):
        self.columns = columns

    def fit(self, X, y=None):  # noqa: N803
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):  # noqa: N803
        X = X.copy()  # noqa: N806
        for col in self.columns or PIMA_ZERO_AS_MISSING:
            X[col] = X[col].astype(float).replace(0.0, np.nan)
        return X

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_


class PimaClinicalFeatures(BaseEstimator, TransformerMixin):
    """Clinical interaction features from the original research notebooks.

    Uses only the input row (no target, no dataset statistics), so it cannot
    leak. Expects already-imputed numeric columns.
    """

    def fit(self, X, y=None):  # noqa: N803
        self.input_features_ = list(X.columns)
        return self

    def transform(self, X):  # noqa: N803
        X = pd.DataFrame(X, columns=self.input_features_).copy()  # noqa: N806
        X["Glucose_BMI_Risk"] = X["Glucose"] * X["BMI"] / 100
        X["Insulin_Age_Index"] = X["Insulin"] * X["Age"] / 100
        X["BP_Age_Risk"] = X["BloodPressure"] * X["Age"]
        X["Glucose_Age_Interaction"] = X["Glucose"] * X["Age"]
        X["Insulin_Resistance"] = X["Glucose"] * X["Insulin"] / 405
        X["Glucose_Category"] = np.digitize(X["Glucose"], [100, 126])  # ADA bands
        X["BMI_Category"] = np.digitize(X["BMI"], [18.5, 25, 30])
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(
            self.input_features_
            + [
                "Glucose_BMI_Risk",
                "Insulin_Age_Index",
                "BP_Age_Risk",
                "Glucose_Age_Interaction",
                "Insulin_Resistance",
                "Glucose_Category",
                "BMI_Category",
            ],
            dtype=object,
        )


class DataFrameImputer(BaseEstimator, TransformerMixin):
    """Median imputation that keeps the DataFrame (medians learned on train)."""

    def fit(self, X, y=None):  # noqa: N803
        self.medians_ = X.median(numeric_only=True)
        return self

    def transform(self, X):  # noqa: N803
        return X.fillna(self.medians_)


# ------------------------------------------------------------ feature selection
class ConsensusFeatureSelector(SelectorMixin, BaseEstimator):
    """Rank-average of seven train-only feature-ranking methods.

    Faithful re-implementation of the selection used in the original BRFSS
    notebook: |Pearson r|, Random-Forest importance, Information Gain,
    Gain Ratio, Mutual Information, RFE (logistic regression) and TreeSHAP
    importance of an XGBoost model. The ``k`` best consensus features are kept.
    """

    def __init__(
        self,
        k: int = 15,
        categorical: list[str] | None = None,
        continuous: list[str] | None = None,
        scale: list[str] | None = None,
        random_state: int = 42,
        use_shap: bool = True,
    ):
        self.k = k
        self.categorical = categorical
        self.continuous = continuous
        self.scale = scale
        self.random_state = random_state
        self.use_shap = use_shap

    def fit(self, X: pd.DataFrame, y):  # noqa: N803
        features = list(X.columns)
        y = pd.Series(np.asarray(y).astype(int), index=X.index)
        categorical = set(self.categorical or [])
        continuous = set(self.continuous or [])
        scale_cols = [c for c in (self.scale or []) if c in features]

        Xs = X.copy()  # noqa: N806
        if scale_cols:
            Xs[scale_cols] = StandardScaler().fit_transform(X[scale_cols])

        scores: dict[str, pd.Series] = {}
        scores["Pearson r"] = X.apply(lambda s: s.corr(y)).fillna(0.0)
        rf = RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            min_samples_leaf=3,
            random_state=self.random_state,
            n_jobs=-1,
        ).fit(Xs, y)
        scores["RF Importance"] = pd.Series(rf.feature_importances_, index=features)

        ig, gr = {}, {}
        for f in features:
            if f in continuous:
                codes = pd.qcut(X[f], q=10, labels=False, duplicates="drop").fillna(-1).astype(int)
            else:
                codes = X[f].astype(int)
            info = mutual_info_score(codes, y)
            intrinsic = mutual_info_score(codes, codes)
            ig[f], gr[f] = info, (info / intrinsic if intrinsic > 0 else 0.0)
        scores["Information Gain"] = pd.Series(ig)
        scores["Gain Ratio"] = pd.Series(gr)

        mask = np.array([f in categorical for f in features])
        scores["Mutual Information"] = pd.Series(
            mutual_info_classif(Xs, y, discrete_features=mask, random_state=self.random_state),
            index=features,
        )
        rfe = RFE(
            LogisticRegression(
                class_weight="balanced", solver="liblinear", max_iter=2000, random_state=self.random_state
            ),
            n_features_to_select=min(self.k, len(features)),
            step=1,
        ).fit(Xs, y)
        rfe_rank = pd.Series(rfe.ranking_, index=features)

        if self.use_shap:
            scores["SHAP Importance"] = self._shap_importance(Xs, y)

        table = pd.DataFrame({name: s.reindex(features).values for name, s in scores.items()}, index=features)
        table["RFE Rank"] = rfe_rank.reindex(features).values
        ranks = pd.DataFrame(index=features)
        ranks["|Pearson| Rank"] = table["Pearson r"].abs().rank(ascending=False)
        for name in ["RF Importance", "Information Gain", "Gain Ratio", "Mutual Information"]:
            ranks[f"{name} Rank"] = table[name].rank(ascending=False)
        ranks["RFE Method Rank"] = table["RFE Rank"].rank(ascending=True)
        if self.use_shap:
            ranks["SHAP Rank"] = table["SHAP Importance"].rank(ascending=False)
        table = table.join(ranks)
        table["Consensus Rank"] = ranks.mean(axis=1)
        table = table.rename_axis("Feature").reset_index()
        table = table.sort_values(["Consensus Rank", "Feature"]).reset_index(drop=True)

        self.ranking_table_ = table
        self.selected_features_ = table["Feature"].head(self.k).tolist()
        self.feature_names_in_ = np.asarray(features, dtype=object)
        self.n_features_in_ = len(features)
        return self

    def _shap_importance(self, Xs: pd.DataFrame, y: pd.Series) -> pd.Series:
        import shap
        from xgboost import XGBClassifier

        neg, pos = np.bincount(y)
        model = XGBClassifier(
            n_estimators=350,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="logloss",
            tree_method="hist",
            scale_pos_weight=neg / pos,
            random_state=self.random_state,
            n_jobs=-1,
        ).fit(Xs, y)
        sample = Xs.sample(n=min(1500, len(Xs)), random_state=self.random_state)
        values = np.asarray(shap.TreeExplainer(model).shap_values(sample))
        if values.ndim == 3:
            values = values[:, :, -1]
        return pd.Series(np.abs(values).mean(axis=0), index=Xs.columns)

    def _get_support_mask(self):
        return np.array([f in self.selected_features_ for f in self.feature_names_in_])

    def transform(self, X):  # noqa: N803
        if isinstance(X, pd.DataFrame):
            return X[[f for f in self.feature_names_in_ if f in self.selected_features_]]
        return super().transform(X)


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Keep a fixed list of columns (used to freeze a selected feature set)."""

    def __init__(self, columns: list[str]):
        self.columns = columns

    def fit(self, X, y=None):  # noqa: N803
        return self

    def transform(self, X):  # noqa: N803
        return X[self.columns]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.columns, dtype=object)


class PartialScaler(BaseEstimator, TransformerMixin):
    """Standardise only the given columns (ordinal/continuous); keep binaries as-is."""

    def __init__(self, columns: list[str] | None = None):
        self.columns = columns

    def fit(self, X, y=None):  # noqa: N803
        self.columns_ = [c for c in (self.columns or []) if c in X.columns]
        self.scaler_ = StandardScaler().fit(X[self.columns_]) if self.columns_ else None
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):  # noqa: N803
        X = X.copy()  # noqa: N806
        if self.scaler_ is not None:
            X[self.columns_] = self.scaler_.transform(X[self.columns_])
        return X.astype("float32")

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_


# ---------------------------------------------------------------- resampling
class ScaledSMOTENC(BaseEstimator):
    """SMOTE-NC whose neighbour space standardises continuous columns first.

    Mirrors the original notebook: continuous columns are standardised with a
    scaler fitted on the training rows, SMOTE-NC generates synthetic rows,
    then values are mapped back to the raw scale and categorical levels are
    rounded. Implements ``fit_resample`` so it only ever runs on training data
    inside an imbalanced-learn pipeline.
    """

    def __init__(
        self,
        categorical: list[str] | None = None,
        continuous: list[str] | None = None,
        sampling_strategy: float | str = 1.0,
        k_neighbors: int = 5,
        random_state: int = 42,
    ):
        self.categorical = categorical
        self.continuous = continuous
        self.sampling_strategy = sampling_strategy
        self.k_neighbors = k_neighbors
        self.random_state = random_state

    def fit_resample(self, X: pd.DataFrame, y):  # noqa: N803
        from imblearn.over_sampling import SMOTE, SMOTENC

        cols = list(X.columns)
        cat = [c for c in (self.categorical or []) if c in cols]
        cont = [c for c in (self.continuous or []) if c in cols]
        space = X.copy()
        scaler = None
        if cont:
            scaler = StandardScaler().fit(X[cont])
            space[cont] = scaler.transform(X[cont])
        if cat:
            sampler = SMOTENC(
                categorical_features=[cols.index(c) for c in cat],
                sampling_strategy=self.sampling_strategy,
                k_neighbors=self.k_neighbors,
                random_state=self.random_state,
            )
        else:
            sampler = SMOTE(
                sampling_strategy=self.sampling_strategy,
                k_neighbors=self.k_neighbors,
                random_state=self.random_state,
            )
        Xr, yr = sampler.fit_resample(space, np.asarray(y).astype(int))  # noqa: N806
        Xr = pd.DataFrame(Xr, columns=cols)  # noqa: N806
        if scaler is not None:
            Xr[cont] = scaler.inverse_transform(Xr[cont])
        for c in cat:
            Xr[c] = Xr[c].round().astype(int)
        return Xr, pd.Series(yr, name=getattr(y, "name", None))
