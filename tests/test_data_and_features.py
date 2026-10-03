from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from diabetes_xai import datasets as ds
from diabetes_xai.features import (
    ConsensusFeatureSelector,
    PartialScaler,
    PimaClinicalFeatures,
    ScaledSMOTENC,
    ZeroToNaN,
)


def test_checksum_mismatch_is_rejected(tmp_path):
    spec = ds.DATASETS["pima"]
    (tmp_path / spec.filename).write_text("1,2,3\n")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        ds.download(spec, tmp_path)


def test_load_validates_schema_and_rows(data_dir):
    df = ds.load(ds.get_spec("synthetic_brfss"), data_dir)
    assert list(df.columns) == ds.BRFSS_COLUMNS
    audit = ds.audit(df, ds.get_spec("synthetic_brfss"))
    assert audit["rows"] == len(df) and 0 < audit["positive_rate"] < 1


def test_unknown_dataset():
    with pytest.raises(ValueError):
        ds.get_spec("nope")


def test_zero_to_nan_and_clinical_features():
    X = pd.DataFrame(
        {
            "Pregnancies": [0],
            "Glucose": [0],
            "BloodPressure": [70],
            "SkinThickness": [0],
            "Insulin": [0],
            "BMI": [30.0],
            "DiabetesPedigreeFunction": [0.5],
            "Age": [40],
        }
    )
    z = ZeroToNaN().fit(X).transform(X)
    assert np.isnan(z.loc[0, "Glucose"]) and z.loc[0, "Pregnancies"] == 0  # 0 pregnancies is valid
    filled = z.fillna({"Glucose": 120.0, "SkinThickness": 20.0, "Insulin": 80.0})
    f = PimaClinicalFeatures().fit(filled).transform(filled)
    assert f.loc[0, "Glucose_BMI_Risk"] == pytest.approx(120 * 30 / 100)
    assert f.loc[0, "Glucose_Category"] == 1  # 100-125 mg/dL = prediabetes band
    assert X.loc[0, "Glucose"] == 0  # input not mutated


def test_consensus_selector_is_train_only_and_deterministic(data_dir):
    spec = ds.get_spec("synthetic_brfss")
    df = ds.load(spec, data_dir)
    X, y = df[spec.features], df[spec.target]
    kw = dict(
        k=6,
        categorical=spec.categorical,
        continuous=spec.continuous,
        scale=spec.ordinal + spec.continuous,
        random_state=0,
    )
    a = ConsensusFeatureSelector(**kw).fit(X, y)
    b = ConsensusFeatureSelector(**kw).fit(X, y)
    assert a.selected_features_ == b.selected_features_
    assert len(a.selected_features_) == 6
    assert {"GenHlth", "HighBP"} & set(a.selected_features_)  # true signal recovered
    assert list(a.transform(X).columns) == [c for c in X.columns if c in a.selected_features_]


def test_scaled_smotenc_balances_and_keeps_valid_levels(data_dir):
    spec = ds.get_spec("synthetic_brfss")
    df = ds.load(spec, data_dir)
    X, y = df[spec.features], df[spec.target]
    Xr, yr = ScaledSMOTENC(categorical=spec.categorical, continuous=spec.continuous).fit_resample(X, y)
    counts = yr.value_counts()
    assert counts[0] == counts[1]
    for c in spec.binary:
        assert set(Xr[c].unique()) <= {0, 1}
    assert Xr["GenHlth"].between(1, 5).all()


def test_partial_scaler_leaves_binary_columns():
    X = pd.DataFrame({"HighBP": [0, 1, 1], "BMI": [20.0, 30.0, 40.0]})
    out = PartialScaler(["BMI"]).fit(X).transform(X)
    assert list(out["HighBP"]) == [0, 1, 1]
    assert out["BMI"].mean() == pytest.approx(0, abs=1e-6)
