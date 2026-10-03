"""Dataset registry, checksum-verified download and schema validation.

Datasets are never committed to Git. They are downloaded on demand into
``data/`` (or any directory you pass) and verified with SHA-256, so every run
uses byte-identical inputs. The BRFSS checksums equal the files used in the
original Colab research notebooks.
"""

from __future__ import annotations

import hashlib
import logging
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_BRFSS_MIRROR = "https://raw.githubusercontent.com/kli252/cdc_diabetes_indicator_dataset/main"

BRFSS_COLUMNS = [
    "Diabetes_binary",
    "HighBP",
    "HighChol",
    "CholCheck",
    "BMI",
    "Smoker",
    "Stroke",
    "HeartDiseaseorAttack",
    "PhysActivity",
    "Fruits",
    "Veggies",
    "HvyAlcoholConsump",
    "AnyHealthcare",
    "NoDocbcCost",
    "GenHlth",
    "MentHlth",
    "PhysHlth",
    "DiffWalk",
    "Sex",
    "Age",
    "Education",
    "Income",
]
BRFSS_BINARY = [
    "HighBP",
    "HighChol",
    "CholCheck",
    "Smoker",
    "Stroke",
    "HeartDiseaseorAttack",
    "PhysActivity",
    "Fruits",
    "Veggies",
    "HvyAlcoholConsump",
    "AnyHealthcare",
    "NoDocbcCost",
    "DiffWalk",
    "Sex",
]
BRFSS_ORDINAL = ["GenHlth", "Age", "Education", "Income"]
BRFSS_CONTINUOUS = ["BMI", "MentHlth", "PhysHlth"]

PIMA_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome",
]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    filename: str
    url: str
    sha256: str
    target: str
    n_rows: int
    columns: list[str]
    has_header: bool = True
    binary: list[str] = field(default_factory=list)
    ordinal: list[str] = field(default_factory=list)
    continuous: list[str] = field(default_factory=list)

    @property
    def features(self) -> list[str]:
        return [c for c in self.columns if c != self.target]

    @property
    def categorical(self) -> list[str]:
        return self.binary + self.ordinal


DATASETS: dict[str, DatasetSpec] = {
    "brfss": DatasetSpec(
        name="brfss",
        filename="diabetes_binary_health_indicators_BRFSS2015.csv",
        url=f"{_BRFSS_MIRROR}/diabetes_binary_health_indicators_BRFSS2015.csv",
        sha256="19f367e3e3350768f0c144c5d73ee5b355f67a57eaaa86ca7bd8aec594d8b1d0",
        target="Diabetes_binary",
        n_rows=253_680,
        columns=BRFSS_COLUMNS,
        binary=BRFSS_BINARY,
        ordinal=BRFSS_ORDINAL,
        continuous=BRFSS_CONTINUOUS,
    ),
    "brfss_5050": DatasetSpec(
        name="brfss_5050",
        filename="diabetes_binary_5050split_health_indicators_BRFSS2015.csv",
        url=f"{_BRFSS_MIRROR}/diabetes_binary_5050split_health_indicators_BRFSS2015.csv",
        sha256="873128ba0264d66a8b09bb6158f21fd4bedc10459d1ed5ecad3b94167a9539b4",
        target="Diabetes_binary",
        n_rows=70_692,
        columns=BRFSS_COLUMNS,
        binary=BRFSS_BINARY,
        ordinal=BRFSS_ORDINAL,
        continuous=BRFSS_CONTINUOUS,
    ),
    "pima": DatasetSpec(
        name="pima",
        filename="pima-indians-diabetes.data.csv",
        url="https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
        sha256="6bfe5d0f379d17a0e0819b996407e3c09bf80febd4287f2ed212190dfff154af",
        target="Outcome",
        n_rows=768,
        columns=PIMA_COLUMNS,
        has_header=False,
        continuous=[c for c in PIMA_COLUMNS if c != "Outcome"],
    ),
}


def get_spec(name: str) -> DatasetSpec:
    try:
        return DATASETS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown dataset '{name}'. Available: {sorted(DATASETS)}") from exc


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(spec: DatasetSpec, data_dir: Path) -> Path:
    """Download the dataset if missing and verify its checksum."""
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / spec.filename
    if not path.exists():
        log.info("Downloading %s -> %s", spec.url, path)
        tmp = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(spec.url, tmp)  # noqa: S310 - fixed https URLs
        tmp.replace(path)
    digest = sha256_of(path)
    if digest != spec.sha256:
        raise ValueError(
            f"Checksum mismatch for {path.name}: expected {spec.sha256}, got {digest}. "
            "Delete the file and download again, or check that you supplied the original CSV."
        )
    return path


def load(spec: DatasetSpec, data_dir: Path | str = "data", download_if_missing: bool = True) -> pd.DataFrame:
    """Load a dataset as a validated DataFrame (columns in canonical order)."""
    data_dir = Path(data_dir)
    path = data_dir / spec.filename
    if download_if_missing:
        path = download(spec, data_dir)
    elif not path.exists():
        raise FileNotFoundError(f"{path} not found; run `diabetes-xai download {spec.name}`")
    if spec.has_header:
        df = pd.read_csv(path)
    else:
        df = pd.read_csv(path, header=None, names=spec.columns)
    validate(df, spec)
    return df[spec.columns]


def validate(df: pd.DataFrame, spec: DatasetSpec) -> None:
    missing = set(spec.columns) - set(df.columns)
    if missing:
        raise ValueError(f"{spec.name}: missing columns {sorted(missing)}")
    if len(df) != spec.n_rows:
        raise ValueError(f"{spec.name}: expected {spec.n_rows} rows, found {len(df)}")
    if not set(np.unique(df[spec.target])) <= {0, 1}:
        raise ValueError(f"{spec.name}: target must be binary 0/1")
    if df[spec.columns].isna().any().any():
        raise ValueError(f"{spec.name}: unexpected empty cells")
    for col in spec.binary:
        if not df[col].isin([0, 1]).all():
            raise ValueError(f"{spec.name}: {col} must be 0/1")


def audit(df: pd.DataFrame, spec: DatasetSpec) -> dict:
    """Descriptive data-quality summary (no learning involved)."""
    y = df[spec.target]
    return {
        "dataset": spec.name,
        "rows": int(len(df)),
        "features": len(spec.features),
        "positive": int((y == 1).sum()),
        "negative": int((y == 0).sum()),
        "positive_rate": round(float(y.mean()), 4),
        "missing_cells": int(df.isna().sum().sum()),
        # Extra copies only (keep="first"); counting all copies gives a larger number.
        "duplicate_rows_extra_copies": int(df.duplicated().sum()),
        "sha256": spec.sha256,
    }
