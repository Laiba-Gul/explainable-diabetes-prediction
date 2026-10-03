"""Pydantic schemas. Field ranges follow the BRFSS 2015 codebook encoding."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Binary = Literal[0, 1]


class PatientIndicators(BaseModel):
    """The 21 BRFSS 2015 health indicators (all required, extra fields rejected)."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "HighBP": 1,
                    "HighChol": 1,
                    "CholCheck": 1,
                    "BMI": 31,
                    "Smoker": 0,
                    "Stroke": 0,
                    "HeartDiseaseorAttack": 0,
                    "PhysActivity": 1,
                    "Fruits": 1,
                    "Veggies": 1,
                    "HvyAlcoholConsump": 0,
                    "AnyHealthcare": 1,
                    "NoDocbcCost": 0,
                    "GenHlth": 3,
                    "MentHlth": 0,
                    "PhysHlth": 5,
                    "DiffWalk": 0,
                    "Sex": 0,
                    "Age": 9,
                    "Education": 5,
                    "Income": 6,
                }
            ]
        },
    )

    HighBP: Binary = Field(..., description="Told by a health professional they have high blood pressure")
    HighChol: Binary = Field(..., description="Told they have high cholesterol")
    CholCheck: Binary = Field(..., description="Cholesterol check in the past 5 years")
    BMI: float = Field(..., ge=10, le=100, description="Body mass index (kg/m^2)")
    Smoker: Binary = Field(..., description="Smoked at least 100 cigarettes in their life")
    Stroke: Binary = Field(..., description="Ever told they had a stroke")
    HeartDiseaseorAttack: Binary = Field(..., description="Coronary heart disease or myocardial infarction")
    PhysActivity: Binary = Field(..., description="Physical activity in the past 30 days (not job)")
    Fruits: Binary = Field(..., description="Consumes fruit 1+ times per day")
    Veggies: Binary = Field(..., description="Consumes vegetables 1+ times per day")
    HvyAlcoholConsump: Binary = Field(..., description="Heavy drinker (men >14, women >7 drinks/week)")
    AnyHealthcare: Binary = Field(..., description="Has any kind of health care coverage")
    NoDocbcCost: Binary = Field(..., description="Could not see a doctor in past year because of cost")
    GenHlth: int = Field(..., ge=1, le=5, description="General health: 1 excellent ... 5 poor")
    MentHlth: int = Field(..., ge=0, le=30, description="Days of poor mental health in past 30 days")
    PhysHlth: int = Field(..., ge=0, le=30, description="Days of poor physical health in past 30 days")
    DiffWalk: Binary = Field(..., description="Serious difficulty walking or climbing stairs")
    Sex: Binary = Field(..., description="0 = female, 1 = male")
    Age: int = Field(..., ge=1, le=13, description="Age band: 1 = 18-24 ... 9 = 60-64 ... 13 = 80+")
    Education: int = Field(
        ..., ge=1, le=6, description="Education level 1 (never attended) ... 6 (college graduate)"
    )
    Income: int = Field(..., ge=1, le=8, description="Income band 1 (< $10k) ... 8 (>= $75k)")


class Factor(BaseModel):
    feature: str
    value: float | None
    shap_value: float = Field(..., description="Contribution to the log-odds of the positive class")
    effect: str


class PredictionResponse(BaseModel):
    predicted_class: int = Field(..., description="1 = diabetes/prediabetes risk flagged, 0 = not flagged")
    label: str
    probability: float = Field(..., ge=0, le=1, description="Estimated probability of diabetes/prediabetes")
    threshold: float = Field(..., description="Decision threshold chosen on validation data")
    model_version: str
    top_factors: list[Factor] | None = Field(None, description="Top SHAP contributions (when explain=true)")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None = None


class ModelInfoResponse(BaseModel):
    model_version: str
    model_name: str
    dataset: str
    features: list[str]
    threshold: float
    threshold_policy: str
    test_metrics: dict
    trained_at: str | None = None
