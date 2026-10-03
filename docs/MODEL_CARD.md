# Model card — BRFSS diabetes risk model (v1.0.0)

## Overview

| | |
|---|---|
| **Model** | CatBoost gradient-boosted trees (900 iterations, depth 7, learning rate 0.04, L2 regularization 5) |
| **Task** | Binary risk estimate: diabetes or prediabetes vs. neither |
| **Input** | The 21 self-reported BRFSS 2015 health indicators (see the `/docs` API schema) |
| **Output** | Probability, decision at threshold 0.261, and optional SHAP feature contributions |
| **Training config** | `configs/serve.yaml`, identical to the CatBoost model in `configs/brfss_ensemble.yaml` |
| **Intended use** | Portfolio and educational demonstration of an explainable, reproducible ML service |
| **Not intended for** | Diagnosis, treatment decisions, or any clinical or insurance use |

## Data

- **Source:** CDC Behavioral Risk Factor Surveillance System (BRFSS) 2015. This is the cleaned "Diabetes Health
  Indicators" release: 253,680 survey responses, 21 features, label `Diabetes_binary`.
- **Integrity check:** the file is verified by SHA-256 (`19f367e3…b1d0`).
- **Prevalence:** 13.9% positive. The label groups prediabetes together with diabetes.
- **Repeated rows:** 24,206 rows are exact copies of other rows. There is no respondent ID, so they are kept,
  because they are not proven to be the same person. Identical profiles can therefore appear in both the
  training and test splits.

## How it was selected

1. The data is split once, before any learning: stratified 70/15/15, seed 42.
2. Five tree models, a logistic regression and two ensembles are compared on **validation** data.
3. The served model is the tree model with the highest validation ROC-AUC (CatBoost, 0.8286). A tree model is
   required so that exact SHAP explanations are available.
4. The decision threshold maximizes F1 on validation data (0.261).
5. The test set is scored once, after all of the above is frozen.

The validation-weighted soft-voting ensemble reaches a test ROC-AUC of 0.8307, against 0.8303 for CatBoost. The
95% confidence intervals overlap almost completely, so the simpler single model is served.

## Test performance (n = 38,052, never used for any decision)

| Metric | Value |
|---|---|
| ROC-AUC | **0.830** (95% bootstrap CI 0.825–0.836) |
| PR-AUC (average precision) | 0.426 (prevalence 0.139) |
| Recall (sensitivity) at threshold 0.261 | 0.549 |
| Specificity | 0.869 |
| Precision | 0.405 |
| F1 | 0.466 |
| MCC | 0.370 |
| Brier score | 0.097 |

### Subgroup performance (same test set and threshold)

| Group | n | Prevalence | ROC-AUC | Recall | Specificity |
|---|---|---|---|---|---|
| Female | 21,321 | 0.130 | 0.843 | 0.551 | 0.886 |
| Male | 16,731 | 0.151 | 0.814 | 0.547 | 0.848 |
| Age 18–39 | 5,906 | 0.032 | 0.854 | **0.175** | 0.991 |
| Age 40–59 | 13,813 | 0.112 | 0.841 | 0.477 | 0.913 |
| Age 60+ | 18,333 | 0.195 | 0.779 | 0.600 | 0.786 |
| Income < $25k | 8,789 | 0.224 | 0.788 | 0.688 | 0.737 |
| Income ≥ $25k | 29,263 | 0.114 | 0.833 | 0.467 | 0.904 |

Generated with `scripts/subgroup_analysis.py`. A single global threshold misses most positive cases among
younger adults (recall 0.175). Any real screening use would need threshold choices per group, or
recalibration, and a clinical review.

## Explainability

The top global drivers by mean |SHAP| on the test set are general health, BMI, age, high blood pressure, high
cholesterol and income (see `results/brfss_ensemble/figures/shap_summary.png`). SHAP values describe how the
model behaves, not causal medical effects. The negative contribution of heavy alcohol consumption, for
example, reflects an association in this survey and is not a recommendation.

## Limitations

- **Data quality:** the data is self-reported and cross-sectional, from 2015, and from a US population only.
  The model may not transfer to other years, countries or clinical data.
- **Label definition:** the label merges prediabetes with diabetes.
- **Probabilities:** they are reasonably calibrated at natural prevalence (Brier 0.097). They would be wrong
  for populations with different prevalence unless the model is recalibrated.
- **Fairness:** subgroup performance differs (see above). No fairness intervention was applied.
- **Stability:** model-to-model differences among the gradient-boosting models are smaller than the
  confidence-interval width, so none of them is meaningfully "better" than the others.
