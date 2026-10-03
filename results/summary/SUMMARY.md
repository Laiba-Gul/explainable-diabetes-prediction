## Best model per experiment (test set)

| experiment | model | roc_auc | roc_auc_ci_low | roc_auc_ci_high | pr_auc | f1 | mcc | recall | specificity | precision | accuracy | threshold |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| brfss_5050 | CatBoost | 0.8278 | 0.8200 | 0.8347 | 0.8004 | 0.7573 | 0.4944 | 0.7922 | 0.7001 | 0.7254 | 0.7461 | 0.5000 |
| brfss_ensemble | Weighted Soft Voting | 0.8307 | 0.8254 | 0.8362 | 0.4295 | 0.2327 | 0.2464 | 0.1447 | 0.9841 | 0.5950 | 0.8671 | 0.5180 |
| brfss_serve | CatBoost | 0.8303 | 0.8249 | 0.8359 | 0.4261 | 0.4658 | 0.3698 | 0.5490 | 0.8691 | 0.4045 | 0.8245 | 0.2610 |
| brfss_xai | CatBoost | 0.8212 | 0.8158 | 0.8275 | 0.4128 | 0.3966 | 0.3195 | 0.3461 | 0.9354 | 0.4643 | 0.8533 | 0.5000 |

## Verification of reported results

| experiment | numbers_checked | max_abs_diff | mean_abs_diff |
|---|---|---|---|
| brfss_5050 | 42 | 0.0005 | 0.0001 |
| brfss_ensemble | 11 | 0.0009 | 0.0002 |
| brfss_xai | 42 | 0.1067 | 0.0098 |

## PIMA leakage audit (mean accuracy, 5x5 CV)

| model | leakage-safe | leaky (notebook protocol) | leaky: class-median imputation | inflation_pp |
|---|---|---|---|---|
| CatBoost | 0.7572 | 0.9152 | 0.8693 | 15.7944 |
| Extra Trees | 0.7552 | 0.9075 | 0.8578 | 15.2329 |
| HEF-IAI Stacking | 0.7617 | 0.9169 | 0.8739 | 15.5215 |
| LightGBM | 0.7497 | 0.9130 | 0.8711 | 16.3341 |
| Logistic Regression | 0.7554 | 0.8310 | 0.8167 | 7.5566 |
| Random Forest | 0.7614 | 0.9122 | 0.8690 | 15.0759 |
| Soft Voting (2:1:1) | 0.7585 | 0.9177 | 0.8708 | 15.9211 |
| XGBoost | 0.7484 | 0.9109 | 0.8726 | 16.2511 |
