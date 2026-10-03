# Research notebooks (supporting material)

These are the original Google Colab notebooks behind this project, cleaned up:

- Outputs and personal Colab metadata were removed.
- Duplicate and abandoned versions were dropped.
- Each notebook starts with a short status header.

**You don't need these notebooks to understand or run the project.** Every validated
experiment has been turned into a configuration file that the `diabetes-xai` package runs
(see the main README). The notebooks show how the work developed.

| # | Notebook | Topic | Reproduced by |
|---|---|---|---|
| 01 | [01_brfss_eda.ipynb](01_brfss_eda.ipynb) | Exploratory data analysis of BRFSS 2015 | `dataset_audit` in each `results/*/summary.json` |
| 02 | [02_brfss_preprocessing_and_resampling.ipynb](02_brfss_preprocessing_and_resampling.ipynb) | Split, scaling, and a comparison of SMOTE, Borderline-SMOTE, SMOTE-Tomek and SMOTE-ENN | — |
| 03 | [03_brfss_data_pipeline.ipynb](03_brfss_data_pipeline.ipynb) | Data preparation on a de-duplicated copy of the data | — |
| 04 | [04_brfss_classical_baselines.ipynb](04_brfss_classical_baselines.ipynb) | Logistic regression, SVM and decision-tree baselines | — |
| 05 | [05_brfss_xai_benchmark.ipynb](05_brfss_xai_benchmark.ipynb) | 12 models, including MLP, 1D-CNN, TabTransformer, FT-Transformer and TabNet, explained with SHAP, LIME and attention | `configs/brfss_xai.yaml` (classical and boosting models) |
| 06 | [06_brfss_5050_xai_benchmark.ipynb](06_brfss_5050_xai_benchmark.ipynb) | The same benchmark on the 50/50 balanced sample | `configs/brfss_5050.yaml` |
| 07 | [07_brfss_heterogeneous_ensemble.ipynb](07_brfss_heterogeneous_ensemble.ipynb) | Weighted soft voting and out-of-fold stacking | `configs/brfss_ensemble.yaml` |

The deep-learning models (MLP, 1D-CNN, TabTransformer, FT-Transformer and TabNet) live only
in notebooks 05 and 06. Run them in Colab with a GPU runtime. In the original runs on BRFSS,
none of them beat gradient boosting: their test ROC-AUC was 0.814–0.819, against 0.821 for
CatBoost.

The earlier exploratory PIMA notebooks are not included, because their accuracy figures
were not valid:

- Missing values were filled using the median of each outcome class, which uses the target
  label to build the inputs.
- Oversampling happened before the train/test split, so synthetic copies of test patients
  ended up in the training data.

The PIMA work is redone correctly in `configs/pima_cv.yaml`, which also measures how much
each of these two mistakes inflated the scores.
