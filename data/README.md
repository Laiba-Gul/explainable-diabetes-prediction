# data/

Datasets are downloaded here by `diabetes-xai download` and verified with SHA-256. They are not committed to Git.

## Sources

| Dataset | Kaggle page | File in `data/` |
|---|---|---|
| CDC Diabetes Health Indicators (BRFSS 2015) | https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset | `diabetes_binary_health_indicators_BRFSS2015.csv`, `diabetes_binary_5050split_health_indicators_BRFSS2015.csv` (same files as on Kaggle) |
| PIMA Indians Diabetes | https://www.kaggle.com/datasets/jamaltariqcheema/pima-indians-diabetes-dataset | `pima-indians-diabetes.data.csv`: the same 768 rows as Kaggle's `diabetes.csv`, without the header row |

## Downloading manually from Kaggle

Kaggle requires a login, so `diabetes-xai download` fetches the files from public mirrors and checks each one
against the SHA-256 checksum in `src/diabetes_xai/datasets.py`.

- **BRFSS:** you can download the two CSVs from Kaggle and copy them here. `diabetes-xai download` keeps files
  that are already present and match the checksum.
- **PIMA:** let `diabetes-xai download` fetch it. Kaggle's `diabetes.csv` has a header row, so its checksum
  differs, even though the values are identical.
