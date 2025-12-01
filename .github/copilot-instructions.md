**Purpose**: Brief, actionable guidance for AI coding agents working in this repo.

**Big Picture**:
- **Type**: Small ML notebook-first project for a classification task.
- **Core artifact**: `ml_project.ipynb` — contains the main data loading, preprocessing, model training, and evaluation code.
- **Data**: `train.csv`, `test.csv`, and `sample_submission.csv` are the primary integration points; any model/code changes should read/write compatible formats with `sample_submission.csv` for submissions.

**Key Patterns & Conventions (do this exactly)**:
- Use `df.select_dtypes(include=["float64"])` to select numeric features as the notebook currently does.
- Drop entirely-empty feature columns with `X = X.dropna(axis=1, how="all")` before modeling.
- Target column is `TRIAL_TYPE` — preserve its name and stratify splits by it (`stratify=y`).
- Use a deterministic seed `random_state=0` for train/test splits and any randomized ops to keep results reproducible.
- Pipeline pattern: the notebook uses `sklearn.pipeline.make_pipeline` with `SimpleImputer(strategy="median")`, `StandardScaler()`, and `LogisticRegression(max_iter=3000, n_jobs=-1)`. Follow this style for quick experiments.

**Files to Inspect / Update**:
- `ml_project.ipynb` — primary working notebook; prefer creating companion `.py` modules in `src/` if work grows.
- `train.csv`, `test.csv`, `sample_submission.csv` — data contract; ensure output columns/indices match `sample_submission.csv`.
- `README.md` — project-level notes; keep minimal changes here.

**Build / Run / Debug Guidance**:
- Notebook-first workflow: run locally with `jupyter lab` / `jupyter notebook`.
- For CI-style or automated runs, execute the notebook with `papermill` or `nbconvert` (example):
  - `pip install papermill` then `papermill ml_project.ipynb output.ipynb`
  - or `jupyter nbconvert --to notebook --execute ml_project.ipynb --output output.ipynb`
- If adding scripts, include a `requirements.txt` (not present) and prefer `python -m venv .venv && .venv/bin/pip install -r requirements.txt` for reproducible environments.

**Modeling / Data Notes**:
- Missing values: use median imputation for numerical features (as notebook does).
- Scaling: standardize numeric features before linear models; keep scaler inside the pipeline.
- If adding feature-engineering/code, keep it deterministic and document any new columns in the notebook or a short `docs/` note.

**What to Avoid / Gotchas**:
- Do not rename the `TRIAL_TYPE` column — many cells and submission logic rely on it.
- Avoid manipulating the original CSVs in-place in the repo; read, transform in-memory and write outputs to new files.
- Notebook cell metadata/ids are preserved in the repo—do not delete cells unless cleaning up interactively and committing intentional changes.

**Examples (copyable)**:
- Train/validate split used in the notebook:
  ```py
  X = df.select_dtypes(include=["float64"]).dropna(axis=1, how="all")
  y = df["TRIAL_TYPE"]
  X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
  ```
- Minimal pipeline pattern to follow:
  ```py
  clf = make_pipeline(
      SimpleImputer(strategy="median"),
      StandardScaler(),
      LogisticRegression(max_iter=3000, n_jobs=-1),
  )
  clf.fit(X_train, y_train)
  ```

**When adding larger changes**:
- Create `src/` and move reusable code there (data loading, preprocessing, model wrappers). Keep notebooks for analysis and short experiments.
- Add `requirements.txt` with exact package versions used for experiments (e.g., `pandas`, `scikit-learn`, `papermill`).

If anything in this file is unclear or you want extra examples (e.g., recommended `requirements.txt` or a starter `src/` layout), tell me which part to expand.
