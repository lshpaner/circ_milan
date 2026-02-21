
# Machine Learning-Based Predictions of Postoperative Outcomes in Adult Male Circumcision

<img src="https://github.com/lshpaner/circ_milan/blob/main/assets/CUT_MD_logo.svg" width="300">

---

## Table of Contents

- [Project Overview](#project-overview)
- [Data Access Requirement](#data-access-requirement)
- [Project Structure](#project-structure)
- [Python & Environment Requirements](#python--environment-requirements)
- [Pipeline Execution Guide](#pipeline-execution-guide)
  - [Step 1: Setup Directories & Environment](#step-1-setup-directories--environment)
  - [Step 2: Preprocessing & Feature Generation](#step-2-preprocessing--feature-generation)
  - [Step 3: Training](#step-3-training)
  - [Step 4: Evaluation](#step-4-evaluation)
  - [Full Pipeline in One Command](#full-pipeline-in-one-command)
  - [Explainability](#explainability)
  - [Inference / Production](#inference--production)
- [Modeling Details](#modeling-details)
- [MLflow Tracking](#mlflow-tracking)
- [Artifacts & Outputs](#artifacts--outputs)
- [Reproducibility](#reproducibility)
- [Authors & Contacts](#authors--contacts)
- [License](#license)

---

## Project Overview

This repository contains the complete machine learning pipeline for preprocessing, modeling, evaluating, and explaining postoperative outcomes related to laser circumcision procedures.

Primary supervised learning target:

- `Bleeding_Edema_Outcome`

The workflow includes:

- Raw data preprocessing
- Feature engineering
- Model training across multiple sampling strategies
- Model evaluation
- SHAP-based explainability
- Inference pipeline for production use
- MLflow experiment tracking

---

## Data Access Requirement

The dataset used in this repository is **not publicly distributed**.

To reproduce results:

1. Obtain the dataset directly from the authors with permission.
2. Place the raw Excel file into:

```
data/raw/Laser_Circumcision_Excel_31.03.2024.xlsx
```

No pipeline step will function until this file is present.

---

## Project Structure

```
circ_milan/
├── core/
│   ├── config.py          # All hyperparameters and configuration
│   ├── constants.py
│   └── functions.py
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   │   └── inference/
├── mlruns/                # MLflow tracking
├── preprocessing/
│   ├── init_project.py
│   ├── create_folders.py
│   ├── preprocessing.py
│   └── feat_gen.py
├── modeling/
│   ├── train.py
│   ├── evaluation.py
│   ├── explainer.py
│   ├── explanations_training.py
│   ├── explanations_inference.py
│   └── predict.py
├── models/
├── notebooks/
├── Makefile
└── requirements.txt
```

---

## Python & Environment Requirements

This project requires **Python 3.11**.

The Makefile does NOT automatically create environments.
It prints instructions and prepares structure only.

### Option A: Conda (Recommended)

```
conda create -n conda_circ_311 python=3.11
conda activate conda_circ_311
pip install -r requirements.txt
```

### Option B: venv (Must Piggyback Off Python 3.11)

The venv must inherit a Python 3.11 interpreter.

You MUST already be inside a Python 3.11 environment, such as the conda environment above.

```
conda activate conda_circ_311
python -m venv venv_circ_311
source venv_circ_311/bin/activate
pip install -r requirements.txt
```

If you are not using Python 3.11, this will create the wrong interpreter.

---

# Pipeline Execution Guide

You may use Make (recommended) or run scripts manually.

---

## Step 1: Setup Directories & Environment

Run:

```
make setup_dir_venv
make requirements
```

This:

- Creates project folder structure
- Initializes required directories
- Prints environment instructions
- Does NOT auto-activate environments

Manual equivalent:

```
python preprocessing/init_project.py
python preprocessing/create_folders.py
```

---

## Step 2: Preprocessing & Feature Generation

Recommended:

```
make preproc_pipeline
```

Manual:

```
python preprocessing/preprocessing.py --stage training
python preprocessing/feat_gen.py --stage training
```

Artifacts produced:

- Saved locally in `data/processed/`
- Logged to MLflow under `mlruns/`

---

## Step 3: Training

Supported models:

- `lr`  (Logistic Regression)
- `rf`  (Random Forest)
- `svm` (Support Vector Machine)

Sampling pipelines:

- `orig`
- `smote`
- `over`

All hyperparameters are stored inside:

```
core/config.py
```

Recommended:

```
make train_all_models
```

Manual example:

```
python modeling/train.py   --model-type lr   --pipeline-type orig   --features-path ./data/processed/X.parquet   --labels-path ./data/processed/y_Bleeding_Edema_Outcome.parquet   --outcome Bleeding_Edema_Outcome
```

---

## Step 4: Evaluation

Recommended:

```
make eval_all_models
```

Manual example:

```
python modeling/evaluation.py   --model-type lr   --pipeline-type orig   --features-path ./data/processed/X.parquet   --labels-path ./data/processed/y_Bleeding_Edema_Outcome.parquet   --outcome Bleeding_Edema_Outcome
```

Evaluation results saved to:

```
models/eval/
```

Metrics also logged to MLflow.

---

## Full Pipeline in One Command

Run:

```
make preproc_train_eval
```

This executes:

- preproc_pipeline
- train_all_models
- eval_all_models

### Why Use Make?

Make:

- Automatically loops over models and pipelines
- Injects correct arguments
- Keeps configuration centralized
- Prevents manual errors
- Improves reproducibility

---

## Explainability

Best model selection:

```
make model_explainer
```

SHAP on training data:

```
make model_explanations_training
```

Combined:

```
make model_explaining_training
```

SHAP on inference data:

```
make model_explanations_inference
```

SHAP outputs stored in:

```
data/processed/
data/processed/inference/
```

---

## Inference / Production

Run:

```
make preproc_pipeline_inf
```

This executes:

- preprocessing in inference mode
- feature generation in inference mode
- prediction

Predictions saved to:

```
data/processed/inference/predictions_Bleeding_Edema_Outcome.csv
```

---

## Modeling Details

Outcome:

- Bleeding_Edema_Outcome

Models:

- Logistic Regression
- Random Forest
- Support Vector Machine

Metric:

- average_precision

Hyperparameters centralized in:

```
core/config.py
```

---

## MLflow Tracking

All preprocessing, training, evaluation, and artifacts are logged to:

```
mlruns/
```

Launch UI:

```
make mlflow_ui
```

Then open:

```
http://localhost:5501
```

---

## Artifacts & Outputs

Generated artifacts include:

- Cleaned datasets
- Feature matrices
- Trained models
- Evaluation metrics
- SHAP values
- Inference predictions

Stored in:

- `data/processed/`
- `models/`
- `mlruns/`

---

## Reproducibility

To fully reproduce the full pipeline:

```
make setup_dir_venv
make requirements
make preproc_train_eval
make model_explaining_training
```

---

## Authors & Contacts

Leonid Shpaner, M.S.  
Data Scientist | Adjunct Professor  

Giuseppe Saitta, M.D.  
Medical Consultant, Data Provider  

---

## License

MIT License.
