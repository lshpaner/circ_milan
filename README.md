# Machine Learning-Based Predictions of Postoperative Outcomes in Adult Male Circumcision

<img src="https://github.com/lshpaner/circ_milan/blob/main/assets/CUT_MD.svg" width="300" style="border: none; outline: none; box-shadow: none;" oncontextmenu="return false;">

---

## 📌 Project Overview

This repository contains the full data science pipeline for preprocessing, modeling, evaluating, and explaining clinical outcomes related to laser circumcision procedures. It focuses specifically on predicting the `Bleeding_Edema_Outcome` complication using multiple supervised learning approaches. The workflow includes data cleaning, feature engineering, model training with different sampling strategies, evaluation, and SHAP-based explainability.

---

## 🧱 Project Structure

```text
circ_milan/
├── assets/                         # Slide decks and static visuals
│   ├── CUT_MD.svg
│   └── my_slides.html
├── data/                           # Datasets at different stages
│   ├── external/                   # Original source files
│   ├── raw/                        # Raw ingested data
│   │   └── Laser_Circumcision_Excel_31.03.2024.xlsx
│   ├── interim/                    # Intermediate cleaned files
│   └── processed/                  # Final data for modeling
│       ├── training/               # Training features and labels
│       │   ├── X.parquet
│       │   └── y_Bleeding_Edema_Outcome.parquet
│       └── inference/              # Inference features and outputs
│           ├── df_inference_process.parquet
│           └── X.parquet
├── images/                         # Exported plots and figures
│   └── figures/
├── mlruns/                         # MLflow tracking server backend logs
├── preprocessing/                  # Data cleaning & feature engineering
│   ├── __init__.py
│   ├── preprocessing.py            # Cleans raw data and saves interim/processed
│   └── feat_gen.py                 # Generates model-ready feature sets
├── modeling/                       # Modeling & explainability scripts
│   ├── __init__.py
│   ├── train.py                    # Train LR, RF, SVM with sampling pipelines
│   ├── evaluation.py               # Evaluate model performance
│   ├── explainer.py                # Select best model & build SHAP explainer
│   ├── explanations_training.py    # Compute SHAP values on training data
│   ├── explanations_inference.py   # Compute SHAP values on inference data
│   └── predict.py                  # Run production predictions
├── models/                         # Stored model artifacts & metrics
│   ├── results/                    # Logs & metrics per outcome
│   │   └── Bleeding_Edema_Outcome/
│   └── eval/                       # Evaluation reports per outcome
│       └── Bleeding_Edema_Outcome/
├── notebooks/                      # Jupyter notebooks for analysis & reporting
│   ├── circ_milan_eda.ipynb
│   ├── circ_milan_model_artifacts_dash.ipynb
│   ├── circ_milan_model_results.ipynb
│   ├── circ_milan_model_explanations.ipynb
│   └── post_modeling_eda.ipynb
├── unittests/                      # Unit tests for core modules
├── config.py                       # Central configuration settings
├── constants.py                    # Global constants
├── functions.py                    # General helper functions
├── project_functions.py            # Project-specific utilities
├── requirements.txt                # Python dependencies
├── setup.py                        # Packaging/install script
├── Makefile                        # Automates setup, training, evaluation, inference
└── README.md                       # Project overview and usage instructions

```

---

## 🔧 Installation

1. **Clone the repo**  
    ```bash
    git clone https://github.com/your-username/circ_milan.git
    cd circ_milan
    ```

2. **Create environment**  
   - **Conda**:  
     ```bash
     conda create -n conda_circ_311 python=3.11
     conda activate conda_circ_311
     ```  
   - **venv**:  
     ```bash
     python -m venv venv_circ_311
     source venv_circ_311/bin/activate
     ```

3. **Install dependencies**  
    ```bash
    pip install -r requirements.txt
    ```

---

## ⚙️ Makefile Commands

| Command                         | Description                                                    |
|---------------------------------|----------------------------------------------------------------|
| `make create_venv`              | Create a virtual environment                                   |
| `make requirements`             | Install dependencies                                           |
| `make preproc_pipeline`         | Run preprocessing + feature generation for training            |
| `make train_all_models`         | Train LR, RF, and SVM models                                   |
| `make eval_all_models`          | Evaluate all trained models                                    |
| `make preproc_train_eval`       | Full pipeline: preprocessing → training → evaluation           |
| `make model_explaining_training`| Run SHAP explainability on training data                       |
| `make preproc_pipeline_inf`     | Run preprocessing + feature generation for inference           |
| `make predict`                  | Run inference and output predictions                           |
| `make mlflow_ui`                | Launch MLflow UI on port 5501                                  |

To list available commands:

```bash
make help
```

## 🔍 Modeling Details

- **Outcome**: `Bleeding_Edema_Outcome`
- **Sampling Pipelines**:  
  - `orig` (original)  
  - `smote` (Synthetic Minority Oversampling)  
  - `over` (Random Oversampling)
- **Models**:  
  - Logistic Regression (`lr`)  
  - Random Forest (`rf`)  
  - Support Vector Machine (`svm`)
- **Metric**: `average_precision`
- **Explainability**: SHAP feature attributions via `explainer.py`

## 📊 MLflow Tracking

All runs, parameters, and metrics are tracked with MLflow.

Launch UI:

```bash
make mlflow_ui
```

## 📝 Notebooks

- `circ_milan_eda.ipynb` – Exploratory Data Analysis  
- `circ_milan_model_results.ipynb` – Model performance visuals  
- `circ_milan_model_explanations.ipynb` – SHAP visualizations  
- `post_modeling_eda.ipynb` – Further diagnostics  

## 🔐 Notes

- SHAP outputs and model artifacts are in `data/processed/` and `models/`  
- Inference predictions are saved to  
  `./data/processed/inference/predictions_Bleeding_Edema_Outcome.csv`  

## 🧪 Reproducibility

Run the full pipeline with:

```bash
make preproc_train_eval
```

## 📎 Authors & Contacts

- **Leonid Shpaner, M.S.**, Data Scientist | Adjunct Professor  
- **Giuseppe Saitta, M.D.**, Medical Consultant (data provider and clinical insights)

## 📄 License

Research and educational use only, all rights reserved unless stated otherwise

