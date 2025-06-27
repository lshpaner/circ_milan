import numpy as np

from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from constants import (
    exp_artifact_name,
    preproc_run_name,
)
from functions import mlflow_loadArtifact

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PROCESSED_DATA_DIR_INFER = DATA_DIR / "processed/inference"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"
RESULTS_DIR = PROJ_ROOT / MODELS_DIR / "results"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

features_path = PROCESSED_DATA_DIR / "X.parquet"

################################################################################
############################ Global Constants ##################################
################################################################################

rstate = 222  # random state for reproducibility
threshold_target_metric = "precision"  # target metric for threshold optimization
target_precision = 0.5  # target precision for threshold optimization

sampler_definitions = {
    "None": None,
    "SMOTE": SMOTE(random_state=rstate),
    "RandomOverSampler": RandomOverSampler(random_state=rstate),
}

################################################################################
# This section here is for categorical variables

categorical_cols = []

# Load feature column names from Mlflow
try:
    X_columns_list = mlflow_loadArtifact(
        experiment_name=exp_artifact_name,
        run_name=preproc_run_name,  # Use the same run_name as training
        obj_name="X_columns_list",
        verbose=False,
    )
    if X_columns_list is None:
        raise ValueError("X_columns_list is None - failed to load from artifacts")
except Exception as e:
    raise Exception(f"Failed to load X_columns_list: {str(e)}")

# Subset the numerical columns only; categorical columns are already defined above
numerical_cols = [col for col in X_columns_list if col not in categorical_cols]


################################################################################
############################### Transformers ###################################
################################################################################

numerical_transformer = Pipeline(
    steps=[
        ("scaler", MinMaxScaler()),
        ("imputer", SimpleImputer(strategy="mean")),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
)

# Create the ColumnTransformer with passthrough
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_cols),
    ],
    remainder="passthrough",
    # prevents prepending transformer names (e.g., 'remainder_') to output
    # feature names
    verbose_feature_names_out=False,
)

################################################################################
################################ Pipelines #####################################
################################################################################

pipeline_scale_imp = [
    ("Preprocessor", preprocessor),
]

pipelines = {
    "orig": {
        "pipeline": pipeline_scale_imp,
        "sampler": None,
        "feature_selection": False,  # No feature selection for orig
    },
    "smote": {
        "pipeline": pipeline_scale_imp,
        "sampler": SMOTE(random_state=rstate),
        "feature_selection": False,  # No feature selection for smote
    },
    "over": {
        "pipeline": pipeline_scale_imp,
        "sampler": RandomOverSampler(random_state=rstate),
        "feature_selection": False,  # No feature selection for under
    },
}


################################################################################
########################## Logistic Regression #################################
################################################################################

# Define the hyperparameters for Logistic Regression
lr_name = "lr"

lr_penalties = ["l2"]
lr_Cs = np.logspace(-4, 0, 2)
# lr_max_iter = [100, 500]

# Structure the parameters similarly to the RF template
tuned_parameters_lr = [
    {
        "lr__penalty": lr_penalties,
        "lr__C": lr_Cs,
    }
]

lr = LogisticRegression(
    class_weight="balanced",
    random_state=rstate,
    n_jobs=2,
)

lr_definition = {
    "clc": lr,
    "estimator_name": lr_name,
    "tuned_parameters": tuned_parameters_lr,
    "randomized_grid": False,
    "n_iter": False,
    "early": False,
}

################################################################################
########################## Random Forest Classifier ############################
################################################################################

# Define the hyperparameters for Random Forest (trimmed for efficiency)
rf_name = "rf"

# Reduced hyperparameters for tuning
rf_parameters = [
    {
        "rf__n_estimators": [10, 50],  # Reduce number of trees for speed
        "rf__max_depth": [None, 10],  # Limit depth to prevent overfitting
        "rf__min_samples_split": [2, 5],  # Fewer options for splitting
    }
]

# Initialize the Random Forest Classifier with a smaller number of trees
rf = RandomForestClassifier(
    n_estimators=10,
    max_depth=None,
    random_state=rstate,
    n_jobs=-1,
)

# Define the Random Forest model setup
rf_definition = {
    "clc": rf,
    "estimator_name": rf_name,
    "tuned_parameters": rf_parameters,
    "randomized_grid": False,
    "n_iter": False,
    "early": False,
}

################################################################################
########################### Support Vector Machines ############################
################################################################################
# Define SVM parameters
svm_name = "svm"

svc_kernel = ["linear", "rbf", "poly", "sigmoid"]
svc_cost = np.logspace(-4, 2, 10).tolist()
svc_gamma = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, "scale", "auto"]

# Correct parameter name: 'C' instead of 'cost'
tuned_parameters_svm = [
    {
        "svm__kernel": svc_kernel,
        "svm__C": svc_cost,
        "svm__gamma": svc_gamma,
    }
]

# Define the SVM model
svm = SVC(
    class_weight="balanced",
    probability=True,
    random_state=rstate,
)

# Define the SVM model configuration
svm_definition = {
    "clc": svm,
    "estimator_name": svm_name,
    "tuned_parameters": tuned_parameters_svm,
    "randomized_grid": False,
    "n_iter": False,
    "early": False,
}

################################################################################
############################ Support Vector Machine ############################
################################################################################


model_definitions = {
    lr_name: lr_definition,
    rf_name: rf_definition,
    svm_name: svm_definition,
}
