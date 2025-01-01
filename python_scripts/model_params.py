import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


################################################################################
############################# Path Variables ###################################
################################################################################

model_output = "model_output"  # model output path
mlflow_data = "mlflow_data"  # path to store mlflow artificats (i.e., results)

################################################################################
############################ Global Constants ##################################
################################################################################

rstate = 222  # random state for reproducibility

################################################################################
############################# Stratification ###################################
################################################################################

# create bins for age along with labels such that age as a continuous series
# can be converted to something more manageable for visualization and analysis

bin_ages = [0, 18, 30, 40, 50, 60, 70, 80, 90, 100]

stratify_list = [
    "Catholic",
    "Jewish",
    "Atheist",
    "Buddhist",
    "Orthodox",
    "Muslims",
]

################################################################################
######################### Support Vector Machinbe ##############################
################################################################################

# Define SVM parameters
svm_name = "svm"

svc_kernel = ["linear", "rbf"]
svc_cost = np.logspace(-4, 0, 10).tolist()
svc_gamma = [0.001, 0.01, 0.1, 0.5, "scale", "auto"]

# Correct parameter name: 'C' instead of 'cost'
tuned_parameters_svm = [
    {"svm__kernel": svc_kernel, "svm__C": svc_cost, "svm__gamma": svc_gamma}
]

# Define the SVM model
svm = SVC(
    class_weight="balanced",
    probability=True,
)

# Define the SVM model configuration
svm_definition = {
    "clc": svm,
    "estimator_name": svm_name,
    "tuned_parameters": tuned_parameters_svm,
    "randomized_grid": False,
    "early": False,
}
################################################################################
########################## Logistic Regression #################################
################################################################################

# Define the hyperparameters for Logistic Regression
lr_name = "lr"

lr_penalties = ["l2"]
lr_Cs = np.logspace(-4, 0, 5)
# lr_max_iter = [100, 500]

# Structure the parameters similarly to the RF template
tuned_parameters_lr = [
    {
        "lr__penalty": lr_penalties[:1],
        "lr__C": lr_Cs[:1],
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
    "early": False,
}

################################################################################
########################## Random Forest Classifier ############################
################################################################################

# Define the hyperparameters for Random Forest
rf_name = "rf"

rf_n_estimators = [100, 200, 300]
rf_max_depths = [None, 5, 10]
rf_criterions = ["gini", "entropy"]
rf_parameters = [
    {
        "rf__n_estimators": rf_n_estimators,
        "rf__max_depth": rf_max_depths,
        "rf__criterion": rf_criterions,
    }
]

rf = RandomForestClassifier(
    class_weight="balanced",
    random_state=rstate,
    n_jobs=2,
)

rf_definition = {
    "clc": rf,
    "estimator_name": rf_name,
    "tuned_parameters": rf_parameters,
    "randomized_grid": False,
    "early": False,
}


model_definitions = {
    svm_name: svm_definition,
    lr_name: lr_definition,
    rf_name: rf_definition,
}
