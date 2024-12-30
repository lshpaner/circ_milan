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

svm_name = "svm"

svc_kernel = ["linear", "rbf"]
svc_cost = np.logspace(-4, 0, 10).tolist()
svc_gamma = [0.001, 0.01, 0.1, 0.5, "scale", "auto"]

tuned_parameters_svm = [{"svc__kernel": svc_kernel, "svc__cost": svc_cost}]

svm = SVC(
    class_weight="balanced",
    probability=True,
)

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

################################################################################
############################## XGBoost Classifier ##############################
################################################################################

# Estimator name prefix for use in GridSearchCV or similar tools
xgb_name = "xgb"

xgb = XGBClassifier(
    objective="binary:logistic",
    random_state=rstate,
    tree_method="hist",
    device="cuda",
    n_jobs=16,
)

# Define the hyperparameters for XGBoost
xgb_learning_rates = [0.0001]  # Learning rate or eta
xgb_n_estimators = [1000]  # Number of trees. Equivalent to n_estimators in GB
xgb_max_depths = [3, 5, 7]  # Maximum depth of the trees
xgb_subsamples = [0.8, 1.0]  # Subsample ratio of the training instances
xgb_colsample_bytree = [0.8, 1.0]

xgb_alpha = [0, 0.1, 1, 10]  # L1 regularization (alpha)
xgb_lambda = [0, 0.1, 10, 100]  # L2 regularization (lambda)
xgb_eval_metric = ["logloss"]  # check out "aucpr"
xgb_early_stopping_rounds = [3]
xgb_verbose = [0]
# Subsample ratio of columns when constructing each tree

# Combining the hyperparameters in a dictionary
xgb_parameters = [
    {
        "xgb__learning_rate": xgb_learning_rates,
        "xgb__n_estimators": xgb_n_estimators,
        "xgb__max_depth": xgb_max_depths,
        "xgb__subsample": xgb_subsamples,
        "xgb__alpha": xgb_alpha,  # L1 regularization (alpha)
        "xgb__lambda": xgb_lambda,  # L2 regularization (lambda)
        "xgb__colsample_bytree": xgb_colsample_bytree,
        "xgb__eval_metric": xgb_eval_metric,
        "xgb__early_stopping_rounds": xgb_early_stopping_rounds,
        "xgb__verbose": xgb_verbose,
    }
]

xgb_definition = {
    "clc": xgb,
    "estimator_name": xgb_name,
    "tuned_parameters": xgb_parameters,
    "randomized_grid": False,
    "early": True,
}

################################################################################
############################ CatBoost Classifier ###############################
################################################################################

cat_name = "cat"

cat = CatBoostClassifier(
    task_type="GPU",
    random_state=222,
)

# Define the hyperparameters for CatBoost
cat_depths = [4, 6, 8, 10]  # Depth of the trees
cat_learning_rates = [1e-4]  # Learning rate
cat_l2_leaf_regs = [3, 10, 100]  # L2 regularization
cat_bagging_temperatures = [0, 0.5, 1]  # Bagging temperature
cat_n_estimators = [10000]  # Number of trees
cat_early_stopping_rounds = [35]  # Early stopping rounds
cat_random_strengths = [1, 10]  # Random strength for feature score randomness
cat_verbose = [0]  # Verbosity level
cat_n_features_to_select = [5, 10, None]  # Features to select for RFE

# Combining the hyperparameters in a dictionary
cat_parameters = [
    {
        "cat__depth": cat_depths,
        "cat__learning_rate": cat_learning_rates,
        "cat__l2_leaf_reg": cat_l2_leaf_regs,
        "cat__bagging_temperature": cat_bagging_temperatures,
        "cat__n_estimators": cat_n_estimators,
        "cat__early_stopping_rounds": cat_early_stopping_rounds,
        "cat__random_strength": cat_random_strengths,
        "cat__verbose": cat_verbose,
    }
]

cat_definition = {
    "clc": cat,
    "estimator_name": cat_name,
    "tuned_parameters": cat_parameters,
    "randomized_grid": False,
    "early": True,
}


model_definitions = {
    svm_name: svm_definition,
    lr_name: lr_definition,
    rf_name: rf_definition,
    xgb_name: xgb_definition,
    cat_name: cat_definition,
}
