import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.calibration import CalibratedClassifierCV


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
############################### SGD Classifier #################################
################################################################################

from sklearn.linear_model import SGDClassifier

# Define the hyperparameters for SGD Classifier
sgd_name = "sgd"

# Hyperparameters for tuning
sgd_parameters = [
    {
        "sgd__loss": [
            "hinge",
            "log_loss",
            "modified_huber",
            "squared_hinge",
            "perceptron",
        ],  # Different loss functions for classification
        "sgd__penalty": ["l1", "l2", "elasticnet"],  # Regularization options
        "sgd__alpha": [1e-4, 1e-3, 1e-2],  # Regularization strength
        "sgd__max_iter": [1000, 2000, 5000],  # Number of iterations
        "sgd__tol": [1e-3, 1e-4, 1e-5],  # Stopping criteria
        "sgd__learning_rate": ["constant", "optimal", "invscaling", "adaptive"],
    }
]

# Initialize the SGD Classifier
sgd = SGDClassifier(
    loss="hinge",
    penalty="l2",
    alpha=1e-4,
    max_iter=1000,
    tol=1e-3,
)

# Define the SGD model setup
sgd_definition = {
    "clc": sgd,
    "estimator_name": sgd_name,
    "tuned_parameters": sgd_parameters,
    "randomized_grid": False,
    "early": False,
}


################################################################################

# Define the hyperparameters for MLP
mlp_name = "mlp"

# MLP hyperparameters
mlp_hidden_layer_sizes = [(32,), (64,), (32, 16)]  # Lightweight hidden layers
mlp_activation = ["relu", "tanh"]  # Common activation functions
mlp_solver = ["adam", "sgd"]  # Solvers for optimization
mlp_alpha = [0.0001, 0.001, 0.01]  # Regularization parameter
mlp_learning_rate_init = [0.001, 0.01]  # Initial learning rate

# Create a list of hyperparameter combinations
mlp_parameters = [
    {
        "mlp__hidden_layer_sizes": mlp_hidden_layer_sizes,
        "mlp__activation": mlp_activation,
        "mlp__solver": mlp_solver,
        "mlp__alpha": mlp_alpha,
        "mlp__learning_rate_init": mlp_learning_rate_init,
    }
]

# Initialize the MLP classifier
mlp = MLPClassifier(max_iter=1000, random_state=rstate, early_stopping=True)

# Define the MLP model setup
mlp_definition = {
    "clc": mlp,
    "estimator_name": mlp_name,
    "tuned_parameters": mlp_parameters,
    "randomized_grid": False,  # Set to True for randomized search if desired
    "early": False,  # Enable early stopping to prevent overfitting
}

################################################################################
############################### Decision Trees #################################
################################################################################

# Define Decision Tree parameters
dt_name = "dt"

# Simplified hyperparameters
dt_max_depth = [None, 10, 20]  # Unbounded, shallow, and medium depths
dt_min_samples_split = [2, 10]  # Default and a stricter option
dt_min_samples_leaf = [1, 5]  # Default and larger leaf nodes for regularization

# Define the parameter grid for Decision Trees
tuned_parameters_dt = [
    {
        "dt__max_depth": dt_max_depth,
        "dt__min_samples_split": dt_min_samples_split,
        "dt__min_samples_leaf": dt_min_samples_leaf,
    }
]

# Define the Decision Tree model
dt = DecisionTreeClassifier(
    class_weight="balanced",
    random_state=rstate,
)

# Define the Decision Tree model configuration
dt_definition = {
    "clc": dt,
    "estimator_name": dt_name,
    "tuned_parameters": tuned_parameters_dt,
    "randomized_grid": False,
    "early": False,
}


################################################################################
#########################      Random Forest      ##############################
################################################################################

# from sklearn.ensemble import RandomForestClassifier

# # Define the hyperparameters for Random Forest
# rf_name = "rf"

# # Hyperparameters for tuning
# rf_parameters = [
#     {
#         "rf__n_estimators": [50, 100, 200],  # Number of trees
#         "rf__criterion": ["gini", "entropy", "log_loss"],  # Splitting criteria
#         "rf__max_depth": [None, 10, 20, 30],  # Maximum depth of trees
#         "rf__min_samples_split": [2, 5, 10],  # Minimum samples required to split
#         "rf__min_samples_leaf": [1, 2, 4],  # Minimum samples per leaf
#         "rf__bootstrap": [True, False],  # Bootstrapping for bagging
#     }
# ]

# # Initialize the Random Forest Classifier
# rf = RandomForestClassifier(
#     n_estimators=100, criterion="gini", max_depth=None, random_state=42
# )

# # Define the Random Forest model setup
# rf_definition = {
#     "clc": rf,
#     "estimator_name": rf_name,
#     "tuned_parameters": rf_parameters,
#     "randomized_grid": False,
#     "early": False,
# }

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
rf = RandomForestClassifier(n_estimators=10, max_depth=None, random_state=42, n_jobs=-1)

# Define the Random Forest model setup
rf_definition = {
    "clc": rf,
    "estimator_name": rf_name,
    "tuned_parameters": rf_parameters,
    "randomized_grid": False,
    "early": False,
}


################################################################################
######################### Support Vector Machines ##############################
################################################################################

# Define SVM parameters
svm_name = "svm"

svc_kernel = ["linear", "rbf", "poly", "sigmoid"]
svc_cost = np.logspace(-4, 2, 10).tolist()
# svc_cost = np.logspace(-4, 0, 1).tolist()
svc_gamma = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, "scale", "auto"]

# Correct parameter name: 'C' instead of 'cost'
tuned_parameters_svm = [
    {"svm__kernel": svc_kernel, "svm__C": svc_cost, "svm__gamma": svc_gamma}
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
    "early": False,
}


######################### Nu Support Vector Machines ###########################

# Define NuSVC parameters
nusvc_name = "nusvc"

nusvc_kernel = ["linear", "rbf", "poly", "sigmoid"]  # Common kernels
nusvc_nu = [0.1, 0.3, 0.5, 0.7, 0.9]  # Fraction of errors and SVs
nusvc_gamma = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, "scale", "auto"]  # Gamma values

# Parameter grid for NuSVC
tuned_parameters_nusvc = [
    {
        "nusvc__kernel": nusvc_kernel,
        "nusvc__nu": nusvc_nu,
        "nusvc__gamma": nusvc_gamma,
    }
]

# Define the NuSVC model
nusvc = NuSVC(
    class_weight="balanced",  # Balance classes if necessary
    probability=True,  # Enable probability estimates
    random_state=rstate,
)

# Define the NuSVC model configuration
nusvc_definition = {
    "clc": nusvc,
    "estimator_name": nusvc_name,
    "tuned_parameters": tuned_parameters_nusvc,
    "randomized_grid": False,  # Use full grid search
    "early": False,  # No early stopping for NuSVC
}


################################################################################

model_definitions = {
    svm_name: svm_definition,
    nusvc_name: nusvc_definition,
    lr_name: lr_definition,
    sgd_name: sgd_definition,
    mlp_name: mlp_definition,
    dt_name: dt_definition,
    rf_name: rf_definition,
}
