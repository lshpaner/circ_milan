import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


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


# ################################################################################
# ############################# KNeighborsClassifier #############################
# ################################################################################

# Define the hyperparameters for K-Nearest Neighbors
knn_name = "knn"

knn_neighbors = [3, 5, 7, 9]  # Number of neighbors to consider
knn_weights = ["uniform", "distance"]  # Uniform weights or distance-based weights
knn_metrics = ["euclidean", "manhattan", "minkowski"]  # Distance metrics

knn_parameters = [
    {
        "knn__n_neighbors": knn_neighbors,
        "knn__weights": knn_weights,
        "knn__metric": knn_metrics,
    }
]

# Initialize the k-NN Classifier
knn = KNeighborsClassifier(n_jobs=-1)  # Use all available cores for faster computation

# Define the k-NN model setup
knn_definition = {
    "clc": knn,
    "estimator_name": knn_name,
    "tuned_parameters": knn_parameters,
    "randomized_grid": False,
    "early": False,
}


################################################################################
############################ Gaussian Naive Bayes ##############################
################################################################################

# Define the hyperparameters for Naive Bayes
nb_name = "nb"

# Naive Bayes doesn't have many hyperparameters, but we can tune prior probabilities
nb_priors = [None]  # Default is None; this can be extended if needed

nb_parameters = [
    {
        "nb__priors": nb_priors,
    }
]

# Initialize the Naive Bayes classifier
nb = GaussianNB()

# Define the Naive Bayes model setup
nb_definition = {
    "clc": nb,
    "estimator_name": nb_name,
    "tuned_parameters": nb_parameters,
    "randomized_grid": False,
    "early": False,
}


################################################################################
######################### Quadratic Discriminant Analysis ######################
################################################################################

# Define the hyperparameters for LDA
lda_name = "lda"

# Hyperparameters for tuning (if needed, e.g., solver selection or shrinkage)
lda_parameters = [
    {
        "lda__solver": [
            "lsqr",
            "eigen",
        ],  # SVD is default; lsqr/eigen support shrinkage
        "lda__shrinkage": [
            None,
            "auto",
            0.1,
            0.5,
        ],  # Only applicable for lsqr/eigen solvers
    }
]

# Initialize the LDA Classifier
lda = LinearDiscriminantAnalysis(solver="svd")  # SVD is robust and avoids shrinkage

# Define the LDA model setup
lda_definition = {
    "clc": lda,
    "estimator_name": lda_name,
    "tuned_parameters": lda_parameters,
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


#################

model_definitions = {
    svm_name: svm_definition,
    lr_name: lr_definition,
    nb_name: nb_definition,
    knn_name: knn_definition,
    lda_name: lda_definition,
    mlp_name: mlp_definition,
}
