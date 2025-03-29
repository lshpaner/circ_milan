from model_tuner import loadObjects
import pandas as pd
import numpy as np
import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt

print(os.path.join(os.pardir, ".."))
sys.path.append(".")

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from python_scripts.functions import ModelCalculator

from model_tuner import *
import shap

if __name__ == "__main__":
    argv = sys.argv[1:]
    model_path = argv[0]
    data_path = argv[1]
    model_arg_name = argv[2]
    pickled_model_object = argv[3]
    csv_filename_per_patient = argv[4]
    csv_filename_overall = argv[5]

    ##################### Hard-coded paths for debugging #######################

    # model_path = "../../models/results/orig_models/"
    # data_path = "../../data/processed"

    ############################################################################
    ######################### Read in Model Object #############################
    ############################################################################

    model_name = loadObjects(
        os.path.join(
            model_path,
            pickled_model_object,
        )
    )

    print("Loaded model object:", model_name)
    print("Original model argument:", model_arg_name)

    ############################################################################
    ############################# Read in Data #################################
    ############################################################################

    X = pd.read_parquet(os.path.join(data_path, "X.parquet"))
    y = pd.read_parquet(os.path.join(data_path, "y.parquet"))


################################################################################
############################# Model Calculator #################################
################################################################################

## Initialize the ModelCalculator
calculator = ModelCalculator(
    model_dict=model_name,
    outcomes=y,
    top_n=5,
)

############################### Row-wise SHAP ##################################

## Generate the predictions and SHAP contributions
results_df_per_patient = calculator.generate_predictions(
    X_test=X,
    y_test=y,
    calculate_shap=False,
    use_coefficients=True,
    include_contributions=False,
    subset_results=True,
)
print(os.path.join(data_path, csv_filename_per_patient))
results_df_per_patient.to_csv(os.path.join(data_path, csv_filename_per_patient))

########################## Overall Coefficients/SHAP ###########################

# Use the original model argument name to check for logistic regression
print(model_arg_name)
if model_arg_name == "model_svm":
    print()
    print("Calculating global coefficients for Logistic Regression...")
    print()
    results_df = calculator.generate_predictions(
        X_test=X,
        y_test=y,
        global_coefficients=True,  # Only calculate coefficients
    )
else:
    print()
    print("Calculating SHAP values for other models...")
    print()
    results_df = calculator.generate_predictions(
        X_test=X,
        y_test=y,
        global_shap=True,  # Only calculate SHAP values
    )
# Save the results
results_df.to_csv(os.path.join(data_path, csv_filename_overall))
