################################################################################
# STEP 1: Import required libraries and modules
################################################################################
from pathlib import Path
import typer
import pandas as pd

from functions import mlflow_loadArtifact, mlflow_load_model
from modeling.predict import find_best_model
from constants import (
    shap_artifact_name,
    shap_run_name,
    shap_artifacts_data,
)

from config import (
    PROCESSED_DATA_DIR,
)

from tqdm import tqdm

tqdm.pandas()

app = typer.Typer()


@app.command()
def main(
    outcome: str = "default_outcome",
    features_path: Path = PROCESSED_DATA_DIR / "X.parquet",
    labels_path: Path = PROCESSED_DATA_DIR / "y_income.parquet",
    metric_name: str = "valid AUC ROC",  # Metric to select the best model
    mode: str = "max",  # max for metrics where higher is better, min otherwise
    explanations_path: Path = "",
    shap_val_flag: int = 1,  # flag for whether or not to print vals next to feats.
    top_n: int = 5,  # top n feats.
    hold_out: str = "kfold",  # holdout set; `valid` for validation
    # -----------------------------------------
):

    ################################################################################
    # STEP 2: Set up experiment parameters
    ################################################################################
    experiment_name = f"{outcome}_model"

    ################################################################################
    # STEP 3: Find and load the best model
    ################################################################################

    run_name, estimator_name = find_best_model(
        experiment_name,
        metric_name,
        mode,
    )

    model_name = f"{estimator_name}_{outcome}"  # retrieve best model_name

    # Load best model and assign it to variable called model
    model = mlflow_load_model(experiment_name, run_name, model_name)

    ################################################################################
    # STEP 4: Load Processed Data (Features & Labels)
    ################################################################################

    X = pd.read_parquet(features_path)
    y = pd.read_parquet(labels_path)
    y = y.squeeze()  # Ensure labels are in Series format

    ################################################################################
    # STEP 5: Split Process into Validation, and Test Sets
    # This gives the end-user the flexibility to run this script on validation
    # or test data
    ################################################################################

    if hold_out == "valid":
        X_holdout, y_holdout = model.get_valid_data(X, y)
    elif hold_out == "test":
        X_holdout, y_holdout = model.get_test_data(X, y)
    elif hold_out == "kfold":
        X_holdout, y_holdout = (X, y)
    else:
        ValueError("Should be either valid or test")

    ################################################################################
    # STEP 6: Prepare the pipeline and transform data
    ################################################################################

    # Retrieve pipeline steps using built-in model_tuner getter
    X_holdout_transformed = (
        model.get_preprocessing_and_feature_selection_pipeline().transform(X_holdout)
    )

    ################################################################################
    # STEP 7: Load SHAP explainer
    ################################################################################
    # Load the SHAP explainer from artifact saved in explainer.py
    # using mlflow_dumpArtifact
    explainer = mlflow_loadArtifact(
        experiment_name=shap_artifact_name,
        run_name=shap_run_name,
        obj_name="explainer",
        artifacts_data_path=shap_artifacts_data,
    )

    ################################################################################
    # STEP 8: Compute SHAP values w/ progress bar
    ################################################################################
    print("Computing SHAP values...")
    with tqdm(total=X_holdout.shape[0], desc="SHAP Explaining") as pbar:
        shap_values = explainer(X_holdout_transformed)
        pbar.update(X_holdout.shape[0])

        ################################################################################
        # STEP 9: Process SHAP results
        ################################################################################

        # Extract transformed feature names from the preprocessing pipeline
        shap_feature_names = model.estimator[:-1].get_feature_names_out()

        # Debug: Inspect the shape of shap_values.values
        print("Shape of shap_values.values:", shap_values.values.shape)

        # For binary classification, select SHAP values for the positive class (class 1)
        # shap_values.values has shape (n_samples, n_features, n_classes), e.g., (194, 13, 2)
        # We want shape (n_samples, n_features), e.g., (194, 13), so select class 1 (index 1)
        shap_values_2d = shap_values.values[:, :, 1]

        # Convert SHAP values to DataFrame using transformed feature names
        shap_results = pd.DataFrame(
            shap_values_2d,
            columns=shap_feature_names,
            index=X_holdout.index,
        )
    ################################################################################
    # STEP 10: Extract top n SHAP features (can be top any # based on make command)
    ################################################################################

    print(f"Extracting Top {top_n} SHAP features per patient...")

    # Get the top n features and their original SHAP values
    # instead of .to_json(), build a dict straight off the nlargest:
    top_shap_pairs = shap_results.progress_apply(
        lambda row: row.abs().round(2).nlargest(top_n).to_dict(),
        axis=1,
    )

    ################################################################################
    # STEP 11: Create SHAP DataFrame
    ################################################################################

    # Initialize a DataFrame to store SHAP output per patient
    shap_df = pd.DataFrame(index=X_holdout.index)

    shap_df = pd.DataFrame(index=X_holdout.index)
    if shap_val_flag:
        # now each entry is already a dict
        shap_df[f"Top {top_n} Features"] = top_shap_pairs
    else:
        # extract just the keys
        shap_df[f"Top {top_n} Features"] = top_shap_pairs.progress_apply(
            lambda d: ", ".join(d.keys())
        )

    ################################################################################
    # STEP 12: Add confusion matrix and predictions metrics to dataframe
    ################################################################################
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import confusion_matrix
    import numpy as np
    from sklearn.base import clone

    # STEP 12a: pull out your preprocessing+selection and your estimator
    preproc = model.get_preprocessing_and_feature_selection_pipeline()
    clf = model.estimator._final_estimator  # or model.estimator.steps[-1][1]

    # STEP 12b: build a fresh pipeline for CV (so we don’t clobber your loaded one)
    from sklearn.pipeline import Pipeline

    cv_pipe = Pipeline(
        [
            ("preproc", clone(preproc)),
            ("clf", clone(clf)),
        ]
    )

    # STEP 12c: do 10-fold out-of-fold PROBAS
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=222)
    y_proba_oof = np.zeros(len(y_holdout))
    y_pred_oof = np.zeros(len(y_holdout), dtype=int)

    for train_idx, test_idx in cv.split(X_holdout, y_holdout):
        X_tr, y_tr = X_holdout.iloc[train_idx], y_holdout.iloc[train_idx]
        X_te = X_holdout.iloc[test_idx]

        # fit on this fold
        cv_pipe.fit(X_tr, y_tr)

        # predict probs + threshold >0.24
        probs = cv_pipe.predict_proba(X_te)[:, 1]
        preds = (probs > 0.24).astype(int)

        y_proba_oof[test_idx] = probs
        y_pred_oof[test_idx] = preds

    # STEP 12d: aggregate your confusion
    tn, fp, fn, tp = confusion_matrix(y_holdout, y_pred_oof).ravel()
    print(f"Aggregated (10-fold) TP @0.24 = {tp}")  # should be 50

    # now stick those OOF flags back into your DataFrame
    shap_df["y_pred_proba"] = y_proba_oof
    shap_df["y_pred"] = y_pred_oof
    shap_df["TP"] = ((y_holdout == 1) & (y_pred_oof == 1)).astype(int)
    shap_df["FN"] = ((y_holdout == 1) & (y_pred_oof == 0)).astype(int)
    shap_df["FP"] = ((y_holdout == 0) & (y_pred_oof == 1)).astype(int)
    shap_df["TN"] = ((y_holdout == 0) & (y_pred_oof == 0)).astype(int)

    ################################################################################
    # STEP 13: Save results to CSV file
    ################################################################################

    shap_df.to_csv(explanations_path, index=True)
    print(f"Results saved to '{explanations_path}'")


if __name__ == "__main__":
    app()
