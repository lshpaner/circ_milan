################################################################################
# STEP 1: Import required libraries and modules
################################################################################
from pathlib import Path
import typer
import pandas as pd
import json

from functions import mlflow_loadArtifact, mlflow_load_model
from modeling.predict import find_best_model
from constants import (
    shap_artifact_name,
    shap_run_name,
    shap_artifacts_data,
)

from config import (
    PROCESSED_DATA_DIR_INFER,
)

from tqdm import tqdm

tqdm.pandas()

app = typer.Typer()


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR_INFER / "X.parquet",
    outcome: str = "default_outcome",
    metric_name: str = "valid AUC ROC",  # Metric to select the best model
    mode: str = "max",  # max for metrics where higher is better, min otherwise
    top_n: int = 5,
    shap_val_flag: int = 1,  # flag for whether or not to print vals next to feats.
    explanations_path: Path = "",
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
    # STEP 4: Load processed data (features & labels)
    ################################################################################

    X = pd.read_parquet(features_path)

    ################################################################################
    # STEP 5: Prepare the pipeline and transform data
    ################################################################################

    # Retrieve pipeline steps using built-in model_tuner getter
    X_transformed = model.get_preprocessing_and_feature_selection_pipeline().transform(
        X
    )

    ################################################################################
    # STEP 6: Load SHAP explainer
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
    # STEP 7: Compute SHAP values w/ progress bar
    ################################################################################
    print("Computing SHAP values...")
    with tqdm(total=X.shape[0], desc="SHAP Explaining") as pbar:
        shap_values = explainer(X_transformed)
        pbar.update(X.shape[0])

    ################################################################################
    # STEP 8: Process SHAP results
    ################################################################################

    # Extract transformed feature names from the preprocessing pipeline
    shap_feature_names = model.estimator.estimator[:-1].get_feature_names_out()

    # Convert SHAP values to DataFrame using transformed feature names
    shap_results = pd.DataFrame(
        shap_values.values,
        columns=shap_feature_names,
        index=X.index,
    )

    ################################################################################
    # STEP 9: Extract top n SHAP features
    ################################################################################
    print(f"Extracting Top {top_n} SHAP features per patient...")

    # Get the top n features and their original SHAP values
    top_shap_pairs = shap_results.progress_apply(
        lambda row: row.abs().round(2).nlargest(top_n).to_json(),
        axis=1,
    )

    ################################################################################
    # STEP 10: Create SHAP DataFrame
    ################################################################################

    shap_df = pd.DataFrame(index=X.index)

    if shap_val_flag:
        # Store top N features with their SHAP values as JSON strings
        joined = []
        for i in tqdm(range(top_shap_pairs.shape[0])):
            shap_feats = top_shap_pairs.iloc[i]
            joined.append(json.dumps(shap_feats))
        shap_df[f"Top {top_n} Features"] = joined

    else:
        # Store only the names of the top N features (without SHAP values)
        # In the if, feats are being added; here are just feat_names
        shap_df[f"Top {top_n} Features"] = top_shap_pairs.apply(
            lambda x: ", ".join(map(str, list(json.loads(x).keys()))),
        )

    ################################################################################
    # STEP 11: Add confusion matrix and predictions metrics to dataframe
    ################################################################################
    # Generate predictions and predicted probabilities on X
    y_pred = model.predict(X, optimal_threshold=True)
    y_pred_proba = model.predict_proba(X)[:, 1]

    # Store predictions and predicted probabilities
    shap_df["y_pred"] = y_pred
    shap_df["y_pred_proba"] = y_pred_proba

    ################################################################################
    # STEP 12: Save results to CSV
    ################################################################################
    shap_df.to_csv(explanations_path, index=True)
    print(f"Results saved to '{explanations_path}'")


if __name__ == "__main__":
    app()
