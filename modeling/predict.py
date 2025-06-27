################################################################################
# STEP 1: Import Required Libraries and Modules
################################################################################
# Import standard libraries, MLflow tools, and custom modules for data handling
# and model inference

from pathlib import Path
import typer
from loguru import logger
import mlflow
import pandas as pd
import os

from constants import mlflow_models_data
from config import PROCESSED_DATA_DIR_INFER
from functions import mlflow_load_model

app = typer.Typer()

################################################################################
# STEP 2: Define Function to Find the Best Model from MLflow Experiment
################################################################################
# Define a helper function to identify the best model based on a specified metric


def find_best_model(
    experiment_name: str,
    metric_name: str,
    mode: str = "max",
    mlruns_location: str = None,
) -> str:
    """
    Finds the best model from a given MLflow experiment based on a specified
    metric.

    :param experiment_name: The name of the MLflow experiment to search in.
    :param metric_name: The metric used to determine the best model.
    :param mode: Specify "max" to select model based on maximum metric value
                 or "min" for minimum. Default is "max".
    :return: The run ID of the best model.
    :raises ValueError: If the experiment does not exist.
    """
    # Get experiment by name
    if mlruns_location is None:
        abs_mlflow_data = os.path.abspath(mlflow_models_data)

    mlflow.set_tracking_uri(f"file://{abs_mlflow_data}")

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(f"Experiment '{experiment_name}' does not exist.")

    experiment_id = experiment.experiment_id

    # Get all runs for the experiment
    order_clause = (
        f"metrics.`{metric_name}` DESC"
        if mode == "max"
        else f"metrics.`{metric_name}` ASC"
    )

    runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        order_by=[order_clause],
    )
    if runs.empty:
        raise ValueError(f"No runs found for experiment '{experiment_name}'")
    # Return the run ID with the best performance metric
    best_run = runs.iloc[0]  # Get the best run
    best_run_id = runs.iloc[0]["run_id"]
    best_metric_value = runs.iloc[0][f"metrics.{metric_name}"]
    print(f"Best Run ID: {best_run_id}, Best {metric_name}: {best_metric_value}")

    # Extract model_type from run_name or parameters
    run_name = best_run["tags.mlflow.runName"]

    # Extract estimator name
    estimator_name = run_name.split("_")[0]
    return run_name, estimator_name


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_data_file: Path = PROCESSED_DATA_DIR_INFER / "X.parquet",
    predictions_path: Path = "predictions.csv",
    outcome: str = "outcome",
    metric_name: str = "valid AUC ROC",  # Metric to select the best model
    mode: str = "max",  # max for metrics where higher is better, min otherwise
    # -----------------------------------------
):

    ################################################################################
    # STEP 3: Load Input Data for Prediction
    ################################################################################
    # Read the processed feature data from a parquet file and select relevant columns

    # Read the input data file
    X = pd.read_parquet(input_data_file)

    ################################################################################
    # STEP 4: Perform Inference Using the Best Model
    ################################################################################

    # Load and use the best model from MLflow for predictions
    # --- LOADING THE BEST MODEL- INFERENCE ---
    logger.info("Loading best model for inference ...")

    experiment_name = f"{outcome}_model"

    # Retrieve the best model's run name and estimator name
    run_name, estimator_name = find_best_model(
        experiment_name,
        metric_name,
        mode,
    )

    # Retrieve model name to use in loading best model
    model_name = f"{estimator_name}_{outcome}"

    # Load best model
    best_model = mlflow_load_model(
        experiment_name,
        run_name,
        model_name,
    )

    print(f"The best model is {best_model.name}.")

    # Generate predictions and probabilities
    X["predictions"] = best_model.predict(X, optimal_threshold=True)
    X["predicted_probas"] = best_model.predict_proba(X)[:, 1]
    logger.success("Inference complete.")
    # -----------------------------------------

    ################################################################################
    # STEP 5: Save Predictions to File
    ################################################################################

    X[["predictions", "predicted_probas"]].to_csv(predictions_path)


if __name__ == "__main__":
    app()
