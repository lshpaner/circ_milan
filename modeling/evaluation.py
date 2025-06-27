from pathlib import Path
import typer
from loguru import logger
import pandas as pd

# Import supportive care functions and constants
from functions import (
    mlflow_load_model,
    return_model_metrics,
    return_model_plots,
    log_mlflow_metrics,
    mlflow_log_parameters_model,
)

from config import (
    PROCESSED_DATA_DIR,
    model_definitions,
)

app = typer.Typer()

################################################################################
# ---- STEP 1: Define command-line arguments with default values ----
################################################################################


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    model_type: str = "lr",
    pipeline_type: str = "orig",
    outcome: str = "default_outcome",
    features_path: Path = PROCESSED_DATA_DIR / "X.parquet",
    labels_path: Path = PROCESSED_DATA_DIR / "y_income.parquet",
    scoring: str = "average_precision",
    # -----------------------------------------
):

    ################################################################################
    # STEP 2: Load Model Configuration & Pipeline Settings
    ################################################################################

    estimator_name = model_definitions[model_type]["estimator_name"]

    print(f"{estimator_name}_{pipeline_type}_training")
    print(f"{estimator_name}_{outcome}")

    ################################################################################
    # STEP 3: Load Pre-Trained Model from MLflow
    ################################################################################

    model = mlflow_load_model(
        experiment_name=f"{outcome}_model",
        run_name=f"{estimator_name}_{pipeline_type}_training",
        model_name=f"{estimator_name}_{outcome}",
    )

    # Print model threshold before optimization
    print(f"Model Threshold Before Threshold Optimization: {model.threshold}")

    ################################################################################
    # STEP 4: Load Processed Data (Features & Labels)
    ################################################################################

    X = pd.read_parquet(features_path)
    y = pd.read_parquet(labels_path)
    y = y.squeeze()  # coerce into a series

    ################################################################################
    # STEP 5: Log Updated Model with Optimized Threshold
    ################################################################################

    mlflow_log_parameters_model(
        experiment_name=f"{outcome}_model",
        run_name=f"{estimator_name}_{pipeline_type}_training",
        model_name=f"{estimator_name}_{outcome}",
        model=model,
    )

    # Print model threshold after optimization
    print(f"Model Threshold After Threshold Optimization: {model.threshold}")

    ################################################################################
    # STEP 6: Compute and Evaluate Model Performance Metrics
    ################################################################################

    all_inputs = {"K-Fold": (X, y)}
    metrics = return_model_metrics(
        inputs=all_inputs,
        model=model,
        estimator_name=estimator_name,
        return_dict=True,
    )

    print(metrics)

    ################################################################################
    # STEP 7: Generate and Save Model Evaluation Plots
    ################################################################################

    # Generate evaluation plots
    all_plots = return_model_plots(
        inputs=all_inputs,
        model=model,
        estimator_name=estimator_name,
        scoring=scoring,
    )

    ################################################################################
    # STEP 8: Log Experiment Details to MLflow
    ################################################################################

    log_mlflow_metrics(
        experiment_name=f"{outcome}_model",
        run_name=f"{estimator_name}_{pipeline_type}_training",
        metrics=metrics[estimator_name],
        images=all_plots,
    )

    ################################################################################
    # STEP 9: Completion Message
    ################################################################################

    logger.success("Modeling evaluation complete.")
    # -----------------------------------------


if __name__ == "__main__":

    app()
