################################################################################
######################### Import Requisite Libraries ###########################
################################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap
import sys
import os
import re
import mlflow
import textwrap
import warnings
import shap
import model_tuner

from tqdm import tqdm  # to show progress bar during model training and tuning

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    average_precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    brier_score_loss,
)

from sklearn.calibration import calibration_curve

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC


################################################################################
############################## Data Conversions ################################
################################################################################


# Function to count comorbidities
def count_comorbidities(row):
    """
    Counts the number of comorbidities in a given string.

    Parameters:
        row (str): A string of comma-separated comorbidities or a placeholder
        for none.

    Returns:
        int: Count of comorbidities. Returns 0 if no comorbidities are present.
    """

    if row == "0" or row == "None Present" or row.strip() == "":
        return 0
    return len(row.split(", "))


class HealthMetrics:
    """
    A class to calculate health-related metrics such as Body Mass Index (BMI)
    and Mean Arterial Pressure (MAP) for individuals in a DataFrame.
    """

    def __init__(self):
        pass  # Not storing state in the instance.

    @staticmethod
    def calculate_bmi_category(
        df,
        bmi_col="BMI",
        bmi_category_col="BMI_Category",
    ):
        """
        Categorize BMI based on the WHO BMI classifications and add the
        categories as a new column.

        Parameters:
            df (DataFrame): DataFrame containing the BMI data.
            bmi_col (str): Column name containing BMI values. Default is "BMI".
            bmi_category_col (str): Column name for BMI categories.
            Default is "BMI_Category".
        """

        def classify_bmi(bmi):
            if bmi < 18.5:
                return "Underweight"
            elif 18.5 <= bmi < 24.9:
                return "Normal weight"
            elif 25 <= bmi < 29.9:
                return "Overweight"
            else:
                return "Obese"

        # Apply the classification function to the BMI column
        df[bmi_category_col] = df[bmi_col].apply(classify_bmi)

    @staticmethod
    def calculate_bmi(
        df,
        weight_col,
        height_col,
        weight_unit="kg",
        height_unit="m",
    ):
        """
        Calculate Body Mass Index (BMI) and update the DataFrame with a new
        BMI column.

        Parameters:
            df (DataFrame): DataFrame containing the data.
            weight_col (str): Column name for weight.
            height_col (str): Column name for height.
            weight_unit (str): Unit of weight, default is "kg".
            height_unit (str): Unit of height, default is "m".
        """
        # Ensure weight is in kilograms
        if weight_unit == "lbs":
            df[weight_col] = df[weight_col] * 0.45359237
        # Ensure height is in meters
        if height_unit == "in":
            df[height_col] = df[height_col] * 0.0254
        elif height_unit == "cm":
            df[height_col] = df[height_col] / 100

        # Calculate BMI and update the DataFrame
        df["BMI"] = round(df[weight_col] / (df[height_col] ** 2), 2)

    @staticmethod
    def calculate_map(
        df,
        map_col_name="MAP",
        systolic_col=None,
        diastolic_col=None,
        combined_bp_col=None,
    ):
        """
        Calculate Mean Arterial Pressure (MAP) and update the DataFrame with a
        new MAP column.

        This method can operate based on separate systolic and diastolic columns,
        or a single combined column in the "Systolic/Diastolic" format. At
        least one of the column parameter sets must be provided.

        Parameters:
            df (DataFrame): The pandas DataFrame to calculate MAP for.
            map_col_name (str): Column name where MAP values will be stored.
            systolic_col (str, optional): Column name for systolic blood pressure.
            diastolic_col (str, optional): Column name for diastolic blood pressure.
            combined_bp_col (str, optional): Column name for combined
            "Systolic/Diastolic" blood pressure readings.
        """
        if systolic_col and diastolic_col:
            # Calculate MAP using separate systolic and diastolic columns
            systolic = df[systolic_col]
            diastolic = df[diastolic_col]
        elif combined_bp_col:
            # Calculate MAP using a combined BP column
            split_bp = df[combined_bp_col].str.split("/", expand=True).astype(float)
            systolic, diastolic = split_bp[0], split_bp[1]
        else:
            raise ValueError(
                "Must provide either systolic_col and diastolic_col, or combined_bp_col"
            )

        # MAP calculation formula: diastolic + (systolic - diastolic) / 3
        df[map_col_name] = round(diastolic + (systolic - diastolic) / 3, 2)


################################################################################
################### Modeling Fit, Predict, and Evaluation ######################
################################################################################


def metrics_report(
    df=None,
    outcome_cols=None,
    pred_cols=None,
    models=None,
    X_valid=None,
    y_valid=None,
    pred_probs_df=None,
):
    """
    Generate a DataFrame of model metrics for given models or predictions.

    Parameters:
    df (DataFrame, optional): DataFrame containing outcome and prediction cols.
    outcome_cols (list, optional): List of outcome column names in df.
    pred_cols (list, optional): List of prediction column names in df.
    models (dict, optional): Dict where key is model name and value is model.
    X_valid (DataFrame, optional): DataFrame with validation data.
    y_valid (Series, optional): Series with outcome data for validation set.
    pred_probs_df (DataFrame, optional): DataFrame with predicted probabilities.

    Returns:
    metrics_df (DataFrame): DataFrame containing model metrics.
    """
    metrics = {}

    # Calculate metrics for each outcome_col-pred_col pair in df
    if outcome_cols is not None and pred_cols is not None and df is not None:
        for outcome_col, pred_col in zip(outcome_cols, pred_cols):
            y_true = df[outcome_col]
            y_pred_proba = df[pred_col]
            y_pred = [1 if prob > 0.5 else 0 for prob in y_pred_proba]
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            roc_auc = roc_auc_score(y_true, y_pred_proba)
            brier_score = brier_score_loss(y_true, y_pred_proba)
            avg_precision = average_precision_score(y_true, y_pred_proba)
            specificity = tn / (tn + fp)
            metrics[pred_col] = {
                "Precision/PPV": precision,
                "Average Precision": avg_precision,
                "Sensitivity": recall,
                "Specificity": specificity,
                "AUC ROC": roc_auc,
                "Brier Score": brier_score,
            }

    if models is not None and X_valid is not None and y_valid is not None:
        for name, model in models.items():
            y_pred = model.predict(X_valid)
            # Attempt to use the predict_proba method if available
            try:
                y_pred_proba = model.predict_proba(X_valid)[:, 1]
            except AttributeError:
                # Fallback for models without predict_proba method
                y_pred_proba = model.predict(X_valid)

            tn, fp, fn, tp = confusion_matrix(y_valid, y_pred).ravel()
            precision = precision_score(y_valid, y_pred)
            recall = recall_score(y_valid, y_pred)
            roc_auc = roc_auc_score(y_valid, y_pred_proba)
            brier_score = brier_score_loss(y_valid, y_pred_proba)
            avg_precision = average_precision_score(y_valid, y_pred_proba)
            specificity = tn / (tn + fp)
            metrics[name] = {
                "Precision/PPV": precision,
                "Average Precision": avg_precision,
                "Sensitivity": recall,
                "Specificity": specificity,
                "AUC ROC": roc_auc,
                "Brier Score": brier_score,
            }

    # Calculate metrics for each column in pred_probs_df
    if pred_probs_df is not None:
        for col in pred_probs_df.columns:
            y_pred_proba = pred_probs_df[col]
            y_pred = [1 if prob > 0.5 else 0 for prob in y_pred_proba]
            tn, fp, fn, tp = confusion_matrix(y_valid, y_pred).ravel()
            precision = precision_score(y_valid, y_pred)
            recall = recall_score(y_valid, y_pred)
            roc_auc = roc_auc_score(y_valid, y_pred_proba)
            brier_score = brier_score_loss(y_valid, y_pred_proba)
            avg_precision = average_precision_score(y_valid, y_pred_proba)
            specificity = tn / (tn + fp)
            metrics[col] = {
                "Precision/PPV": precision,
                "Average Precision": avg_precision,
                "Sensitivity": recall,
                "Specificity": specificity,
                "AUC ROC": roc_auc,
                "Brier Score": brier_score,
            }

    metrics_df = pd.DataFrame(metrics).round(3)
    metrics_df["Mean"] = metrics_df.mean(axis=1).round(3)
    return metrics_df


################################################################################
######################### Create Custom Stratification #########################
################################################################################


# def create_stratified_other_column(
#     X,
#     stratify_list,
#     other_columns=None,
#     other_column=None,
#     patient_id=var_index,
#     age=None,
#     age_bin=None,
#     bin_ages=None,
# ):
#     """
#     Create a dataframe with a combined 'Other' column and optionally stratify
#     based on the specified columns and age bins.

#     Parameters:
#     -----------
#     X : pd.DataFrame
#         The original dataframe containing patient data and binary race/ethnicity
#         columns.

#     stratify_list : list of str
#         List of column names in X to use for stratification.

#     other_columns : list of str, optional
#         List of column names in X to be combined into the specified 'Other'
#         column using bitwise OR operation. If None or empty, the 'Other' column
#         will not be modified.

#     other_column : str, optional (default=None)
#         Name of the column in X that will be used to store the combined result.
#         If None, no 'Other' column will be created.

#     patient_id : str, optional (default='patient_id')
#         Name of the column in X that represents the unique identifier for
#         joining the data.

#     age : str, optional (default=None)
#         The name of the column in X that contains age data.

#     age_bin : str, optional (default=None)
#         The name of the column that will store the age bins.

#     bin_ages : list of int, optional (default=None)
#         The bin edges to be used for age stratification. If None, age
#         stratification will not be performed.

#     Returns:
#     --------
#     pd.DataFrame
#         A dataframe with the specified stratification columns and the combined
#         'Other' column if applicable, and age bins if specified.
#     """
#     # Create a copy of the dataframe
#     X_copy = X.copy()

#     # If other_columns and other_column are provided, perform the bitwise OR operation
#     if other_columns and other_column:
#         # Ensure that the specified 'Other' column is initially binary (0 or 1)
#         X_copy[other_column] = X_copy[other_column].astype(int)

#         # Combine the specified columns into the 'Other' column using bitwise OR
#         for column in other_columns:
#             X_copy[other_column] |= X_copy[column]

#     # If age stratification is needed
#     if age and age_bin and bin_ages:
#         # Stratify based on age using the provided bins
#         stratify_df = (
#             pd.cut(
#                 X_copy[age].fillna(
#                     X_copy[age].mean()
#                 ),  # Fill missing ages with the mean
#                 bins=bin_ages,  # Use the provided bin_ages
#                 right=False,  # Bin intervals are left-inclusive
#                 include_lowest=True,  # Include the lowest value in the first bin
#             )
#             .astype(str)
#             .to_frame(age_bin)
#         )
#     else:
#         # If no age stratification is needed, create an empty dataframe
#         stratify_df = pd.DataFrame(index=X_copy.index)

#     # Join the specified stratification columns to the stratify_df
#     if stratify_list:
#         stratify_df = stratify_df.join(
#             X_copy[stratify_list],
#             how="inner",
#             on=patient_id,
#         )

#     # If 'other_column' exists, join it as well
#     if other_column and other_column in X_copy.columns:
#         stratify_df = stratify_df.join(
#             X_copy[[other_column]],
#             how="inner",
#             on=patient_id,
#         )

#     return stratify_df


################################################################################
###########################  Model-Specific Curves ############################
################################################################################


class PlotMetrics:
    def __init__(self, images_path=None):
        """
        Initialize the PlotMetrics class.

        Parameters:
        -----------
        images_path : str, optional
            Path to save the generated plots.
        """
        self.images_path = images_path

    def _save_plot(self, title, extension="png"):
        """
        Save the plot to the specified path if images_path is provided.

        Parameters:
        -----------
        title : str
            Title of the plot.
        extension : str, optional (default="png")
            File extension for the saved plot.
        """
        if self.images_path:
            filename = f"{title.replace(' ', '_').replace(':', '')}.{extension}"
            plt.savefig(os.path.join(self.images_path, filename), format=extension)

    def plot_roc(
        self,
        df=None,
        outcome_cols=None,
        pred_cols=None,
        models=None,
        X_valid=None,
        y_valid=None,
        pred_probs_df=None,
        model_name=None,
        custom_name=None,
        show=True,
    ):
        fig, ax = plt.subplots(figsize=(8, 8))
        title = None

        if outcome_cols and pred_cols and df is not None:
            for outcome_col, pred_col in zip(outcome_cols, pred_cols):
                y_prob = df[pred_col]
                fpr, tpr, _ = roc_curve(df[outcome_col], y_prob)
                auc_score = roc_auc_score(df[outcome_col], y_prob)
                plt.plot(fpr, tpr, label=f"{outcome_col} (AUC={auc_score:.2f})")
                num_var = re.findall(r"\d+", pred_col)[0] if "var" in pred_col else None
                title = f"AUC ROC: {num_var}-Variable KFRE"

        if models and X_valid is not None and y_valid is not None:
            if model_name:
                y_score = models[model_name].predict_proba(X_valid)[:, 1]
                fpr, tpr, _ = roc_curve(y_valid, y_score)
                auc_score = roc_auc_score(y_valid, y_score)
                plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc_score:.2f})")
            else:
                for name, model in models.items():
                    y_score = model.predict_proba(X_valid)[:, 1]
                    fpr, tpr, _ = roc_curve(y_valid, y_score)
                    auc_score = roc_auc_score(y_valid, y_score)
                    plt.plot(fpr, tpr, label=f"{name} (AUC={auc_score:.2f})")

        if pred_probs_df is not None:
            for col in pred_probs_df.columns:
                y_score = pred_probs_df[col].values
                fpr, tpr, _ = roc_curve(y_valid, y_score)
                auc_score = roc_auc_score(y_valid, y_score)
                plt.plot(fpr, tpr, label=f"{col} (AUC={auc_score:.2f})")

        plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")

        if title is None:
            title = "Receiver Operating Characteristic"

        plt.title(title)
        self._save_plot(title)
        if show:
            plt.show()

        return fig

    def plot_precision_recall(
        self,
        df=None,
        outcome_cols=None,
        pred_cols=None,
        models=None,
        X_valid=None,
        y_valid=None,
        pred_probs_df=None,
        model_name=None,
        custom_name=None,
        show=True,
    ):
        fig, ax = plt.subplots(figsize=(8, 8))
        title = None

        if outcome_cols and pred_cols and df is not None:
            for outcome_col, pred_col in zip(outcome_cols, pred_cols):
                y_prob = df[pred_col]
                precision, recall, _ = precision_recall_curve(df[outcome_col], y_prob)
                avg_precision = average_precision_score(df[outcome_col], y_prob)
                plt.plot(
                    recall, precision, label=f"{outcome_col} (AP={avg_precision:.2f})"
                )

        if models and X_valid is not None and y_valid is not None:
            if model_name:
                y_score = models[model_name].predict_proba(X_valid)[:, 1]
                precision, recall, _ = precision_recall_curve(y_valid, y_score)
                avg_precision = average_precision_score(y_valid, y_score)
                plt.plot(
                    recall, precision, label=f"{model_name} (AP={avg_precision:.2f})"
                )
            else:
                for name, model in models.items():
                    y_score = model.predict_proba(X_valid)[:, 1]
                    precision, recall, _ = precision_recall_curve(y_valid, y_score)
                    avg_precision = average_precision_score(y_valid, y_score)
                    plt.plot(
                        recall, precision, label=f"{name} (AP={avg_precision:.2f})"
                    )

        if pred_probs_df is not None:
            for col in pred_probs_df.columns:
                y_score = pred_probs_df[col].values
                precision, recall, _ = precision_recall_curve(y_valid, y_score)
                avg_precision = average_precision_score(y_valid, y_score)
                plt.plot(recall, precision, label=f"{col} (AP={avg_precision:.2f})")

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend(loc="lower left")

        if title is None:
            title = "Precision-Recall Curve"

        plt.title(title)
        self._save_plot(title)
        if show:
            plt.show()

        return fig

    def plot_confusion_matrix(
        self,
        df=None,
        outcome_cols=None,
        pred_cols=None,
        models=None,
        X_valid=None,
        y_valid=None,
        threshold=0.5,
        custom_name=None,
        model_name=None,
        normalize=None,
        cmap="Blues",
        show=True,
    ):
        fig, ax = plt.subplots(figsize=(8, 8))
        title = None

        if outcome_cols and pred_cols and df is not None:
            for outcome_col, pred_col in zip(outcome_cols, pred_cols):
                y_true = df[outcome_col]
                y_pred = (df[pred_col] >= threshold).astype(int)
                cm = confusion_matrix(y_true, y_pred, normalize=normalize)
                disp = ConfusionMatrixDisplay(
                    confusion_matrix=cm, display_labels=[0, 1]
                )
                disp.plot(ax=ax, cmap=cmap, colorbar=False)

        if models and X_valid is not None and y_valid is not None:
            if model_name:
                y_pred = (
                    models[model_name].predict_proba(X_valid)[:, 1] >= threshold
                ).astype(int)
                cm = confusion_matrix(y_valid, y_pred, normalize=normalize)
                disp = ConfusionMatrixDisplay(
                    confusion_matrix=cm, display_labels=[0, 1]
                )
                disp.plot(ax=ax, cmap=cmap, colorbar=False)
            else:
                for name, model in models.items():
                    y_pred = (model.predict_proba(X_valid)[:, 1] >= threshold).astype(
                        int
                    )
                    cm = confusion_matrix(y_valid, y_pred, normalize=normalize)
                    disp = ConfusionMatrixDisplay(
                        confusion_matrix=cm, display_labels=[0, 1]
                    )
                    disp.plot(ax=ax, cmap=cmap, colorbar=False)

        if title is None:
            title = (
                f"{custom_name} - Confusion Matrix"
                if custom_name
                else "Confusion Matrix"
            )

        plt.title(title)
        self._save_plot(title)
        if show:
            plt.show()

        return fig


################################################################################
################################## MlFlow ######################################
################################################################################


def log_mlflow_experiment(
    mlflow_data,
    experiment_name,
    model_name,
    best_params,
    metrics,
    images={},
):
    """
    Logs ML experiments, including metrics and parameters, to MLflow.

    This function sets up an MLflow experiment with a given name, logs various
    metrics, parameters, and prediction results for a list of models, and
    handles both existing and new experiments. It also supports logging
    additional results for a specific model.

    Parameters:
    - mlflow_data: The MLflow tracking URI or a location where MLflow tracking
      server is running.
    - experiment_name: The name of the experiment to log in MLflow.
    - predictions_list: list of DataFrames containing predictions from ea. model.
    - model_names: A list of names of the models corresponding to the predictions.
    - best_params: A dictionary or a structure containing the best parameters of
                   the models.
    - results: A DataFrame containing metrics and other results for each model.
    - results_ak (optional): Additional results to be logged for a specific
                             model (model_name_ak).

    It includes two nested helper functions:
    - sanitize_metric_name(name): Sanitizes metric names by replacing
                                  non-allowed characters with underscores.
    - is_numeric(value): Checks if a given value is numeric.

    The function handles the creation of new experiments or the retrieval of
    existing ones, logs metrics, parameters, and prediction results as CSV
    artifacts, and also ensures the cleanliness of the local file system by
    removing temporary files.

    Note:
    - It is assumed that 'model_name_ak' is a global variable or is defined
      elsewhere in the context where this function is used, as it is not
      explicitly passed to the function.
    - The function depends on the MLflow library and its proper configuration.

    Returns:
    None
    """

    def sanitize_metric_name(name):
        """Replace any characters not allowed in MLflow metric names with an
        underscore.
        """
        return "".join(
            c if c.isalnum() or c in ["_", "-", ".", " ", "/"] else "_" for c in name
        )

    def is_numeric(value):
        """Check if the value is numeric."""
        try:
            float(value)
            return True
        except ValueError:
            return False

    # Set the tracking URI to the specified mlflow_data location
    mlflow.set_tracking_uri(mlflow_data)

    # Check if the experiment already exists
    existing_experiment = mlflow.get_experiment_by_name(experiment_name)

    # If the experiment doesn't exist, create a new experiment
    if existing_experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"New Experiment_ID: {experiment_id}")
    else:
        # If the experiment already exists, retrieve its ID
        experiment_id = existing_experiment.experiment_id
        print(f"Existing Experiment_ID: {experiment_id}")

    # Iterate over the models and log their metrics and parameters
    with mlflow.start_run(experiment_id=experiment_id, run_name=model_name):

        # log model's best parameters
        if isinstance(best_params, dict):
            mlflow.log_params(best_params)
        else:
            mlflow.log_param("best_params_unavailable", str(best_params))

        # Extract the row for the current model
        result = metrics
        if not result.empty:

            # Log the parameters and metrics
            for col in result.index:
                # Sanitize the column name
                sanitized_name = sanitize_metric_name(col)
                value = result[col]

                # Check if the value is numeric and not null
                if is_numeric(value) and pd.notnull(value):
                    # Log the metric
                    mlflow.log_metric(sanitized_name, float(value))

            # log model's best parameters
            if isinstance(best_params, dict):
                mlflow.log_params(best_params)
            else:
                mlflow.log_param("best_params_unavailable", str(best_params))

            for name, image in images.items():
                mlflow.log_figure(image, name)


################################################################################
################################################################################
################################################################################


class ModelEvaluationMetrics:
    def __init__(self):
        """
        Initialize the ModelEvaluationPlots class.
        """
        pass

    def _save_plot(self, filename, save_plot, image_path_png, image_path_svg):
        """
        Save the plot to specified directories.
        """
        if save_plot:
            if not (image_path_png or image_path_svg):
                raise ValueError(
                    "save_plot is set to True, but no image path is provided. "
                    "Please specify at least one of `image_path_png` or `image_path_svg`."
                )
            if image_path_png:
                os.makedirs(image_path_png, exist_ok=True)
                plt.savefig(
                    os.path.join(image_path_png, f"{filename}.png"),
                    bbox_inches="tight",
                )
            if image_path_svg:
                os.makedirs(image_path_svg, exist_ok=True)
                plt.savefig(
                    os.path.join(image_path_svg, f"{filename}.svg"),
                    bbox_inches="tight",
                )

    def get_predictions(self, model, X, y, model_threshold, custom_threshold, score):

        test_model = model.test_model
        threshold = 0.5
        if custom_threshold:
            threshold = custom_threshold
        else:
            if model_threshold:
                if score is not None:
                    threshold = model.threshold[score]
                else:
                    threshold = model.threshold[model.scoring[0]]

        if hasattr(model, "kfold") and model.kfold:  # Handle k-fold logic
            print("\nRunning k-fold model metrics...\n")
            aggregated_y_true = []
            aggregated_y_pred = []
            aggregated_y_prob = []

            for fold_idx, (train, test) in tqdm(
                enumerate(model.kf.split(X, y), start=1),
                total=model.kf.get_n_splits(),
                desc="Processing Folds",
            ):
                X_train, X_test = X.iloc[train], X.iloc[test]
                y_train, y_test = y.iloc[train], y.iloc[test]

                # Fit and predict for this fold
                test_model.fit(X_train, y_train.values.ravel())

                y_pred_proba = test_model.predict_proba(X_test)[:, 1]
                y_pred = 1 * (y_pred_proba > threshold)
                aggregated_y_true.extend(y_test.values.tolist())
                aggregated_y_pred.extend(y_pred.tolist())
                aggregated_y_prob.extend(y_pred_proba.tolist())
        else:
            aggregated_y_true = y
            aggregated_y_prob = test_model.predict_proba(X)[:, 1]
            aggregated_y_pred = 1 * (aggregated_y_prob > threshold)
        return aggregated_y_true, aggregated_y_prob, aggregated_y_pred, threshold

    def summarize_model_performance(
        self,
        pipelines_or_models,
        X,
        y,
        model_threshold=None,
        model_titles=None,
        custom_threshold=None,
        score=None,
        return_df=False,
    ):
        """
        Summarize key performance metrics for multiple models.

        Parameters:
        - pipelines_or_models: list
            A list of models or pipelines to evaluate.
            Each pipeline should either end with a classifier or contain one.
        - X: array-like
            The input features for generating predictions.
        - y_true: array-like
            The true labels corresponding to the input features.
        - model_threshold: dict or None, optional
            A dictionary mapping model names to predefined thresholds for binary
            classification. If provided, these thresholds will be displayed in
            the table but not used for metric recalculations when `custom_threshold`
            is set.
        - model_titles: list or None, optional
            A list of custom titles for individual models. If not provided, the
            names of the models will be extracted automatically.
        - custom_threshold: float or None, optional
            A custom threshold to apply for recalculating metrics. If set, this
            threshold will override the default threshold of 0.5 and any thresholds
            from `model_threshold` for all models.
            When specified, the "Model Threshold" row is omitted from the table.
        - return_df: bool, optional
            Whether to return the metrics as a pandas DataFrame instead of printing
            them to the console. Default is False.

        Returns:
        - pd.DataFrame or None
            If `return_df` is True, returns a DataFrame summarizing model performance
            metrics, including precision, recall, specificity, F1-Score, AUC ROC,
            and Brier Score. Otherwise, prints the metrics in a formatted table.

        Notes:
        - If `model_threshold` is provided and `custom_threshold` is not set, the
        "Model Threshold" row will display the values from `model_threshold`.
        - If `custom_threshold` is set, it applies to all models for metric
        recalculations, and the "Model Threshold" row is excluded from the table.
        - Automatically extracts model names if `model_titles` is not provided.
        - Models must support `predict_proba` or `decision_function` for predictions.
        """

        if not isinstance(pipelines_or_models, list):
            pipelines_or_models = [pipelines_or_models]

        metrics_data = []

        for i, model in enumerate(pipelines_or_models):
            # Determine the model name
            if model_titles:
                name = model_titles[i]
            else:
                name = self._extract_model_name(model)  # Extract detailed name

            y_true, y_prob, y_pred, threshold = self.get_predictions(
                model, X, y, model_threshold, custom_threshold, score
            )

            # Compute metrics
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)  # Sensitivity
            specificity = recall_score(y_true, y_pred, pos_label=0)
            auc_roc = roc_auc_score(y_true, y_prob)
            brier = brier_score_loss(y_true, y_prob)
            avg_precision = average_precision_score(y_true, y_prob)
            f1 = f1_score(y_true, y_pred)

            # Append metrics for this model
            model_metrics = {
                "Model": name,
                "Precision/PPV": precision,
                "Average Precision": avg_precision,
                "Sensitivity/Recall": recall,
                "Specificity": specificity,
                "F1-Score": f1,
                "AUC ROC": auc_roc,
                "Brier Score": brier,
                "Model Threshold": threshold,
            }
            metrics_data.append(model_metrics)

        # Create a DataFrame
        metrics_df = pd.DataFrame(metrics_data)
        metrics_df.set_index("Model", inplace=True)
        metrics_df = metrics_df.T

        # Return the DataFrame if requested
        if return_df:
            return metrics_df

        # Adjust column widths for center alignment
        col_widths = {col: max(len(col), 8) + 2 for col in metrics_df.columns}
        row_name_width = max(len(row) for row in metrics_df.index) + 2

        # Center-align headers
        headers = [
            f"{'Metric'.center(row_name_width)}"
            + "".join(f"{col.center(col_widths[col])}" for col in metrics_df.columns)
        ]

        # Separator line
        separator = "-" * (row_name_width + sum(col_widths.values()))

        # Print table header
        print("Model Performance Metrics:")
        print("\n".join(headers))
        print(separator)

        # Center-align rows
        for row_name, row_data in metrics_df.iterrows():
            row = f"{row_name.center(row_name_width)}" + "".join(
                (
                    f"{f'{value:.4f}'.center(col_widths[col])}"
                    if isinstance(value, float)
                    else f"{str(value).center(col_widths[col])}"
                )
                for col, value in zip(metrics_df.columns, row_data)
            )
            print(row)

    def _get_model_probabilities(self, model, X, name):
        """
        Extract probabilities for the positive class from the model.
        """
        if hasattr(model, "predict_proba"):  # Direct model with predict_proba
            return model.predict_proba(X)[:, 1]
        elif hasattr(model, "named_steps"):  # Pipeline
            final_model = list(model.named_steps.values())[-1]
            if hasattr(final_model, "predict_proba"):
                return model.predict_proba(X)[:, 1]
            elif hasattr(final_model, "decision_function"):
                y_scores = final_model.decision_function(X)
                return 1 / (1 + np.exp(-y_scores))  # Convert to probabilities
        elif hasattr(
            model, "decision_function"
        ):  # Standalone model with decision_function
            y_scores = model.decision_function(X)
            return 1 / (1 + np.exp(-y_scores))  # Convert to probabilities
        else:
            raise ValueError(
                f"Model {name} does not support probability-based prediction."
            )

    def _extract_model_titles(self, models_or_pipelines, model_titles=None):
        """
        Extract titles from models or pipelines using an optional external list of model titles.

        Parameters:
        -----------
        models_or_pipelines : list
            A list of model or pipeline objects.

        model_titles : list, optional
            A list of human-readable model titles to map class names to.

        Returns:
        --------
        list
            A list of extracted or matched model titles.
        """
        titles = []
        for model in models_or_pipelines:
            try:
                if hasattr(model, "named_steps"):  # Check if it's a pipeline
                    final_model = list(model.named_steps.values())[-1]
                    class_name = final_model.__class__.__name__
                else:
                    class_name = model.__class__.__name__

                # If model_titles is provided, attempt to map class_name
                if model_titles:
                    matched_title = next(
                        (title for title in model_titles if class_name in title),
                        class_name,
                    )
                else:
                    matched_title = (
                        class_name  # Default to class_name if no titles are provided
                    )

                titles.append(matched_title)
            except AttributeError:
                titles.append("Unknown Model")

        return titles

    # def _extract_model_titles(self, models_or_pipelines):
    #     """
    #     Extract titles from models or pipelines.
    #     """
    #     titles = []
    #     for model in models_or_pipelines:
    #         if hasattr(model, "named_steps"):  # Pipeline
    #             final_model = list(model.named_steps.values())
    #             title = getattr(final_model, "__class__", type(final_model)).__name__
    #         else:
    #             title = getattr(model, "__class__", type(model)).__name__
    #         titles.append(title)
    #     return titles

    # Helper function to extract detailed model names from pipelines or models
    def _extract_model_name(self, pipeline_or_model):
        if hasattr(pipeline_or_model, "steps"):  # It's a pipeline
            return pipeline_or_model.steps[-1][
                1
            ].__class__.__name__  # Final estimator's class name
        return pipeline_or_model.__class__.__name__  # Individual model class name

    def plot_confusion_matrix(
        self,
        pipelines_or_models,
        X,
        y,
        model_titles=None,
        model_threshold=None,
        custom_threshold=None,
        class_labels=None,
        cmap="Blues",
        save_plot=False,
        image_path_png=None,
        image_path_svg=None,
        text_wrap=None,
        figsize=(8, 6),
        labels=True,
        label_fontsize=12,
        tick_fontsize=10,
        inner_fontsize=10,
        grid=False,  # Added grid option
        score=None,
        **kwargs,
    ):
        """
        Compute and plot confusion matrices for multiple pipelines or models.

        Parameters:
        (Documentation remains unchanged...)

        Returns:
        - None
        """
        if not isinstance(pipelines_or_models, list):
            pipelines_or_models = [pipelines_or_models]

        if model_titles is None:
            model_titles = [
                self._extract_model_name(model) for model in pipelines_or_models
            ]

        if class_labels is None:
            class_labels = ["Class 0", "Class 1"]

        # Setup grid if enabled
        if grid:
            n_cols = kwargs.get("n_cols", 2)
            n_rows = (len(pipelines_or_models) + n_cols - 1) // n_cols
            fig, axes = plt.subplots(
                n_rows, n_cols, figsize=(figsize[0] * n_cols, figsize[1] * n_rows)
            )
            axes = axes.flatten()
        else:
            axes = [None] * len(pipelines_or_models)

        for idx, (model, ax) in enumerate(zip(pipelines_or_models, axes)):
            # Determine the model name
            if model_titles:
                name = model_titles[idx]
            else:
                name = self._extract_model_name(model)

            y_true, y_prob, y_pred, threshold = self.get_predictions(
                model, X, y, model_threshold, custom_threshold, score
            )

            # Compute confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            print(f"Confusion Matrix for {name}:")
            print(cm)

            # Plot the confusion matrix
            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm, display_labels=class_labels
            )
            if grid:
                disp.plot(cmap=cmap, ax=ax, colorbar=kwargs.get("show_colorbar", True))
            else:
                fig, ax = plt.subplots(figsize=figsize)
                disp.plot(cmap=cmap, ax=ax, colorbar=kwargs.get("show_colorbar", True))

            # Adjust title wrapping
            title = f"Confusion Matrix: {name} (Threshold = {threshold:.2f})"
            if text_wrap is not None and isinstance(text_wrap, int):
                title = "\n".join(textwrap.wrap(title, width=text_wrap))
            ax.set_title(title, fontsize=label_fontsize)

            # Adjust font sizes for axis labels and tick labels
            ax.xaxis.label.set_size(label_fontsize)
            ax.yaxis.label.set_size(label_fontsize)
            ax.tick_params(axis="both", labelsize=tick_fontsize)

            # Adjust the font size for the numeric values directly
            if disp.text_ is not None:
                for text in disp.text_.ravel():
                    text.set_fontsize(inner_fontsize)  # Apply inner_fontsize here

            # Add labels (TN, FP, FN, TP) only if `labels` is True
            if labels:
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        label_text = (
                            "TN"
                            if i == 0 and j == 0
                            else (
                                "FP"
                                if i == 0 and j == 1
                                else "FN" if i == 1 and j == 0 else "TP"
                            )
                        )
                        rgba_color = disp.im_.cmap(disp.im_.norm(cm[i, j]))
                        luminance = (
                            0.2126 * rgba_color[0]
                            + 0.7152 * rgba_color[1]
                            + 0.0722 * rgba_color[2]
                        )
                        ax.text(
                            j,
                            i - 0.3,  # Slight offset above numeric value
                            label_text,
                            ha="center",
                            va="center",
                            fontsize=inner_fontsize,
                            color="white" if luminance < 0.5 else "black",
                        )

            # Always display numeric values (confusion matrix counts)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    rgba_color = disp.im_.cmap(disp.im_.norm(cm[i, j]))
                    luminance = (
                        0.2126 * rgba_color[0]
                        + 0.7152 * rgba_color[1]
                        + 0.0722 * rgba_color[2]
                    )
                    ax.text(
                        j,
                        i,  # Exact position for numeric value
                        f"{cm[i, j]}",
                        ha="center",
                        va="center",
                        fontsize=inner_fontsize,
                        color="white" if luminance < 0.5 else "black",
                    )

            if not grid:
                self._save_plot(
                    f"Confusion_Matrix_{name}",
                    save_plot,
                    image_path_png,
                    image_path_svg,
                )
                plt.show()

        if grid:
            for ax in axes[len(pipelines_or_models) :]:
                ax.axis("off")
            plt.tight_layout()
            self._save_plot(
                "Grid_Confusion_Matrix", save_plot, image_path_png, image_path_svg
            )
            plt.show()

    def plot_roc_auc(
        self,
        pipelines_or_models,
        X,
        y,
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        model_titles=None,
        decimal_places=2,
        overlay=False,
        title=None,
        save_plot=False,
        image_path_png=None,
        image_path_svg=None,
        text_wrap=None,
        curve_kwgs=None,
        linestyle_kwgs=None,
        grid=False,  # Grid layout option
        n_cols=2,  # Number of columns for the grid
        figsize=None,  # User-defined figure size
        label_fontsize=12,  # Font size for title and axis labels
        tick_fontsize=10,  # Font size for tick labels and legend
        gridlines=True,
    ):
        """
        Plot ROC curves for models or pipelines with optional styling and grid layout.

        Parameters:
        - pipelines_or_models: list
            List of models or pipelines to plot.
        - X: array-like
            Features for prediction.
        - y: array-like
            True labels.
        - model_titles: list of str, optional
            Titles for individual models. Required when providing a nested dictionary for
            `curve_kwgs`.
        - overlay: bool
            Whether to overlay multiple models on a single plot.
        - title: str, optional
            Custom title for the plot when `overlay=True`.
        - save_plot: bool
            Whether to save the plot.
        - image_path_png: str, optional
            Path to save PNG images.
        - image_path_svg: str, optional
            Path to save SVG images.
        - text_wrap: int, optional
            Max width for wrapping titles.
        - curve_kwgs: list or dict, optional
            Styling for individual model curves. If `model_titles` is specified as a list
            of titles, `curve_kwgs` must be a nested dictionary with model titles as keys
            and their respective style dictionaries as values. Otherwise, `curve_kwgs`
            must be a list of style dictionaries corresponding to the models.
        - linestyle_kwgs: dict, optional
            Styling for the random guess diagonal line.
        - grid: bool, optional
            Whether to organize plots in a grid layout (default: False).
        - n_cols: int, optional
            Number of columns in the grid layout (default: 2).
        - figsize: tuple, optional
            Custom figure size (width, height) for the plot(s).
        - label_fontsize: int, optional
            Font size for title and axis labels.
        - tick_fontsize: int, optional
            Font size for tick labels and legend.

        Raises:
        - ValueError: If `grid=True` and `overlay=True` are both set.
        """
        if overlay and grid:
            raise ValueError("`grid` cannot be set to True when `overlay` is True.")

        if overlay and model_titles is not None:
            raise ValueError(
                "`model_titles` can only be provided when plotting models as "
                "separate plots (when `overlay=False`). If you want to specify "
                "a custom title for this plot, use the `title` input."
            )

        if not isinstance(pipelines_or_models, list):
            pipelines_or_models = [pipelines_or_models]

        if model_titles is None:
            model_titles = [
                self._extract_model_name(model) for model in pipelines_or_models
            ]

        if model_titles:
            name = model_titles
        else:
            name = self._extract_model_name(model)
        # if model_titles is None:
        #     model_titles = self._extract_model_titles(pipelines_or_models)

        if isinstance(curve_kwgs, dict):
            curve_styles = [curve_kwgs.get(name, {}) for name in model_titles]
        elif isinstance(curve_kwgs, list):
            curve_styles = curve_kwgs
        else:
            curve_styles = [{}] * len(pipelines_or_models)

        if len(curve_styles) != len(pipelines_or_models):
            raise ValueError(
                "The length of `curve_kwgs` must match the number of models."
            )

        if overlay:
            plt.figure(figsize=figsize or (8, 6))

        if grid and not overlay:
            import math

            n_rows = math.ceil(len(pipelines_or_models) / n_cols)
            fig, axes = plt.subplots(
                n_rows, n_cols, figsize=figsize or (n_cols * 6, n_rows * 4)
            )
            axes = axes.flatten()

        for idx, (model, name, curve_style) in enumerate(
            zip(pipelines_or_models, model_titles, curve_styles)
        ):
            y_true, y_prob, y_pred, threshold = self.get_predictions(
                model, X, y, None, None, None
            )

            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = roc_auc_score(y_true, y_prob)

            print(f"AUC for {name}: {roc_auc:.{decimal_places}f}")

            if overlay:
                plt.plot(
                    fpr,
                    tpr,
                    label=f"{name} (AUC = {roc_auc:.{decimal_places}f})",
                    **curve_style,
                )
            elif grid:
                ax = axes[idx]
                ax.plot(
                    fpr,
                    tpr,
                    label=f"ROC Curve (AUC = {roc_auc:.{decimal_places}f})",
                    **curve_style,
                )
                linestyle_kwgs = linestyle_kwgs or {}
                linestyle_kwgs.setdefault("color", "gray")
                linestyle_kwgs.setdefault("linestyle", "--")
                ax.plot(
                    [0, 1],
                    [0, 1],
                    label="Random Guess",
                    **linestyle_kwgs,
                )
                ax.set_xlabel(xlabel, fontsize=label_fontsize)
                ax.set_ylabel(ylabel, fontsize=label_fontsize)
                ax.tick_params(axis="both", labelsize=tick_fontsize)
                if text_wrap:
                    grid_title = "\n".join(
                        textwrap.wrap(f"ROC Curve: {name}", width=text_wrap)
                    )
                else:
                    grid_title = f"ROC Curve: {name}"
                if grid_title != "":
                    ax.set_title(grid_title, fontsize=label_fontsize)
                ax.legend(loc="lower right", fontsize=tick_fontsize)
                ax.grid(visible=gridlines)
            else:
                plt.figure(figsize=figsize or (8, 6))
                plt.plot(
                    fpr,
                    tpr,
                    label=f"ROC Curve (AUC = {roc_auc:.{decimal_places}f})",
                    **curve_style,
                )
                linestyle_kwgs = linestyle_kwgs or {}
                linestyle_kwgs.setdefault("color", "gray")
                linestyle_kwgs.setdefault("linestyle", "--")
                plt.plot(
                    [0, 1],
                    [0, 1],
                    label="Random Guess",
                    **linestyle_kwgs,
                )
                plt.xlabel(xlabel, fontsize=label_fontsize)
                plt.ylabel(ylabel, fontsize=label_fontsize)
                plt.tick_params(axis="both", labelsize=tick_fontsize)
                if text_wrap:
                    title = "\n".join(
                        textwrap.wrap(f"ROC Curve: {name}", width=text_wrap)
                    )
                else:
                    title = f"ROC Curve: {name}"
                if title != "":
                    plt.title(title, fontsize=label_fontsize)
                plt.legend(loc="lower right", fontsize=tick_fontsize)
                plt.grid()
                self._save_plot(
                    f"{name}_ROC", save_plot, image_path_png, image_path_svg
                )
                plt.show()

        if overlay:
            linestyle_kwgs = linestyle_kwgs or {}
            linestyle_kwgs.setdefault("color", "gray")
            linestyle_kwgs.setdefault("linestyle", "--")
            plt.plot(
                [0, 1],
                [0, 1],
                label="Random Guess",
                **linestyle_kwgs,
            )
            plt.xlabel(xlabel, fontsize=label_fontsize)
            plt.ylabel(ylabel, fontsize=label_fontsize)
            plt.tick_params(axis="both", labelsize=tick_fontsize)
            if text_wrap:
                title = "\n".join(
                    textwrap.wrap(title or "ROC Curves: Overlay", width=text_wrap)
                )
            else:
                title = title or "ROC Curves: Overlay"
            if title != "":
                plt.title(title, fontsize=label_fontsize)
            plt.legend(loc="lower right", fontsize=tick_fontsize)
            plt.grid(visible=gridlines)
            self._save_plot("Overlay_ROC", save_plot, image_path_png, image_path_svg)
            plt.show()
        elif grid:
            for ax in axes[len(pipelines_or_models) :]:
                ax.axis("off")
            plt.tight_layout()
            self._save_plot("Grid_ROC", save_plot, image_path_png, image_path_svg)
            plt.show()

    def plot_pr_auc(
        self,
        pipelines_or_models,
        X,
        y,
        xlabel="Recall",
        ylabel="Precision",
        model_titles=None,
        decimal_places=2,
        overlay=False,
        title=None,
        save_plot=False,
        image_path_png=None,
        image_path_svg=None,
        text_wrap=None,
        curve_kwgs=None,
        grid=False,  # Grid layout option
        n_cols=2,  # Number of columns for the grid
        figsize=None,  # User-defined figure size
        label_fontsize=12,  # Font size for title and axis labels
        tick_fontsize=10,  # Font size for tick labels and legend
        gridlines=True,
    ):
        """
        Plot PR curves for models or pipelines with optional styling and grid layout.

        Parameters:
        - pipelines_or_models: list
            List of models or pipelines to plot.
        - X: array-like
            Features for prediction.
        - y: array-like
            True labels.
        - model_titles: list of str, optional
            Titles for individual models.
        - overlay: bool
            Whether to overlay multiple models on a single plot.
        - title: str, optional
            Custom title for the plot.
        - save_plot: bool
            Whether to save the plot.
        - image_path_png: str, optional
            Path to save PNG images.
        - image_path_svg: str, optional
            Path to save SVG images.
        - text_wrap: int, optional
            Max width for wrapping titles.
        - curve_kwgs: list or dict, optional
            Styling for individual model curves.
        - grid: bool, optional
            Whether to organize plots in a grid layout (default: False).
        - n_cols: int, optional
            Number of columns in the grid layout (default: 2).
        - figsize: tuple, optional
            Custom figure size (width, height) for the plot(s).
        - label_fontsize: int, optional
            Font size for title and axis labels.
        - tick_fontsize: int, optional
            Font size for tick labels and legend.

        Raises:
        - ValueError: If `grid=True` and `overlay=True` are both set.
        """
        if overlay and grid:
            raise ValueError("`grid` cannot be set to True when `overlay` is True.")

        if overlay and model_titles is not None:
            raise ValueError(
                "`model_titles` can only be provided when plotting models as "
                "separate plots (when `overlay=False`). If you want to specify "
                "a custom title for this plot, use the `title` input."
            )

        if not isinstance(pipelines_or_models, list):
            pipelines_or_models = [pipelines_or_models]

        if model_titles is None:
            model_titles = self._extract_model_titles(pipelines_or_models)

        if isinstance(curve_kwgs, dict):
            curve_styles = [curve_kwgs.get(name, {}) for name in model_titles]
        elif isinstance(curve_kwgs, list):
            curve_styles = curve_kwgs
        else:
            curve_styles = [{}] * len(pipelines_or_models)

        if len(curve_styles) != len(pipelines_or_models):
            raise ValueError(
                "The length of `curve_kwgs` must match the number of models."
            )

        if overlay:
            plt.figure(figsize=figsize or (8, 6))  # Use figsize if provided

        if grid and not overlay:
            import math

            n_rows = math.ceil(len(pipelines_or_models) / n_cols)
            fig, axes = plt.subplots(
                n_rows, n_cols, figsize=figsize or (n_cols * 6, n_rows * 4)
            )
            axes = axes.flatten()  # Flatten axes for easy iteration

        for idx, (model, name, curve_style) in enumerate(
            zip(pipelines_or_models, model_titles, curve_styles)
        ):
            y_true, y_prob, y_pred, threshold = self.get_predictions(
                model, X, y, None, None, None
            )

            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            avg_precision = average_precision_score(y_true, y_prob)

            print(f"Average Precision for {name}: {avg_precision:.{decimal_places}f}")

            if overlay:
                plt.plot(
                    recall,
                    precision,
                    label=f"{name} (AP = {avg_precision:.{decimal_places}f})",
                    **curve_style,
                )
            elif grid:
                ax = axes[idx]
                ax.plot(
                    recall,
                    precision,
                    label=f"PR Curve (AP = {avg_precision:.{decimal_places}f})",
                    **curve_style,
                )
                ax.set_xlabel(xlabel, fontsize=label_fontsize)
                ax.set_ylabel(ylabel, fontsize=label_fontsize)
                ax.tick_params(axis="both", labelsize=tick_fontsize)
                if text_wrap:
                    grid_title = "\n".join(
                        textwrap.wrap(f"PR Curve: {name}", width=text_wrap)
                    )
                else:
                    grid_title = f"PR Curve: {name}"
                if grid_title != "":
                    ax.set_title(grid_title, fontsize=label_fontsize)
                ax.legend(loc="lower left", fontsize=tick_fontsize)
                ax.grid(visible=gridlines)
            else:
                plt.figure(figsize=figsize or (8, 6))  # Use figsize if provided
                plt.plot(
                    recall,
                    precision,
                    label=f"PR Curve (AP = {avg_precision:.{decimal_places}f})",
                    **curve_style,
                )
                plt.xlabel(xlabel, fontsize=label_fontsize)
                plt.ylabel(ylabel, fontsize=label_fontsize)
                plt.tick_params(axis="both", labelsize=tick_fontsize)
                if text_wrap:
                    title = "\n".join(
                        textwrap.wrap(f"PR Curve: {name}", width=text_wrap)
                    )
                else:
                    title = f"PR Curve: {name}"
                if title != "":
                    plt.title(title, fontsize=label_fontsize)
                plt.legend(loc="lower left", fontsize=tick_fontsize)
                plt.grid(visible=gridlines)
                self._save_plot(f"{name}_PR", save_plot, image_path_png, image_path_svg)
                plt.show()

        if overlay:
            plt.xlabel(xlabel, fontsize=label_fontsize)
            plt.ylabel(ylabel, fontsize=label_fontsize)
            plt.tick_params(axis="both", labelsize=tick_fontsize)
            if text_wrap:
                title = "\n".join(
                    textwrap.wrap(title or "PR Curves: Overlay", width=text_wrap)
                )
            else:
                title = title or "PR Curves: Overlay"
            if title != "":
                plt.title(title, fontsize=label_fontsize)
            plt.legend(loc="lower left", fontsize=tick_fontsize)
            plt.grid()
            self._save_plot("Overlay_PR", save_plot, image_path_png, image_path_svg)
            plt.show()
        elif grid:
            for ax in axes[len(pipelines_or_models) :]:
                ax.axis("off")
            plt.tight_layout()
            self._save_plot("Grid_PR", save_plot, image_path_png, image_path_svg)
            plt.show()

    def plot_calibration_curve(
        self,
        pipelines_or_models,
        X,
        y,
        xlabel="Mean Predicted Probability",
        ylabel="Fraction of Positives",
        model_titles=None,
        overlay=False,
        title=None,
        save_plot=False,
        image_path_png=None,
        image_path_svg=None,
        text_wrap=None,
        curve_kwgs=None,
        grid=False,  # Grid layout option
        n_cols=2,  # Number of columns for the grid
        figsize=None,  # User-defined figure size
        label_fontsize=12,
        tick_fontsize=10,
        bins=10,  # Number of bins for calibration curve
        marker="o",  # Marker style for the calibration points
        show_brier_score=True,
        gridlines=True,
        linestyle_kwgs=None,
        **kwargs,
    ):
        """
        Plot calibration curves for models or pipelines with optional styling and
        grid layout.

        Parameters:
        - pipelines_or_models: list
            List of models or pipelines to plot.
        - X: array-like
            Features for prediction.
        - y: array-like
            True labels.
        - model_titles: list of str, optional
            Titles for individual models.
        - overlay: bool
            Whether to overlay multiple models on a single plot.
        - title: str, optional
            Custom title for the plot when `overlay=True`.
        - save_plot: bool
            Whether to save the plot.
        - image_path_png: str, optional
            Path to save PNG images.
        - image_path_svg: str, optional
            Path to save SVG images.
        - text_wrap: int, optional
            Max width for wrapping titles.
        - curve_kwgs: list or dict, optional
            Styling for individual model curves.
        - grid: bool, optional
            Whether to organize plots in a grid layout (default: False).
        - n_cols: int, optional
            Number of columns in the grid layout (default: 2).
        - figsize: tuple, optional
            Custom figure size (width, height) for the plot(s).
        - label_fontsize: int, optional
            Font size for axis labels and title.
        - tick_fontsize: int, optional
            Font size for tick labels and legend.
        - bins: int, optional
            Number of bins for the calibration curve (default: 10).
        - marker: str, optional
            Marker style for calibration curve points (default: "o").

        Raises:
        - ValueError: If `grid=True` and `overlay=True` are both set.
        """
        if overlay and grid:
            raise ValueError("`grid` cannot be set to True when `overlay` is True.")

        if not isinstance(pipelines_or_models, list):
            pipelines_or_models = [pipelines_or_models]

        if model_titles is None:
            model_titles = self._extract_model_titles(pipelines_or_models)

        if isinstance(curve_kwgs, dict):
            curve_styles = [curve_kwgs.get(name, {}) for name in model_titles]
        elif isinstance(curve_kwgs, list):
            curve_styles = curve_kwgs
        else:
            curve_styles = [{}] * len(pipelines_or_models)

        if len(curve_styles) != len(pipelines_or_models):
            raise ValueError(
                "The length of `curve_kwgs` must match the number of models."
            )

        if overlay:
            plt.figure(figsize=figsize or (8, 6))

        if grid and not overlay:
            import math

            n_rows = math.ceil(len(pipelines_or_models) / n_cols)
            fig, axes = plt.subplots(
                n_rows, n_cols, figsize=figsize or (n_cols * 6, n_rows * 4)
            )
            axes = axes.flatten()

        for idx, (model, name, curve_style) in enumerate(
            zip(pipelines_or_models, model_titles, curve_styles)
        ):
            y_true, y_prob, y_pred, threshold = self.get_predictions(
                model, X, y, None, None, None
            )
            prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=bins)

            # Calculate Brier score if enabled
            brier_score = brier_score_loss(y_true, y_prob) if show_brier_score else None

            legend_label = f"{name}"
            if show_brier_score:
                legend_label += f" $\Rightarrow$ (Brier score: {brier_score:.4f})"

            if overlay:
                plt.plot(
                    prob_pred,
                    prob_true,
                    marker=marker,
                    label=legend_label,
                    **curve_style,
                    **kwargs,
                )
            elif grid:
                ax = axes[idx]
                ax.plot(
                    prob_pred,
                    prob_true,
                    marker=marker,
                    label=legend_label,
                    **curve_style,
                    **kwargs,
                )
                linestyle_kwgs = linestyle_kwgs or {}
                linestyle_kwgs.setdefault("color", "gray")
                linestyle_kwgs.setdefault("linestyle", "--")
                ax.plot(
                    [0, 1],
                    [0, 1],
                    label="Perfectly Calibrated",
                    **linestyle_kwgs,
                )
                ax.set_xlabel(xlabel, fontsize=label_fontsize)
                ax.set_ylabel(ylabel, fontsize=label_fontsize)
                if text_wrap:
                    grid_title = "\n".join(
                        textwrap.wrap(f"Calibration Curve: {name}", width=text_wrap)
                    )
                else:
                    grid_title = f"Calibration Curve: {name}"
                if grid_title:
                    ax.set_title(grid_title, fontsize=label_fontsize)
                ax.legend(loc="upper left", fontsize=tick_fontsize)
                ax.tick_params(axis="both", labelsize=tick_fontsize)
                ax.grid(visible=gridlines)
            else:
                plt.figure(figsize=figsize or (8, 6))
                plt.plot(
                    prob_pred,
                    prob_true,
                    marker=marker,
                    label=legend_label,
                    **curve_style,
                    **kwargs,
                )
                linestyle_kwgs = linestyle_kwgs or {}
                linestyle_kwgs.setdefault("color", "gray")
                linestyle_kwgs.setdefault("linestyle", "--")
                plt.plot(
                    [0, 1],
                    [0, 1],
                    label="Perfectly Calibrated",
                    **linestyle_kwgs,
                )
                plt.xlabel(xlabel, fontsize=label_fontsize)
                plt.ylabel(ylabel, fontsize=label_fontsize)
                if text_wrap:
                    title = "\n".join(
                        textwrap.wrap(f"Calibration Curve: {name}", width=text_wrap)
                    )
                else:
                    title = f"Calibration Curve: {name}"
                if title:
                    plt.title(title, fontsize=label_fontsize)
                plt.legend(loc="upper left", fontsize=tick_fontsize)
                plt.grid(visible=gridlines)
                self._save_plot(
                    f"{name}_Calibration", save_plot, image_path_png, image_path_svg
                )
                plt.show()

        if overlay:
            linestyle_kwgs = linestyle_kwgs or {}
            linestyle_kwgs.setdefault("color", "gray")
            linestyle_kwgs.setdefault("linestyle", "--")
            plt.plot(
                [0, 1],
                [0, 1],
                label="Perfectly Calibrated",
                **linestyle_kwgs,
            )
            plt.xlabel(xlabel, fontsize=label_fontsize)
            plt.ylabel(ylabel, fontsize=label_fontsize)
            if text_wrap:
                title = "\n".join(
                    textwrap.wrap(
                        title or "Calibration Curves: Overlay", width=text_wrap
                    )
                )
            else:
                title = title or "Calibration Curves: Overlay"
            if title:
                plt.title(title, fontsize=label_fontsize)
            plt.legend(loc="upper left", fontsize=tick_fontsize)
            plt.grid(visible=gridlines)
            self._save_plot(
                "Overlay_Calibration", save_plot, image_path_png, image_path_svg
            )
            plt.show()
        elif grid:
            for ax in axes[len(pipelines_or_models) :]:
                ax.axis("off")
            plt.tight_layout()
            self._save_plot(
                "Grid_Calibration", save_plot, image_path_png, image_path_svg
            )
            plt.show()


################################################################################
######### ModelCalculator Class to calculate SHAP values,
######### model coefficients(where applicable), tp,fn,fn,fp, and pred_probas
################################################################################


class ModelCalculator:
    def __init__(self, model_dict, outcomes, top_n=3):
        """
        Initialize the ModelCalculator class.

        Parameters:
        -----------
        model_dict : dict
            A dictionary where keys are outcomes and values are trained models.

        outcomes : list
            A list of outcome names to evaluate.

        top_n : int, optional (default=3)
            The number of top SHAP features to extract for each row.
        """
        self.model_dict = model_dict
        self.outcomes = outcomes
        self.top_n = top_n

    def generate_predictions(
        self,
        X_test,
        y_test,
        calculate_shap=False,
        use_coefficients=False,
        include_contributions=False,
        subset_results=False,
        global_shap=False,  # Global SHAP extraction
        global_coefficients=False,  # Global coefficient extraction
    ):
        """
        Generate a prediction DataFrame with metrics, SHAP values, and
        coefficients.

        This function calculates predictions, metrics, SHAP values, and
        coefficients for the provided test data. It supports both row-wise and
        global-level computations.

        Parameters:
        -----------
        X_test : pd.DataFrame or dict
            The test dataset. Can be:
            - A single DataFrame used for all outcomes, or
            - A dictionary where keys are outcomes and values are corresponding
              DataFrames.

        y_test : pd.DataFrame
            A DataFrame containing true labels for each outcome as columns.

        calculate_shap : bool, optional (default=False)
            Whether to calculate SHAP values for each row in the dataset.

        use_coefficients : bool, optional (default=False)
            Whether to calculate coefficients for each row in the dataset.

        include_contributions : bool, optional (default=False)
            If True, include the contribution values alongside feature names
            for SHAP or coefficient calculations.

        subset_results : bool, optional (default=False)
            Whether to return a subset of columns (e.g., metrics and top-n features)
            for row-wise results. Only applicable when `calculate_shap` or
            `use_coefficients` is True.

        global_shap : bool, optional (default=False)
            Whether to compute global SHAP values across the entire dataset.
            If enabled, returns a DataFrame with global SHAP values.

        global_coefficients : bool, optional (default=False)
            Whether to compute global coefficients for the model. If enabled,
            returns a DataFrame with global coefficients.

        Returns:
        --------
        pd.DataFrame
            - If `global_shap` is True: Returns a DataFrame with global SHAP values.
            - If `global_coefficients` is True: Returns a DataFrame with global
              coefficients.
            - Otherwise: Returns a DataFrame containing row-wise predictions,
              metrics, and top-n features or coefficients.

        Raises:
        -------
        ValueError
            If conflicting parameters are set to True simultaneously:
            - `calculate_shap` and `use_coefficients`
            - `global_shap` and `global_coefficients`
            - `global_coefficients` and `subset_results`
            - `global_shap` and `subset_results`

        Notes:
        ------
        - When `global_shap` or `global_coefficients` is enabled, the computation
          immediately stops after generating the corresponding global-level output,
          and no row-wise computations are performed.
        - The `subset_results` flag only applies to row-wise outputs and has no
          effect when `global_shap`
          or `global_coefficients` is enabled.
        """

        ## Define conflicting conditions and their error messages
        conflicts = [
            (
                calculate_shap and use_coefficients,
                "Both 'calculate_shap' and 'use_coefficients' cannot be True simultaneously.",
            ),
            (
                global_coefficients and global_shap,
                "Both 'global_shap' and 'global_coefficients' cannot be True simultaneously.",
            ),
            (
                global_coefficients and subset_results,
                "Both 'global_coefficients' and 'subset_results' cannot be True simultaneously.",
            ),
            (
                global_shap and subset_results,
                "Both 'global_shap' and 'subset_results' cannot be True simultaneously.",
            ),
        ]

        ## Check for conflicts
        for condition, message in conflicts:
            if condition:
                raise ValueError(message)

        results = []

        for outcome in self.outcomes:
            print(f"Running metrics for {outcome}...")

            ## Retrieve the trained model
            model = self.model_dict["model"].get(outcome)
            if model is None:
                print(f"Model for outcome '{outcome}' not found. Skipping...")
                continue

            if hasattr(model, "estimator"):
                estimator = model.estimator
            else:
                estimator = model

            ## Extract X_test and y_test
            X_test_m = X_test[outcome] if isinstance(X_test, dict) else X_test.copy()
            y_test_m = y_test[outcome].copy()

            ## Handle Global Coefficients
            if global_coefficients:
                print("Calculating global coefficients...")
                global_coeff_df = self._calculate_coefficients(
                    estimator, X_test_m, global_coefficients=True
                )
                print(f"Global Coefficients for {outcome}:\n{global_coeff_df}")
                return global_coeff_df  # Immediately return the global coefficients DataFrame

            ## Handle Global SHAP Values
            if global_shap:
                print("Calculating global SHAP values...")
                global_shap_df = self._calculate_shap_values(
                    estimator, X_test_m, global_shap=True
                )
                print(f"Global SHAP Values for {outcome}:\n{global_shap_df}")
                return global_shap_df  # Immediately return the global SHAP DataFrame

            ## Row-wise Calculations
            if hasattr(model, "threshold"):
                threshold = list(model.threshold.values())[0]
                print(f"Using optimal threshold {threshold}")
                y_pred_proba = model.predict_proba(X_test_m)[:, 1]
                y_pred = 1 * (y_pred_proba > threshold)
                print("total predicted positive:", y_pred.sum())
            else:
                y_pred = model.predict(X_test_m)

            try:
                y_pred_proba = model.predict_proba(X_test_m)[:, 1]
            except AttributeError:
                y_pred_proba = [0.5] * len(y_pred)

            # Add metrics
            X_test_with_metrics = self._add_metrics(
                X_test_m.copy(), y_test_m, y_pred, y_pred_proba, outcome
            )

            # Per-row SHAP
            if calculate_shap:
                print("Calculating SHAP values per row...")
                shap_values = self._calculate_shap_values(
                    estimator, X_test_m, include_contributions
                )
                X_test_with_metrics[f"top_{self.top_n}_features"] = shap_values

            # Per-row Coefficients
            if use_coefficients:
                print("Calculating coefficients per row...")
                coeff_values = self._calculate_coefficients(
                    estimator,
                    X_test_m,
                    include_contributions=include_contributions,
                )
                X_test_with_metrics[f"top_{self.top_n}_coefficients"] = coeff_values

            results.append(X_test_with_metrics)

        ## Combine row-wise results
        results_df = pd.concat(results, axis=0)

        ## Subset results only for row-wise data
        if subset_results:
            print("Subset results only applies to row-wise outputs.")
            top_features_col = (
                f"top_{self.top_n}_features"
                if calculate_shap
                else f"top_{self.top_n}_coefficients"
            )
            subset_cols = ["TP", "FN", "FP", "TN", "y_pred_proba", top_features_col]
            results_df = results_df[subset_cols]

        print(f"Shape of results_df: {results_df.shape}")
        return results_df

    def _extract_final_model(self, model):
        """
        Extracts the final estimator from a model or pipeline.

        Parameters:
        -----------
        model : object
            The model or pipeline to extract the final estimator from.

        Returns:
        --------
        final_model : object
            The final estimator object.

        Raises:
        -------
        ValueError
            If the model or pipeline structure is unsupported.
        """
        try:
            # Standalone model

            if hasattr(model, "steps"):
                pipeline = model
                if hasattr(pipeline, "estimator"):
                    final_model = pipeline.estimator
                else:
                    final_model = pipeline.steps[-1][1]  # Access the last step
                print(
                    f"Pipeline detected. Extracted model: {type(final_model).__name__}"
                )
                return final_model
            elif (
                hasattr(model, "predict_proba")
                or hasattr(model, "decision_function")
                or hasattr(model, "predict")
            ):
                print("Standalone model detected.")
                return model
            # Pipeline logic
            # Custom handling for non-callable models
            elif hasattr(model, "model"):
                # Handle wrapped models
                return model.model
            else:
                raise ValueError("Unsupported model or pipeline structure.")
        except Exception as e:
            raise ValueError(f"Error extracting final model: {e}")

    def _add_metrics(
        self,
        X_test_m,
        y_test_m,
        y_pred,
        y_pred_proba,
        outcome,
    ):
        """
        Add performance metrics to the DataFrame.

        Parameters:
        -----------
        X_test_m : pd.DataFrame
            The feature matrix.

        y_test_m : pd.Series
            The true labels.

        y_pred : array-like
            The predicted labels.

        y_pred_proba : array-like
            The predicted probabilities.

        outcome : str
            The name of the outcome.

        Returns:
        --------
        X_test_m : pd.DataFrame
            The feature matrix with metrics added.
        """
        metrics_df = X_test_m.copy()
        metrics_df["TP"] = ((y_test_m == 1) & (y_pred == 1)).astype(int)
        metrics_df["FN"] = ((y_test_m == 1) & (y_pred == 0)).astype(int)
        metrics_df["FP"] = ((y_test_m == 0) & (y_pred == 1)).astype(int)
        metrics_df["TN"] = ((y_test_m == 0) & (y_pred == 0)).astype(int)
        metrics_df["y_pred_proba"] = y_pred_proba
        metrics_df["Outcome"] = outcome
        return metrics_df

    def _calculate_shap_values(
        self,
        model,
        X_test_m,
        include_contributions=False,
        global_shap=False,
        sample_size=None,
    ):
        """
        Calculate SHAP values for a given model or pipeline.

        Dynamically handles standalone models and pipelines by extracting the final estimator.
        """
        if X_test_m.index.name is not None:
            X_test_m = X_test_m.reset_index(drop=True)
        X_test_m = X_test_m.select_dtypes(include=["number"])

        if global_shap and sample_size is not None and sample_size < len(X_test_m):
            print(f"Sampling {sample_size} rows from X_test_m for global SHAP...")
            X_test_m = X_test_m.sample(n=sample_size, random_state=42)

        # Dynamically handle pipelines and standalone models
        try:
            if hasattr(model, "predict_proba"):
                # Standalone model
                final_model = model.predict_proba
                X_transformed = X_test_m

            else:
                raise ValueError("Unsupported model or pipeline structure.")
        except Exception as e:
            raise ValueError(f"Error extracting model or transforming data: {e}")

        # Initialize SHAP Explainer
        try:
            explainer = shap.Explainer(final_model, X_transformed)
        except Exception as e:
            print(f"Error initializing SHAP explainer with default method: {e}")
            # Fallback to KernelExplainer
            try:
                explainer = shap.KernelExplainer(
                    final_model.predict_proba, X_transformed
                )
                print("Using KernelExplainer as a fallback.")
            except Exception as kernel_error:
                raise ValueError(
                    f"Unable to initialize SHAP explainer for the provided model. "
                    f"Default error: {e}, KernelExplainer error: {kernel_error}"
                )

        # Global SHAP calculation
        if global_shap:
            print("Computing global SHAP values...")
            shap_values_accumulated = []
            try:
                for row_batch in tqdm(
                    X_transformed.itertuples(index=False, name=None),
                    total=len(X_transformed),
                    desc="Global SHAP Progress",
                ):
                    row = pd.DataFrame([row_batch], columns=X_test_m.columns)
                    shap_values = explainer(row).values
                    shap_values_accumulated.append(shap_values)

                shap_values = np.vstack(shap_values_accumulated)

                if len(shap_values.shape) == 3:  # Multi-class SHAP values
                    global_shap_values = shap_values.mean(axis=0).mean(axis=1)
                elif len(shap_values.shape) == 2:  # Single-class SHAP values
                    global_shap_values = shap_values.mean(axis=0)
                else:
                    raise ValueError(
                        f"Unexpected SHAP values shape: {shap_values.shape}"
                    )

                shap_df = pd.DataFrame(
                    {"Feature": X_test_m.columns, "SHAP Value": global_shap_values}
                ).sort_values(by="SHAP Value", key=abs, ascending=False)
                return shap_df.reset_index(drop=True)

            except KeyboardInterrupt:
                print(
                    "\nKeyboardInterrupt detected! Returning partial global SHAP values..."
                )
                if shap_values_accumulated:
                    shap_values = np.vstack(shap_values_accumulated)
                    global_shap_values = shap_values.mean(axis=0).mean(axis=1)
                    shap_df = pd.DataFrame(
                        {"Feature": X_test_m.columns, "SHAP Value": global_shap_values}
                    ).sort_values(by="SHAP Value", key=abs, ascending=False)
                    return shap_df.reset_index(drop=True)
                else:
                    return pd.DataFrame({"Feature": [], "SHAP Value": []})

        # Row-wise SHAP calculation
        shap_values = []
        try:
            # for row in tqdm(
            #     X_transformed.itertuples(index=False),
            #     total=len(X_transformed),
            #     desc="SHAP Progress",
            # ):
            #     shap_values.append(
            #         explainer(
            #             pd.DataFrame([row], columns=X_test_m.columns),
            #         ).values[0]
            #     )
            # X_transformed.columns = X_test_m.columns
            print(X_transformed)

            shap_values = explainer(
                pd.DataFrame(X_transformed.values, columns=X_test_m.columns)
            ).values
            # quit()
        except KeyboardInterrupt:
            print("\nKeyboardInterrupt detected. Returning results calculated so far.")
            return shap_values

        # Extract top-n SHAP features per row
        top_shap_features = [
            [
                feature
                for feature, shap_value in sorted(
                    zip(
                        X_test_m.columns,
                        (
                            shap_row.flatten()
                            if hasattr(shap_row, "flatten")
                            else shap_row
                        ),
                    ),
                    key=lambda x: abs(x[1]),
                    reverse=True,
                )[: self.top_n]
            ]
            for shap_row in shap_values
        ]

        if include_contributions:
            return [
                {feature: value for feature, value in zip(X_test_m.columns, shap_row)}
                for shap_row in shap_values
            ]

        return top_shap_features

    def _calculate_coefficients(
        self, model, X_test_m, include_contributions=False, global_coefficients=False
    ):
        final_model = self._extract_final_model(model)
        print(final_model)

        if hasattr(final_model, "coef_"):
            coefficients = final_model.coef_
            if coefficients.ndim > 1:
                coefficients = coefficients[0]
            feature_names = getattr(
                final_model, "feature_names_in_", X_test_m.columns[: len(coefficients)]
            )
            if global_coefficients:
                return pd.DataFrame(
                    {"Feature": feature_names, "Coefficient": coefficients}
                ).sort_values(by="Coefficient", key=abs, ascending=False)
            contributions = X_test_m[feature_names].values * coefficients
            return [
                {
                    feature: value
                    for feature, value in sorted(
                        zip(feature_names, row), key=lambda x: abs(x[1]), reverse=True
                    )[: self.top_n]
                }
                for row in contributions
            ]

        if hasattr(final_model, "feature_importances_"):
            feature_importances = final_model.feature_importances_
            feature_names = X_test_m.columns[: len(feature_importances)]
            return pd.DataFrame(
                {"Feature": feature_names, "Importance": feature_importances}
            ).sort_values(by="Importance", key=abs, ascending=False)

        raise ValueError(f"Model {type(final_model)} does not support coefficients.")

    # def _calculate_coefficients(
    #     self,
    #     model,
    #     X_test_m,
    #     include_contributions=False,
    #     global_coefficients=False,
    # ):
    #     """
    #     Calculate feature contributions for models with `coef_`.

    #     Parameters:
    #     -----------
    #     model : trained model
    #         The trained model, which must have a `coef_` attribute.

    #     X_test_m : pd.DataFrame
    #         The feature matrix with possibly additional columns.

    #     include_contributions : bool, optional (default=False)
    #         Whether to include contribution values alongside feature names.

    #     global_coefficients : bool, optional (default=False)
    #         Whether to calculate global coefficients for the model.

    #     Returns:
    #     --------
    #     - If global_coefficients is True: Dictionary of feature names and coefficients.
    #     - If global_coefficients is False: Per-row contributions (existing functionality).
    #     """
    #     ## Extract the final model
    #     if hasattr(model, "estimator"):
    #         wrapped_estimator = model.estimator
    #         if isinstance(wrapped_estimator, CalibratedClassifierCV):
    #             pipeline = wrapped_estimator.estimator
    #             final_model = (
    #                 pipeline.steps[-1][1]
    #                 if hasattr(
    #                     pipeline,
    #                     "steps",
    #                 )
    #                 else pipeline
    #             )
    #         else:
    #             final_model = wrapped_estimator
    #     else:
    #         final_model = model

    #     if not hasattr(final_model, "coef_"):
    #         raise ValueError(f"The model {type(final_model)} does not support `coef_`.")

    #     ## Retrieve coefficients
    #     coefficients = final_model.coef_
    #     if len(coefficients.shape) > 1:
    #         ## Use the first class if it's a multi-class model
    #         coefficients = coefficients[0]

    #     ## Global Coefficients: Return as a dictionary
    #     if global_coefficients:
    #         feature_names = (
    #             final_model.feature_names_in_
    #             if hasattr(final_model, "feature_names_in_")
    #             else X_test_m.columns[: len(coefficients)]
    #         )
    #         coeff_df = pd.DataFrame(
    #             {"Feature": feature_names, "Coefficient": coefficients}
    #         ).sort_values(by="Coefficient", key=abs, ascending=False)
    #         return coeff_df.reset_index(drop=True)

    #     ## Per-Row Contributions: Existing functionality
    #     feature_names = (
    #         final_model.feature_names_in_
    #         if hasattr(final_model, "feature_names_in_")
    #         else X_test_m.columns[: len(coefficients)]
    #     )
    #     X_test_features = X_test_m[feature_names]
    #     contributions = X_test_features.values * coefficients

    #     ## Collect top features per row with tqdm for progress tracking
    #     top_coeff_features = []
    #     for row_contributions in tqdm(
    #         contributions,
    #         desc="Calculating Coefficients",
    #         total=len(X_test_features),
    #     ):
    #         sorted_features = sorted(
    #             zip(feature_names, row_contributions),
    #             key=lambda x: abs(x[1]),  ## Sort by absolute contribution
    #             reverse=True,
    #         )[: self.top_n]
    #         if include_contributions:
    #             top_coeff_features.append(
    #                 {feature: value for feature, value in sorted_features}
    #             )
    #         else:
    #             top_coeff_features.append([feature for feature, _ in sorted_features])

    #     return top_coeff_features
