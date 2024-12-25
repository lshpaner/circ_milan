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
import warnings


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
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    brier_score_loss,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC


################################################################################
############################## Data Conversions ################################
################################################################################


class HealthMetrics:
    """
    A class to calculate health-related metrics such as Body Mass Index (BMI)
    and Mean Arterial Pressure (MAP) for individuals in a DataFrame.
    """

    def __init__(self):
        pass  # Not storing state in the instance.

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
######################### Stratified KFold Split ###############################
################################################################################


def stratified_kfold_split(
    df,
    target_col,
    data_path,
    n_splits=5,
    random_state=222,
):
    """
    Perform stratified K-fold splitting of a dataframe, save the splits to files,
    print the size and percentage of each fold, and return the splits.

    Parameters:
    - df : pandas.DataFrame
        The DataFrame to split.
    - target_col : str
        The name of the column to use as the target for stratification.
    - data_path : str
        The directory path where the splits will be saved.
    - n_splits : int, default 5
        The number of folds.
    - random_state : int, default 222
        The random seed for reproducibility.

    Returns:
    - List of dictionaries: Each dictionary contains 'X_train', 'X_test',
    'y_train', 'y_test' for each fold.
    """
    np.random.seed(random_state)  # For reproducibility
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    total_size = len(df)  # Total size of the dataset
    fold_counter = 1
    splits = []  # Initialize the list to store results from each fold

    for train_index, test_index in skf.split(
        df.drop(target_col, axis=1), df[target_col]
    ):
        X_train, X_test = (
            df.drop(target_col, axis=1).iloc[train_index],
            df.drop(target_col, axis=1).iloc[test_index],
        )
        y_train, y_test = (
            df[target_col].iloc[train_index],
            df[target_col].iloc[test_index],
        )

        # Define file paths for train and test sets
        train_filename = os.path.join(
            data_path,
            f"train_split_{fold_counter}.parquet",
        )
        test_filename = os.path.join(
            data_path,
            f"test_split_{fold_counter}.parquet",
        )

        # Save the training set
        X_train.join(y_train).to_parquet(train_filename)
        # Save the testing set
        X_test.join(y_test).to_parquet(test_filename)

        # Print fold details
        train_percent = 100 * len(train_index) / total_size
        test_percent = 100 * len(test_index) / total_size
        print(f"Fold {fold_counter}:")
        print(f"Train Set Size: {len(train_index)}, {train_percent:.2f}% of total")
        print(f"Test Set Size: {len(test_index)}, {test_percent:.2f}% of total")
        print(f"Saved fold {fold_counter} to {train_filename} and {test_filename}")
        print("---")  # Separator for clarity

        # Append the split information to the list
        splits.append(
            {
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test,
            }
        )

        fold_counter += 1

    print("All splits have been saved successfully.")
    return splits


################################################################################
################### Modeling Fit, Predict, and Evaluation ######################
################################################################################


def evaluate_model_pipelines(splits, pipelines, param_grids=None):
    results = {pipeline_name: [] for pipeline_name, _ in pipelines}
    predictions = {pipeline_name: [] for pipeline_name, _ in pipelines}
    true_values = {pipeline_name: [] for pipeline_name, _ in pipelines}
    roc_data = {pipeline_name: [] for pipeline_name, _ in pipelines}
    aggregated_conf_matrices = {pipeline_name: None for pipeline_name, _ in pipelines}
    best_params = {
        pipeline_name: [] for pipeline_name, _ in pipelines
    }  # Store best parameters for each pipeline

    for fold_index, (X_train, X_test, y_train, y_test) in enumerate(splits, start=1):
        fold_progress = tqdm(
            total=len(pipelines), desc=f"Fold {fold_index}/{len(splits)}"
        )
        for pipeline_name, pipeline in pipelines:
            if param_grids and pipeline_name in param_grids:
                grid_search = GridSearchCV(
                    pipeline,
                    param_grids[pipeline_name],
                    cv=5,
                    scoring="roc_auc",
                    n_jobs=-1,
                    verbose=0,
                )
                grid_search.fit(X_train, y_train)
                best_pipeline = grid_search.best_estimator_
                best_params[pipeline_name].append(
                    grid_search.best_params_
                )  # Collect the best parameters for each fold
                y_pred = best_pipeline.predict(X_test)
                y_prob = best_pipeline.predict_proba(X_test)[:, 1]
                fold_progress.update(1)
            else:
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)
                y_prob = pipeline.predict_proba(X_test)[:, 1]
                fold_progress.update(1)

            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_data[pipeline_name].append((fpr, tpr, _))
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            auc_roc = roc_auc_score(y_test, y_prob)
            pr_auc = average_precision_score(y_test, y_prob)
            avg_precision = average_precision_score(y_test, y_prob)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
            brier = brier_score_loss(y_test, y_prob)
            conf_matrix = confusion_matrix(y_test, y_pred)

            if aggregated_conf_matrices[pipeline_name] is None:
                aggregated_conf_matrices[pipeline_name] = conf_matrix
            else:
                aggregated_conf_matrices[pipeline_name] += conf_matrix

            detailed_report = classification_report(y_test, y_pred, zero_division=0)
            formatted_conf_matrix = (
                f"[ TN = {tn:3d} , FP = {fp:3d} ]\n[ FN = {fn:3d} , TP = {tp:3d} ]"
            )

            predictions[pipeline_name].append(y_prob)
            true_values[pipeline_name].append(y_test)
            results[pipeline_name].append(
                {
                    "fold": fold_index,
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                    "auc_roc": auc_roc,
                    "pr_auc": pr_auc,
                    "avg_precision": avg_precision,
                    "specificity": specificity,
                    "brier_score": brier,
                    "confusion_matrix": formatted_conf_matrix,
                    "detailed_report": detailed_report,
                    "conf_matrix": conf_matrix,
                }
            )

            print(f"\nResults for {pipeline_name} - Fold {fold_index}")
            print(
                f"Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1-Score: {f1:.3f}"
            )
            print(
                f"AUC ROC: {auc_roc:.3f}, PR AUC: {pr_auc:.3f}, Avg Precision: {avg_precision:.3f}, Specificity: {specificity:.3f}"
            )
            print(f"Brier Score: {brier:.3f}")
            print("Classification Report:")
            print(detailed_report)
            print("Confusion Matrix:")
            print(formatted_conf_matrix)
            print("-" * 60)

        fold_progress.close()

    # After all folds are processed, print the best parameters for each model
    for pipeline_name, params in best_params.items():
        print(f"\nOverall best parameters for {pipeline_name} after all folds:")
        print(params)

    return results, roc_data, true_values, predictions, aggregated_conf_matrices


################################################################################


def summarize_evaluation(
    results,
    model_labels=None,  # Mapping of internal model names to descriptive labels
    display_metrics=True,
    display_confusion_matrices=True,
    include_std=True,  # Control the display of standard deviation
    print_results=False,  # New parameter to control the printing of results
):

    if model_labels is None:
        model_labels = {}

    summary = {
        model_labels.get(pipeline_name, pipeline_name): {
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1_score": [],
            "auc_roc": [],
            "pr_auc": [],
            "avg_precision": [],
            "specificity": [],
            "brier_score": [],
            "conf_matrix": np.zeros((2, 2), dtype=int),
        }
        for pipeline_name in results
    }

    for pipeline_name, folds in results.items():
        # Use custom label if available
        display_name = model_labels.get(pipeline_name, pipeline_name)
        for fold in folds:
            summary[display_name]["accuracy"].append(fold["accuracy"])
            summary[display_name]["precision"].append(fold["precision"])
            summary[display_name]["recall"].append(fold["recall"])
            summary[display_name]["f1_score"].append(fold["f1_score"])
            summary[display_name]["auc_roc"].append(fold["auc_roc"])
            summary[display_name]["pr_auc"].append(fold["pr_auc"])
            summary[display_name]["avg_precision"].append(fold["avg_precision"])
            summary[display_name]["specificity"].append(fold["specificity"])
            summary[display_name]["brier_score"].append(fold["brier_score"])
            summary[display_name]["conf_matrix"] += fold["conf_matrix"]

    # Create a list of tuples for each metric and statistic
    metrics_data = []
    statistics = ["Mean"]
    if include_std:
        statistics.append("Standard Deviation")

    for model_name, metrics in summary.items():
        for stat in statistics:
            metric_values = []
            for metric in [
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "auc_roc",
                "pr_auc",
                "avg_precision",
                "specificity",
                "brier_score",
            ]:
                if metric != "conf_matrix":
                    if stat == "Mean":
                        metric_values.append(round(np.mean(metrics[metric]), 3))
                    elif stat == "Standard Deviation":
                        metric_values.append(round(np.std(metrics[metric]), 3))
            metrics_data.append([stat, model_name] + metric_values)

    # Create the DataFrame without setting an index
    metrics_df = pd.DataFrame(
        metrics_data,
        columns=["Metric", "Model_Name"]
        + [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "AUC ROC",
            "PR AUC",
            "Average Precision",
            "Specificity",
            "Brier Score",
        ],
    )

    # Remove the default integer index
    metrics_df.reset_index(drop=True, inplace=True)

    # Create DataFrame for confusion matrices, if needed
    confusion_matrices = {}
    if display_confusion_matrices:
        for model_name in summary:
            cm = summary[model_name]["conf_matrix"]
            cm_df = pd.DataFrame(
                cm,
                columns=["Predicted Negative", "Predicted Positive"],
                index=["Actual Negative", "Actual Positive"],
            )
            confusion_matrices[model_name] = cm_df

            # Optional printing of confusion matrix
            if print_results:
                print(f"{model_name} Confusion Matrix:")
                print(cm_df.to_string(index=False))
                print()

    # Optional printing of metrics DataFrame
    if print_results:
        print("Evaluation Metrics:")
        print(metrics_df.to_string(index=False))
        print()

    return metrics_df, cm_df


################################################################################
########################## Aggregated Confusion Matrix #########################
################################################################################


def plot_aggregated_confusion_matrices(
    aggregated_conf_matrices,
    model_labels=None,
    image_path_png=None,
    image_path_svg=None,
):
    if model_labels is None:
        model_labels = {}

    num_models = len(aggregated_conf_matrices)

    # Check if there's only one model and adjust layout accordingly
    if num_models == 1:
        fig, ax = plt.subplots(
            figsize=(5, 4)
        )  # You can set the desired single figsize here
        axes = [ax]  # Wrap ax in a list to make it iterable
    else:
        fig, axes = plt.subplots(
            1, num_models, figsize=(10 * num_models, 8)
        )  # Horizontal layout for multiple models

    for ax, (model_name, cm) in zip(axes, aggregated_conf_matrices.items()):
        display_name = model_labels.get(
            model_name, model_name
        )  # Use custom label if available
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="viridis", ax=ax
        )  # 'viridis' is a good choice for visibility
        ax.set_title(f"{display_name}: Confusion Matrix")
        ax.set_xlabel("Predicted labels")
        ax.set_ylabel("True labels")

    plt.tight_layout()
    # Check if image paths are provided and save the figures accordingly
    if image_path_png:
        plt.savefig(os.path.join(image_path_png, "aggregated_confusion_matrix.png"))
    if image_path_svg:
        plt.savefig(os.path.join(image_path_svg, "aggregated_confusion_matrix.svg"))
    plt.show()


################################################################################
############################### ROC AUC per KFold ##############################
################################################################################


def plot_roc_curves(
    roc_data,
    true_values,
    prob_predictions,
    plot_type="both",  # Options: 'individual', 'aggregated', 'both'
    individual_line_style="-",  # Default to solid line for individual curves
    aggregated_line_style="-",  # Always solid for aggregated curves
    pipeline_labels=None,  # Mapping of pipeline_name to custom labels
    aggregate_on_one_plot=False,  # Plot all aggregated curves on one plot
    image_path_png=None,
    image_path_svg=None,
):
    if pipeline_labels is None:
        pipeline_labels = {}

    # Container to store information for aggregated plot
    aggregated_data = []

    for pipeline_name, data in roc_data.items():
        display_name = pipeline_labels.get(pipeline_name, pipeline_name)
        individual_aucs = [
            roc_auc_score(
                true_values[pipeline_name][i], prob_predictions[pipeline_name][i]
            )
            for i in range(len(data))
        ]
        average_auc = np.mean(individual_aucs)

        # Collect all true values and predictions if aggregated curves are required
        if plot_type in ["aggregated", "both"]:
            all_y_test = np.concatenate(
                [true_values[pipeline_name][i] for i in range(len(data))]
            )
            all_y_prob = np.concatenate(
                [prob_predictions[pipeline_name][i] for i in range(len(data))]
            )
            if aggregate_on_one_plot:
                aggregated_data.append(
                    (all_y_test, all_y_prob, display_name, average_auc)
                )

        # Plotting individual or both (without combined aggregation)
        if not aggregate_on_one_plot or plot_type == "individual":
            plt.figure()
            plt.plot([0, 1], [0, 1], "k--")  # No effect line

            title = ""
            if plot_type in ["individual", "both"]:
                title = f"AUC ROC for {display_name} by Fold"
                for fold_index, (fold_fpr, fold_tpr, _) in enumerate(data):
                    plt.plot(
                        fold_fpr,
                        fold_tpr,
                        linestyle=individual_line_style,
                        label=(
                            f"{display_name} Fold {fold_index + 1} "
                            f"(AUC = {individual_aucs[fold_index]:.3f})"
                        ),
                    )

            if plot_type in ["aggregated", "both"]:
                title = (
                    f"AUC ROC by Fold with Aggregated {display_name}"
                    if plot_type == "both"
                    else f"AUC ROC for {display_name}"
                )
                fpr_aggregated, tpr_aggregated, _ = roc_curve(all_y_test, all_y_prob)
                plt.plot(
                    fpr_aggregated,
                    tpr_aggregated,
                    linestyle=aggregated_line_style,
                    label=f"{display_name} (AUC = {average_auc:.3f})",
                )

            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(title)
            plt.legend(loc="lower right")

            # Determine file name for saving the plot
            file_name = (
                title.replace(" ", "_")
                .replace("{", "")
                .replace("}", "")
                .replace("__", "_")
                .lower()
            )

            # Save to image files if paths are provided
            if image_path_png:
                plt.savefig(os.path.join(image_path_png, f"{file_name}.png"))
            if image_path_svg:
                plt.savefig(os.path.join(image_path_svg, f"{file_name}.svg"))

            plt.show()

    # Aggregated curves on a single plot
    if aggregate_on_one_plot and plot_type in ["aggregated", "both"]:
        plt.figure()
        plt.plot([0, 1], [0, 1], "k--")  # No effect line
        for all_y_test, all_y_prob, display_name, average_auc in aggregated_data:
            fpr_aggregated, tpr_aggregated, _ = roc_curve(all_y_test, all_y_prob)
            plt.plot(
                fpr_aggregated,
                tpr_aggregated,
                linestyle=aggregated_line_style,
                label=f"{display_name} (AUC = {average_auc:.3f})",
            )
        plt.title("Aggregated ROC Curves Comparison")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")

        # Determine file name for saving the plot
        file_name = "aggregated_roc_curves_comparison".replace(" ", "_").lower()

        # Save to image files if paths are provided
        if image_path_png:
            plt.savefig(os.path.join(image_path_png, f"{file_name}.png"))
        if image_path_svg:
            plt.savefig(os.path.join(image_path_svg, f"{file_name}.svg"))

        plt.show()
