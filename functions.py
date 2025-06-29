################################################################################
######################### Import Requisite Libraries ###########################
################################################################################
import pandas as pd
import numpy as np
import math
import subprocess
import matplotlib.pyplot as plt
from tqdm import tqdm
import mlflow
from mlflow.tracking import MlflowClient
import pickle
import os
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFE


from sklearn.metrics import (
    auc,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    brier_score_loss,
    precision_recall_curve,
)

from sklearn.calibration import calibration_curve
from tqdm import tqdm

from constants import (
    mlflow_artifacts_data,
    mlflow_models_data,
)


################################################################################
################################# Preprocessing ################################
#                                                                              #

############################## Get Terminal Size ###############################
################################################################################


def get_true_terminal_width():
    try:
        columns = int(
            subprocess.check_output(
                ["tput", "cols"], stderr=subprocess.DEVNULL
            ).strip(),
        )
    except Exception:
        columns = 200  # go big or default
    return columns


terminal_width = get_true_terminal_width()


############################ Cleaning DataFrames ###############################
################################################################################


def clean_dataframe(df, cols_with_thousand_separators):
    """
    Cleans a pandas DataFrame by replacing specific values with NaN, removing
    thousand separators, and converting columns to numeric types where possible.

    Parameters:
    ----------
    df : pandas.DataFrame
        The DataFrame to be cleaned.

    cols_with_thousand_separators : list of str
        A list of column names that contain thousand separators and need to be
        processed.

    Returns:
    -------
    pandas.DataFrameThe cleaned DataFrame with NaNs in place of specified values, thousand
        separators removed from specified columns, and columns converted to
        numeric types where applicable.

    Steps:
    -----
    1. Replace None and blank values with NaN using tqdm for progress tracking.
       The following replacements are made:
       - None to NaN
       - Blank strings ("") to NaN
       - Sequences of two or more hyphens ("--") to NaN
       - Sequences of two or more dots ("..") to NaN

    2. Remove thousand separators and convert to numeric for specified columns.
       - If the column is of object type, remove thousand separators (commas).
       - Convert the cleaned columns to numeric types, coercing errors to NaN.

    3. Convert all other columns to numeric types where possible.
       - Columns that are not listed in `cols_with_thousand_separators` are
         attempted to be converted to numeric types.
       - Errors during conversion are ignored, keeping the original non-numeric
         values.

    Notes:
    -----
    - Uses tqdm for progress tracking during the cleaning process.
    """

    # Step 1: Replace None and blank values with NaN using tqdm for progress tracking
    replacements = {
        None: np.nan,
        "": np.nan,
        "-{2,}": np.nan,
        "\.{2,}": np.nan,
    }

    for col in tqdm(df.columns, desc="Replacing values in columns"):
        for to_replace, value in replacements.items():
            if to_replace is None:
                df[col] = df[col].map(lambda x: value if x is to_replace else x)
            else:
                df[col] = df[col].replace(to_replace, value, regex=True)

    # Step 2: Remove thousand separators and convert to numeric
    desc_text = "Processing columns with thousand separators"
    for col in tqdm(cols_with_thousand_separators, desc=desc_text):
        if col in df.columns:

            # Remove thousand separators only if the column is of object type
            if df[col].dtype == "object":
                df[col] = df[col].str.replace(",", "", regex=False)
            # Convert to numeric
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Step 3: Convert all other columns to numeric if possible
    for col in tqdm(df.columns, desc="Converting columns to numeric"):
        if col not in cols_with_thousand_separators:
            df[col] = pd.to_numeric(df[col], errors="ignore")

    return df


def clean_feature_selection_params(pipeline_steps, tuned_parameters):
    """
    Remove feature selection parameters from tuned_parameters if RFE is not in
    0pipeline_steps.

    Args:
        pipeline_steps (list): List of tuples containing pipeline steps
        (name, estimator).
        tuned_parameters (list): List of dictionaries with parameters to tune.
    """
    # Check if any step in pipeline_steps is an RFE instance
    has_rfe = any(isinstance(step[1], RFE) for step in pipeline_steps)

    # If no RFE is found, remove feature selection-related parameters
    if not has_rfe:
        for key in list(tuned_parameters[0].keys()):
            if "feature_selection" in key:
                del tuned_parameters[0][key]


def adjust_preprocessing_pipeline(
    model_type,
    pipeline_steps,
    numerical_cols,
    categorical_cols,
    sampler=None,  # Add sampler as an optional argument
):
    no_scale_models = ["xgb", "cat"]
    has_rfe = any(isinstance(step[1], RFE) for step in pipeline_steps)
    use_smote = isinstance(sampler, SMOTE)  # Check if sampler is SMOTE

    if model_type in no_scale_models:
        # Impute if RFE or SMOTE is present
        if has_rfe or use_smote:
            numerical_transformer = Pipeline(
                steps=[("imputer", SimpleImputer(strategy="mean"))],
            )
            categorical_transformer = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="constant",
                            fill_value="missing",
                        ),
                    ),
                    (
                        "encoder",
                        OneHotEncoder(handle_unknown="ignore"),
                    ),
                ]
            )
        else:
            numerical_transformer = Pipeline(
                steps=[("passthrough", "passthrough")],
            )
            categorical_transformer = Pipeline(
                steps=[
                    ("encoder", OneHotEncoder(handle_unknown="ignore")),
                ]
            )

        adjusted_preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_transformer, numerical_cols),
                ("cat", categorical_transformer, categorical_cols),
            ],
            remainder="passthrough",
        )
        return [
            (name, adjusted_preprocessor if name == "Preprocessor" else step)
            for name, step in pipeline_steps
        ]

    return pipeline_steps


################################################################################
# Compare 2 dataframes
################################################################################


# Function to compare two DataFrames
def compare_dataframes(df1, df2):
    if df1.shape != df2.shape:
        print("DataFrames have different shapes:", df1.shape, df2.shape)
        return

    if list(df1.columns) != list(df2.columns):
        print("DataFrames have different columns")
        print("df1 columns:", df1.columns)
        print("df2 columns:", df2.columns)
        return

    if df1.dtypes.equals(df2.dtypes) == False:
        print("DataFrames have different data types")
        print("Differences:\n", df1.dtypes.compare(df2.dtypes))
        return

    if not df1.equals(df2):
        print("DataFrames have different content")
        diff = (df1 != df2).stack()
        print(diff[diff].index.tolist())  # Show locations of differences
        return

    print("No differences found between the DataFrames!")


################################################################################


def extract_relevant_days_hcc_ccs_columns(df):
    """
    Extracts column names from the dataframe based on specific substring conditions.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe.

    Returns:
    --------
    list
        A list of unique column names that match the filtering criteria.
    """
    # Create a list 'days_to' containing all column names in 'df' that
    # include the substring "Daysto"
    days_to = [col for col in df.columns if "Daysto" in col]

    # Create a list 'hcc' containing all column names in 'df' that include
    # the substring "HCC" but do not end with "_HCC"
    hcc = [col for col in df.columns if "HCC" in col and not col.endswith("_HCC")]

    # Create a list 'ccs' containing all column names in 'df' that include
    # the substring "CCS" but do not end with "_CCS"
    ccs = [col for col in df.columns if "CCS" in col and not col.endswith("_CCS")]

    return np.unique(days_to + hcc + ccs).tolist()


################################################################################
############################# Handle Missing Values ############################
################################################################################


def handle_missing_values(df, columns, fillna_value=None):
    """
    Handles missing values in specified columns of a DataFrame.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to modify.
    columns : list
        List of column names to process.
    fillna_value : optional
        The value to fill NaN values with. If None, no filling is performed.

    Returns:
    --------
    pandas.DataFrame
        The modified DataFrame with optional missing value handling.
    """
    if fillna_value is not None:
        df[columns] = df[columns].fillna(fillna_value)

    return df


################################################################################
####################### Safe Conversion of Numeric Features ####################
################################################################################


def safe_to_numeric(series):
    """
    Safely converts a pandas Series to a numeric type, handling errors explicitly.

    This function attempts to convert a pandas Series to a numeric type using
    `pd.to_numeric`. If the conversion fails due to a `ValueError` or `TypeError`,
    it will return the original Series unmodified.

    Parameters:
    -----------
    series : pandas.Series
        The input Series to be converted to a numeric type.

    Returns:
    --------
    pandas.Series
        The converted Series if the conversion is successful; otherwise, the
        original Series is returned.
    """
    try:
        return pd.to_numeric(series)
    except (ValueError, TypeError):
        return series  # If conversion fails, return the original series


################################################################################
############################## Top N Features ##################################
################################################################################


def top_n(series, n=10):
    """
    Returns the top N most frequent unique values from a Pandas Series.

    Parameters:
    -----------
    series : pandas.Series
        The input Series from which to extract the top N most frequent values.
    n : int, optional (default=10)
        The number of top values to return.

    Returns:
    --------
    set
        A set containing the N most frequently occurring unique values in the Series.
    """

    out = set(series.value_counts().head(n).index)

    return out


################################################################################


# Customize the background color for missing values using applymap
def highlight_null(val):
    """
    Highlights null (NaN) values in a DataFrame with a red background color.

    This function checks if a given value is null (NaN) and returns a CSS style
    to apply a red background color if the value is null. If the value is not
    null, it returns an empty string, meaning no styling will be applied.

    Parameters:
    -----------
    val : any
        The value to check. Typically an element of a pandas DataFrame.

    Returns:
    --------
    str
        A string representing the CSS style to apply. If the value is null,
        'background-color: red' is returned; otherwise, an empty string is
        returned.

    Examples:
    ---------
    >>> df.style.applymap(highlight_null)

    This applies the `highlight_null` function element-wise to the DataFrame
    `df`, highlighting any null values with a red background.
    """
    color = "background-color: red" if pd.isnull(val) else ""
    return color


################################################################################
##################### Score Column by Outcome Aggregation ######################
################################################################################

##### Used to inspect death outcomes in supportive_care_eda_preprocessing.ipynb


def generate_aggregated_dataframes(df_scores, df_death, score_columns):
    """
    Generates aggregated dataframes with sum, mean, and percentage calculations
    for each grouping column in df_death.

    Parameters:
    - df_scores (pd.DataFrame): DataFrame containing score columns.
    - df_death (pd.DataFrame): DataFrame containing columns to group by.
    - score_columns (list): List of score column names to aggregate in df_scores.

    Returns:
    - dict: A dictionary where each key is a grouping column from df_death, and
            each value is the resulting DataFrame with calculated totals and percentages.
    """
    # Dictionary to store each resulting DataFrame
    all_results = {}

    # Loop through each column in df_death to create a separate DataFrame for each
    for grouping_col in df_death.columns:
        # Perform grouping and aggregation for each score column
        aggregations = {score: ["sum", "mean"] for score in score_columns}
        grouped_custom = df_scores.groupby(df_death[grouping_col]).agg(aggregations).T

        # Calculate totals and percentages
        grouped_custom["Total"] = grouped_custom[0] + grouped_custom[1]
        grouped_custom["No Death Percent"] = grouped_custom[0] / grouped_custom["Total"]
        grouped_custom["Death Percent"] = grouped_custom[1] / grouped_custom["Total"]
        grouped_custom["Total Percent"] = (
            grouped_custom["No Death Percent"] + grouped_custom["Death Percent"]
        ) * 100

        # Store each DataFrame in a dictionary with the column name as the key
        all_results[grouping_col] = grouped_custom

    return all_results


################################################################################
########################  Metrics to Store in MLFlow ###########################


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
        """
        Plot ROC curves from model predictions or predicted probabilities.

        Parameters
        ----------
        df : pd.DataFrame, optional
            DataFrame containing actual and predicted probability columns.
        outcome_cols : list of str, optional
            Column names for actual binary outcomes in `df`.
        pred_cols : list of str, optional
            Column names for predicted probabilities in `df`.
        models : dict, optional
            Dictionary of trained models with `.predict_proba()` methods.
        X_valid : pd.DataFrame, optional
            Validation features for generating model predictions.
        y_valid : array-like, optional
            True binary labels corresponding to `X_valid`.
        pred_probs_df : pd.DataFrame, optional
            DataFrame of predicted probabilities (one column per model or method).
        model_name : str, optional
            Key to select a specific model from `models`.
        custom_name : str, optional
            Custom title prefix to display on the plot.
        show : bool, default=True
            Whether to display the plot immediately.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The resulting matplotlib figure object.
        """

        fig, _ = plt.subplots(figsize=(8, 8))
        title = None

        if outcome_cols and pred_cols and df is not None:
            for outcome_col, pred_col in zip(outcome_cols, pred_cols):
                y_prob = df[pred_col]
                fpr, tpr, _ = roc_curve(df[outcome_col], y_prob)
                auc_score = roc_auc_score(df[outcome_col], y_prob)
                plt.plot(fpr, tpr, label=f"{outcome_col} (AUC={auc_score:.2f})")

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

        title = (
            f"{custom_name} - Receiver Operating Characteristic"
            if custom_name
            else "Receiver Operating Characteristic"
        )

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
        """
        Plot precision-recall curves from model predictions or predicted
        probabilities.

        Parameters
        ----------
        df : pd.DataFrame, optional
            DataFrame containing actual and predicted probability columns.
        outcome_cols : list of str, optional
            Column names for actual binary outcomes in `df`.
        pred_cols : list of str, optional
            Column names for predicted probabilities in `df`.
        models : dict, optional
            Dictionary of trained models with `.predict_proba()` methods.
        X_valid : pd.DataFrame, optional
            Validation features for generating model predictions.
        y_valid : array-like, optional
            True binary labels corresponding to `X_valid`.
        pred_probs_df : pd.DataFrame, optional
            DataFrame of predicted probabilities (one column per model).
        model_name : str, optional
            Key to select a specific model from `models`.
        custom_name : str, optional
            Custom title prefix to display on the plot.
        show : bool, default=True
            Whether to display the plot immediately.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The resulting matplotlib figure object.
        """

        fig, _ = plt.subplots(figsize=(8, 8))
        title = None

        if outcome_cols and pred_cols and df is not None:
            for outcome_col, pred_col in zip(outcome_cols, pred_cols):
                y_prob = df[pred_col]
                precision, recall, _ = precision_recall_curve(df[outcome_col], y_prob)
                auc_pr = auc(recall, precision)
                plt.plot(
                    recall, precision, label=f"{outcome_col} (AUC-PR={auc_pr:.2f})"
                )

        if models and X_valid is not None and y_valid is not None:
            if model_name:
                y_score = models[model_name].predict_proba(X_valid)[:, 1]
                precision, recall, _ = precision_recall_curve(y_valid, y_score)
                auc_pr = auc(recall, precision)
                plt.plot(recall, precision, label=f"{model_name} (AUC-PR={auc_pr:.2f})")
            else:
                for name, model in models.items():
                    y_score = model.predict_proba(X_valid)[:, 1]
                    precision, recall, _ = precision_recall_curve(y_valid, y_score)
                    auc_pr = auc(recall, precision)
                    plt.plot(recall, precision, label=f"{name} (AUC-PR={auc_pr:.2f})")

        if pred_probs_df is not None:
            for col in pred_probs_df.columns:
                y_score = pred_probs_df[col].values
                precision, recall, _ = precision_recall_curve(y_valid, y_score)
                auc_pr = auc(recall, precision)
                plt.plot(recall, precision, label=f"{col} (AUC-PR={auc_pr:.2f})")

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend(loc="lower left")

        title = (
            f"{custom_name} - Precision-Recall Curve"
            if custom_name
            else "Precision-Recall Curve"
        )

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
        use_optimal_threshold=False,
    ):
        """
        Plot a confusion matrix from predicted probabilities or model outputs.

        Parameters
        ----------
        df : pd.DataFrame, optional
            DataFrame containing actual and predicted probability columns.
        outcome_cols : list of str, optional
            Column names for actual binary outcomes in `df`.
        pred_cols : list of str, optional
            Column names for predicted probabilities in `df`.
        models : dict, optional
            Dictionary of trained models with `.predict_proba()` or `.predict()`
            methods.
        X_valid : pd.DataFrame, optional
            Validation features to use for model predictions.
        y_valid : array-like, optional
            True labels corresponding to `X_valid`.
        threshold : float, default=0.5
            Threshold to binarize predicted probabilities when `optimal_threshold`
            is False.
        custom_name : str, optional
            Custom name to use in the plot title.
        model_name : str, optional
            Key to select a specific model from `models`.
        normalize : {'true', 'pred', 'all'}, optional
            Normalization method for the confusion matrix.
        cmap : str, default='Blues'
            Matplotlib colormap for the heatmap.
        show : bool, default=True
            Whether to display the plot immediately.
        use_optimal_threshold : bool, default=False
            If True, uses model's `predict(..., optimal_threshold=True)` method
            instead of manual thresholding.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The resulting matplotlib figure object.
        """

        fig, ax = plt.subplots(figsize=(8, 8))
        title = None

        if outcome_cols and pred_cols and df is not None:
            for outcome_col, pred_col in zip(outcome_cols, pred_cols):
                y_true = df[outcome_col]
                if use_optimal_threshold and hasattr(model, "predict"):
                    y_pred = model.predict(X_valid, optimal_threshold=True)
                else:
                    y_pred = (df[pred_col] > threshold).astype(int)
                cm = confusion_matrix(y_true, y_pred, normalize=normalize)
                cm = model.conf_mat
                # model.conf_mat_class_kfold(model, X_valid, y_valid, use_optimal_threshold)
                disp = ConfusionMatrixDisplay(
                    confusion_matrix=cm,
                    display_labels=[0, 1],
                )
                disp.plot(ax=ax, cmap=cmap, colorbar=False)

                # Add TP, FP, TN, FN labels
                labels = [["TN", "FP"], ["FN", "TP"]]
                # Normalize for brightness scaling
                norm_cm = cm.astype(float) / cm.max()

                for i in range(2):
                    for j in range(2):
                        # Get colormap color
                        color = plt.get_cmap(cmap)(norm_cm[i, j])
                        brightness = (
                            color[0] * 0.299 + color[1] * 0.587 + color[2] * 0.114
                        )  # Grayscale brightness
                        text_color = (
                            "white" if brightness < 0.5 else "black"
                        )  # Adaptive text color

                        ax.text(
                            j,
                            i - 0.15,
                            labels[i][j],  # Position slightly above the number
                            ha="center",
                            va="center",
                            fontsize=12,
                            color=text_color,
                        )

        if models and X_valid is not None and y_valid is not None:
            if model_name:
                model = models[model_name]
                print(model.conf_mat)
                return
                if use_optimal_threshold:
                    y_pred = model.predict(X_valid, optimal_threshold=True)
                else:
                    y_pred = (model.predict_proba(X_valid)[:, 1] > threshold).astype(
                        int
                    )
                cm = confusion_matrix(y_valid, y_pred, normalize=normalize)
                cm = model.conf_mat
                disp = ConfusionMatrixDisplay(
                    confusion_matrix=cm, display_labels=[0, 1]
                )
                disp.plot(ax=ax, cmap=cmap, colorbar=False)

                # Add TP, FP, TN, FN labels
                labels = [["TN", "FP"], ["FN", "TP"]]
                norm_cm = cm.astype(float) / cm.max()

                for i in range(2):
                    for j in range(2):
                        color = plt.get_cmap(cmap)(norm_cm[i, j])
                        brightness = (
                            color[0] * 0.299 + color[1] * 0.587 + color[2] * 0.114
                        )
                        text_color = "white" if brightness < 0.5 else "black"

                        ax.text(
                            j,
                            i - 0.15,
                            labels[i][j],
                            ha="center",
                            va="center",
                            fontsize=12,
                            color=text_color,
                        )
            else:
                for _, model in models.items():
                    y_pred = (model.predict_proba(X_valid)[:, 1] > threshold).astype(
                        int
                    )
                    cm = confusion_matrix(
                        y_valid,
                        y_pred,
                        normalize=normalize,
                    )
                    disp = ConfusionMatrixDisplay(
                        confusion_matrix=cm,
                        display_labels=[0, 1],
                    )
                    disp.plot(ax=ax, cmap=cmap, colorbar=False)

                    # Add TP, FP, TN, FN labels
                    labels = [["TN", "FP"], ["FN", "TP"]]
                    norm_cm = cm.astype(float) / cm.max()

                    for i in range(2):
                        for j in range(2):
                            color = plt.get_cmap(cmap)(norm_cm[i, j])
                            brightness = (
                                color[0] * 0.299 + color[1] * 0.587 + color[2] * 0.114
                            )
                            text_color = "white" if brightness < 0.5 else "black"

                            ax.text(
                                j,
                                i - 0.15,
                                labels[i][j],
                                ha="center",
                                va="center",
                                fontsize=12,
                                color=text_color,
                            )

        title = (
            f"{custom_name} - Confusion Matrix" if custom_name else "Confusion Matrix"
        )

        title += f" Threshold = {threshold}"

        plt.title(title)
        self._save_plot(title)
        if show:
            plt.show()

        return fig

    def plot_calibration_curve(
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
        n_bins=10,
        show=True,
    ):
        """
        Plot calibration curves to assess the agreement between predicted
        probabilities and actual outcomes.

        Parameters
        ----------
        df : pd.DataFrame, optional
            DataFrame containing actual and predicted probability columns.
        outcome_cols : list of str, optional
            Column names for actual binary outcomes in `df`.
        pred_cols : list of str, optional
            Column names for predicted probabilities in `df`.
        models : dict, optional
            Dictionary of trained models with `.predict_proba()` methods.
        X_valid : pd.DataFrame, optional
            Validation features for generating model predictions.
        y_valid : array-like, optional
            True binary labels corresponding to `X_valid`.
        pred_probs_df : pd.DataFrame, optional
            DataFrame of predicted probabilities (one column per model or method).
        model_name : str, optional
            Key to select a specific model from `models`.
        custom_name : str, optional
            Custom title to display on the plot.
        n_bins : int, default=10
            Number of bins to use when grouping predicted probabilities for
            calibration.
        show : bool, default=True
            Whether to display the plot immediately.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The resulting matplotlib figure object.
        """

        fig, _ = plt.subplots(figsize=(8, 8))

        # Handle predictions and true labels
        if df is not None and outcome_cols and pred_cols:
            for outcome_col, pred_col in zip(outcome_cols, pred_cols):
                y_true = df[outcome_col]
                y_prob = df[pred_col]
                frac_pos, mean_pred = calibration_curve(
                    y_true,
                    y_prob,
                    n_bins=n_bins,
                )
                brier = brier_score_loss(y_true, y_prob)
                plt.plot(
                    mean_pred,
                    frac_pos,
                    marker="o",
                    label=f"{pred_col} (Brier={brier:.3f})",
                )

        elif models and X_valid is not None and y_valid is not None:
            if model_name:
                y_prob = models[model_name].predict_proba(X_valid)[:, 1]
                frac_pos, mean_pred = calibration_curve(
                    y_valid,
                    y_prob,
                    n_bins=n_bins,
                )
                brier = brier_score_loss(y_valid, y_prob)
                plt.plot(
                    mean_pred,
                    frac_pos,
                    marker="o",
                    label=f"{model_name} (Brier={brier:.3f})",
                )
            else:
                for name, model in models.items():
                    y_prob = model.predict_proba(X_valid)[:, 1]
                    frac_pos, mean_pred = calibration_curve(
                        y_valid,
                        y_prob,
                        n_bins=n_bins,
                    )
                    brier = brier_score_loss(y_valid, y_prob)
                    plt.plot(
                        mean_pred,
                        frac_pos,
                        marker="o",
                        label=f"{name} (Brier={brier:.3f})",
                    )

        elif pred_probs_df is not None and y_valid is not None:
            for col in pred_probs_df.columns:
                y_prob = pred_probs_df[col].values
                frac_pos, mean_pred = calibration_curve(
                    y_valid,
                    y_prob,
                    n_bins=n_bins,
                )
                brier = brier_score_loss(y_valid, y_prob)
                plt.plot(
                    mean_pred,
                    frac_pos,
                    marker="o",
                    label=f"{col} (Brier={brier:.3f})",
                )

        # Perfect calibration line
        plt.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated")
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Fraction of Positives")
        plt.legend(loc="lower right")

        # Set title with custom_name or default to "Calibration Curve"
        title = custom_name if custom_name else "Calibration Curve"
        plt.title(title)
        self._save_plot(title)
        if show:
            plt.show()

        return fig

    def plot_metrics_vs_thresholds(
        self,
        models=None,
        X_valid=None,
        y_valid=None,
        df=None,
        outcome_cols=None,
        pred_cols=None,
        pred_probs_df=None,
        model_name=None,
        custom_name=None,
        scoring=None,
        show=True,
    ):
        """
        Plot Precision, Recall, F1 Score, and Specificity against thresholds,
        automatically marking the optimal threshold from the model.

        Parameters:
        -----------
        models : dict, optional
            Dictionary of model names and their fitted instances.
        X_valid : array-like, optional
            Validation features for the models.
        y_valid : array-like or pandas.Series, optional
            True labels for validation data.
        df : pandas.DataFrame, optional
            DataFrame containing true outcomes and predicted probabilities.
        outcome_cols : list, optional
            Column names in df for true outcomes.
        pred_cols : list, optional
            Column names in df for predicted probabilities.
        pred_probs_df : pandas.DataFrame, optional
            DataFrame with precomputed predicted probabilities.
        model_name : str, optional
            Specific model name to plot.
        custom_name : str, optional
            Custom name for the plot title.
        show : bool, optional (default=True)
            Whether to display the plot.

        Returns:
        --------
        fig : matplotlib.figure.Figure
            The generated figure object.
        """

        fig, ax = plt.subplots(figsize=(10, 6))
        title = custom_name or "Precision, Recall, F1, Specificity vs. Thresholds"

        def plot_curves(
            y_true,
            y_pred_probs,
            threshold,
            label_prefix="",
        ):
            precision, recall, thresholds = precision_recall_curve(
                y_true,
                y_pred_probs,
            )
            # Avoid div by zero
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
            fpr, _, roc_thresholds = roc_curve(y_true, y_pred_probs)
            specificity = 1 - fpr

            ax.plot(
                thresholds,
                f1_scores[:-1],
                label=f"{label_prefix}F1 Score",
                color="red",
            )
            ax.plot(
                thresholds,
                recall[:-1],
                label=f"{label_prefix}Recall",
                color="green",
            )
            ax.plot(
                thresholds,
                precision[:-1],
                label=f"{label_prefix}Precision",
                color="blue",
            )
            ax.plot(
                roc_thresholds,
                specificity,
                label=f"{label_prefix}Specificity",
                color="purple",
            )

            # Add vertical line for the model's threshold
            ax.axvline(
                x=float(threshold),  # Ensure it's a float
                color="black",
                linestyle="--",
                linewidth=2,
                label=f"{label_prefix}Threshold ({float(threshold):.2f})",
            )

        # Case 1: Direct model predictions
        if models and X_valid is not None and y_valid is not None:
            y_valid = y_valid.squeeze()
            if model_name:
                model = models[model_name]
                # Get the threshold dictionary
                threshold_dict = getattr(model, "threshold", {})
                # Extract using `scoring`
                threshold = float(threshold_dict.get(scoring, 0.5))
                plot_curves(
                    y_valid,
                    model.predict_proba(X_valid)[:, 1],
                    threshold,
                    label_prefix=f"{model_name} ",
                )
            else:
                for name, model in models.items():
                    threshold_dict = getattr(model, "threshold", {})
                    # Extract using `scoring`
                    threshold = float(threshold_dict.get(scoring, 0.5))
                    plot_curves(
                        y_valid,
                        model.predict_proba(X_valid)[:, 1],
                        threshold,
                        label_prefix=f"{name} ",
                    )

        # Case 2: Provided dataframe with outcome/prediction columns
        # (defaults to 0.5)
        elif df is not None and outcome_cols and pred_cols:
            for outcome_col, pred_col in zip(outcome_cols, pred_cols):
                plot_curves(
                    df[outcome_col],
                    df[pred_col],
                    threshold=0.5,
                    label_prefix=f"{pred_col} ",
                )

        # Case 3: Precomputed prediction probabilities DataFrame with y_valid
        # (defaults to 0.5)
        elif pred_probs_df is not None and y_valid is not None:
            y_valid = y_valid.squeeze()
            for col in pred_probs_df.columns:
                plot_curves(
                    y_valid,
                    pred_probs_df[col].values,
                    threshold=0.5,
                    label_prefix=f"{col} ",
                )

        ax.set_title(title)
        ax.set_xlabel("Thresholds")
        ax.set_ylabel("Metrics")
        ax.legend(loc="best")
        ax.grid()

        if show:
            plt.show()

        return fig


################################################################################
####################### MLFlow Models and Artifacts ############################
################################################################################


######################### MlFLow Helper Functions ##############################
def set_or_create_experiment(experiment_name, verbose=True):
    """
    Set up or create an MLflow experiment.

    Args:
        experiment_name: Name of the experiment.

    Returns:
        Experiment ID.
    """

    existing_experiment = mlflow.get_experiment_by_name(experiment_name)
    if existing_experiment is None:
        print(f"Experiment '{experiment_name}' does not exist. Creating a new one.")
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = existing_experiment.experiment_id
        if verbose:
            print(f"Using Existing Experiment_ID: {experiment_id}")
    mlflow.set_experiment(experiment_name)
    return experiment_id


def start_new_run(run_name):
    """
    Start a new MLflow run with the given name.

    Args:
        run_name: Name of the run.

    Returns:
        Run ID of the newly started run.
    """
    run = mlflow.start_run(run_name=run_name)
    run_id = run.info.run_id
    mlflow.end_run()
    print(f"Starting New Run_ID: {run_id} for {run_name}")
    return run_id


def get_run_id_by_name(experiment_name, run_name, verbose=True):
    """
    Query MLflow to find the run_id for the given run_name in the experiment.
    If no run exists, create a new one.

    Args:
        experiment_name: Name of the MLflow experiment.
        run_name: Name of the run to search for or create.

    Returns:
        Run ID of the most recent run matching the run_name, or a new run ID
        if none exists.
    """
    client = MlflowClient()

    # Get the experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(f"Experiment {experiment_name} not found.")

    # Search for existing runs
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        order_by=["start_time DESC"],  # Get the most recent run
    )

    if runs:
        run_id = runs[0].info.run_id  # Use the latest run_id for this run_name
        if verbose:
            print(
                f"Found Run_ID: {run_id} for run_name '{run_name}' in experiment '{experiment_name}'"
            )
    else:
        # No runs found, create a new one
        if verbose:
            print(
                f"No runs found with run_name '{run_name}' in experiment '{experiment_name}'. Creating a new run."
            )
        run_id = start_new_run(run_name)

    return run_id


################## Dump artificats (e.g. to preprocessing) #####################


def mlflow_dumpArtifact(
    experiment_name,
    run_name,
    obj_name,
    obj,
    get_existing_id=True,
    artifact_run_id=None,
    artifacts_data_path=mlflow_artifacts_data,
):
    """
    Log an object as an MLflow artifact with a persistent run ID.

    Args:
        experiment_name: Name of the MLflow experiment.
        run_name: Name of the run within the experiment.
        obj_name: Name of the artifact (without .pkl extension).
        obj: Object to serialize and log.
        get_existing_id: If True, try to reuse an existing run ID (default: True).
        artifact_run_id: Specific run ID to use (optional).
        artifacts_data_path: Path to MLflow artifacts directory
        (default: mlflow_artifacts_data from constants).

    Returns:
        None
    """

    # Initialize or reuse the artifacts_run_id as a function attribute
    if not hasattr(mlflow_dumpArtifact, "artifacts_run_id"):
        mlflow_dumpArtifact.artifacts_run_id = None
    else:
        mlflow_dumpArtifact.artifacts_run_id = artifact_run_id
    abs_mlflow_data = os.path.abspath(artifacts_data_path)
    mlflow.set_tracking_uri(f"file://{abs_mlflow_data}")

    # Set or create experiment
    experiment_id = set_or_create_experiment(experiment_name)
    print(f"Experiment_ID for artifact {obj_name}: {experiment_id}")

    if get_existing_id:
        mlflow_dumpArtifact.artifacts_run_id = get_run_id_by_name(
            experiment_name,
            run_name,
        )

    # Get or create a single run_id for all artifacts
    if mlflow_dumpArtifact.artifacts_run_id:
        run_id = mlflow_dumpArtifact.artifacts_run_id
        print(f"Reusing Existing Artifacts Run_ID: {run_id} for {run_name}")
    else:
        run_id = start_new_run(run_name)
        # Store the run_id for future calls
        mlflow_dumpArtifact.artifacts_run_id = run_id

    with mlflow.start_run(run_id=run_id, nested=True):
        temp_file = f"{obj_name}.pkl"
        with open(temp_file, "wb") as f:
            pickle.dump(obj, f)
        mlflow.log_artifact(temp_file)
        os.remove(temp_file)

    print(f"Artifact {obj_name} logged successfully in MLflow under Run_ID: {run_id}.")
    return None


################# Load artificats (e.g. from preprocessing) ####################


def mlflow_loadArtifact(
    experiment_name,
    run_name,  # Use run_name to query the single artifacts run_id
    obj_name,
    verbose=True,
    artifacts_data_path=mlflow_artifacts_data,
):
    """
    Load an object from MLflow artifacts by experiment and run name.

    Args:
        experiment_name: Name of the MLflow experiment.
        run_name: Name of the run within the experiment.
        obj_name: Name of the artifact (without .pkl extension).

    Returns:
        Deserialized object from the artifact.

    Raises:
        ValueError: If experiment or run is not found.
    """
    abs_mlflow_data = os.path.abspath(artifacts_data_path)
    mlflow.set_tracking_uri(f"file://{abs_mlflow_data}")

    set_or_create_experiment(experiment_name, verbose=verbose)

    # Get the run_id using the helper function
    run_id = get_run_id_by_name(experiment_name, run_name, verbose=verbose)

    # Download the artifact from the run's artifact directory
    client = MlflowClient()

    local_path = client.download_artifacts(run_id, f"{obj_name}.pkl")
    with open(local_path, "rb") as f:
        obj = pickle.load(f)
    return obj


################### Return model metrics to be used in MlFlow ##################


def return_model_metrics(
    inputs: dict,
    model,
    estimator_name,
    return_dict: bool = False,
) -> pd.Series:
    """
    Compute and return model performance metrics for multiple input types.

    Parameters:
    ----------
    inputs : dict
        A dictionary where keys are dataset names (e.g., "train", "test") and
        values are tuples containing feature matrices (X) and target arrays (y).
    model : object
        A model instance with a `return_metrics` method that computes evaluation
        metrics.
    estimator_name : str
        The name of the estimator to label the output.

    Returns:
    -------
    pd.Series
        A Series containing the computed metrics, indexed by input type and
        metric name.
    """

    all_metrics = []
    for input_type, (X, y) in inputs.items():
        print(input_type)
        return_metrics_dict = model.return_metrics(
            X,
            y,
            optimal_threshold=True,
            print_threshold=True,
            model_metrics=True,
            return_dict=return_dict,
        )

        metrics = pd.Series(return_metrics_dict).to_frame(estimator_name)
        metrics = round(metrics, 3)
        metrics.index = [input_type + " " + ind for ind in metrics.index]
        all_metrics.append(metrics)
    return pd.concat(all_metrics)


####################### Enter the model plots into MlFlow ######################


def return_model_plots(
    inputs,
    model,
    estimator_name,
    scoring,
):
    """
    Generate evaluation plots for a given model on multiple input datasets.

    Parameters:
    ----------
    inputs : dict
        A dictionary where keys are dataset names (e.g., "train", "test") and
        values are tuples containing feature matrices (X) and target arrays (y).
    model : object
        A trained model with a `threshold` attribute used for evaluation.
    estimator_name : str
        The name of the estimator to label the plots.

    Returns:
    -------
    dict
        A dictionary mapping plot filenames to generated plots, including:
        - ROC curves (`roc_{input_type}.png`)
        - Confusion matrices (`cm_{input_type}.png`)
        - Precision-recall curves (`pr_{input_type}.png`)
    """

    all_plots = {}
    plotter = PlotMetrics()
    for input_type, (X, y) in inputs.items():
        all_plots[f"roc_{input_type}.png"] = plotter.plot_roc(
            models={estimator_name: model},
            X_valid=X,
            y_valid=y,
            custom_name=estimator_name,
            show=False,
        )
        all_plots[f"cm_{input_type}.png"] = plotter.plot_confusion_matrix(
            models={estimator_name: model},
            X_valid=X,
            y_valid=y,
            threshold=next(iter(model.threshold.values())),
            custom_name=estimator_name,
            show=False,
            use_optimal_threshold=True,
        )

        all_plots[f"pr_{input_type}.png"] = plotter.plot_precision_recall(
            models={estimator_name: model},
            X_valid=X,
            y_valid=y,
            custom_name=estimator_name,
            show=False,
        )

        all_plots[f"calib_{input_type}.png"] = plotter.plot_calibration_curve(
            models={estimator_name: model},
            X_valid=X,
            y_valid=y,
            custom_name=f"{estimator_name} - Calibration Curve",
            show=False,
        )

        all_plots[f"metrics_thresh_{input_type}.png"] = (
            plotter.plot_metrics_vs_thresholds(
                models={estimator_name: model},
                X_valid=X,
                y_valid=y,
                custom_name=f"{estimator_name} - Precision, Recall, F1 Score, Specificity vs. Thresholds",
                scoring=scoring,
                show=False,
            )
        )

    return all_plots


def mlflow_log_parameters_model(
    model_type: str = None,
    n_iter: int = None,
    kfold: bool = None,
    outcome: str = None,
    run_name: str = None,
    experiment_name: str = None,
    model_name: str = None,
    model=None,
    hyperparam_dict=None,
):
    """
    Log model-specific parameters, hyperparameters from a dictionary, and the
    trained model under a single MLflow run in mlruns/modeling.

    Args:
        model_type: Type of the model (e.g., 'lr', 'rf', 'xgb', 'cat').
        n_iter: Number of iterations for hyperparameter tuning.
        kfold: Whether k-fold cross-validation was used.
        outcome: Target variable name.
        run_name: Name of the MLflow run.
        experiment_name: Name of the MLflow experiment.
        model_name: Name for the logged model artifact.
        model: The trained model object.
        hyperparam_dict: Dictionary of hyperparameters to loop through and log
        (default None).
    """

    abs_mlflow_data = os.path.abspath(mlflow_models_data)
    # mlflow.set_tracking_uri(f"file://{abs_mlflow_data}")
    mlflow.set_tracking_uri("https://mlflow-circ-milan.onrender.com")

    # Set or create the experiment_id for the model and parameters
    experiment_id = set_or_create_experiment(experiment_name)
    run_id = get_run_id_by_name(experiment_name, run_name)

    print(f"Experiment_ID for model {model_type} and parameters: {experiment_id}")

    with mlflow.start_run(experiment_id=experiment_id, run_id=run_id) as run:
        print(f"experiment_id={experiment_id}, run_id={run_id}")

        # Log parameters under the active run
        if model_type is not None:
            mlflow.log_param("model_type", model_type)
        if n_iter is not None:
            mlflow.log_param("n_iter", n_iter)
        if kfold is not None:
            mlflow.log_param("kfold", kfold)
        if outcome is not None:
            mlflow.log_param("outcome", outcome)

        # Logging best model hyperparameters
        if hyperparam_dict is not None:
            mlflow.log_params(hyperparam_dict)

        # Logging model
        mlflow.sklearn.log_model(
            model,
            model_name,
        )

        print("Parameters and model logged successfully in MLflow.")

    return None


########################## Load the model object ###############################


def mlflow_load_model(
    experiment_name,
    run_name,
    model_name,
    mlruns_location: str = None,
):
    """
    Load a scikit-learn model from MLflow by experiment and run name.

    Args:
        experiment_name: Name of the MLflow experiment.
        run_name: Name of the run within the experiment.
        model_name: Name of the model artifact.

    Returns:
        Scikit-learn model instance.

    Raises:
        ValueError: If experiment or run is not found.
    """
    if mlruns_location is None:
        abs_mlflow_data = os.path.abspath(mlflow_models_data)
    else:
        abs_mlflow_data = os.path.abspath(mlruns_location)
    # mlflow.set_tracking_uri(f"file://{abs_mlflow_data}")
    mlflow.set_tracking_uri("https://mlflow-circ-milan.onrender.com")

    # Query MLflow to find the latest run_id for the given run_name in the
    # experiment (for models)
    client = MlflowClient()
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(f"Experiment {experiment_name} not found.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        order_by=["start_time DESC"],  # Get the most recent run
    )

    if not runs:
        raise ValueError(
            f"No runs found with run_name '{run_name}' in experiment '{experiment_name}'."
        )

    run_id = runs[
        0
    ].info.run_id  # Use the latest run_id for this run_name (for the specific model)

    # Load the scikit-learn model
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/{model_name}")
    return model


########################### MLFlow Model Evaluation ############################


def log_mlflow_metrics(
    experiment_name,
    run_name,
    metrics=None,
    images={},
):
    """
    Logs experiment metrics and visualizations to MLflow.

    This function sets up an MLflow experiment, retrieves the appropriate run,
    and logs key metrics and visual artifacts for model evaluation.

    Parameters:
    -----------
    - experiment_name (str):
        The name of the MLflow experiment.
    - run_name (str):
        The name of the specific run within the experiment.
    - metrics (pd.Series, optional):
        A Pandas Series containing performance metrics (e.g., precision,
        recall, F1-score). Each metric is logged individually.
    - images (dict, optional):
        A dictionary where keys are filenames and values are Matplotlib figure
        objects. These visualizations are logged to MLflow as artifacts.
    Returns:
    --------
    None
    """

    # Set the tracking URI to the specified mlflow_data location
    abs_mlflow_data = os.path.abspath(mlflow_models_data)  # Use models path
    # mlflow.set_tracking_uri(f"file://{abs_mlflow_data}")
    mlflow.set_tracking_uri("https://mlflow-circ-milan.onrender.com")

    # Set or create the experiment_id for the model and parameters
    experiment_id = set_or_create_experiment(experiment_name)
    run_id = get_run_id_by_name(experiment_name, run_name)

    # Iterate over the models and log their metrics and parameters
    with mlflow.start_run(experiment_id=experiment_id, run_id=run_id):

        # Extract the row for the current model
        if metrics is not None:
            result = metrics
            if not result.empty:

                # Log the parameters and metrics
                for col in result.index:
                    value = result[col]
                    mlflow.log_metric(col, float(value))

        for name, image in images.items():
            mlflow.log_figure(image, name)


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
    else:
        abs_mlflow_data = os.path.abspath(mlruns_location)
    # mlflow.set_tracking_uri(f"file://{abs_mlflow_data}")
    mlflow.set_tracking_uri("https://mlflow-circ-milan.onrender.com")

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


#################################################################################
######################### Crosstab Plotting Function ############################
#################################################################################


def crosstab_plot(
    df,
    list_name,
    outcome,
    bbox_to_anchor,
    w_pad,
    h_pad,
    figsize=(12, 8),
    label_fontsize=12,
    tick_fontsize=10,
    n_rows=None,
    n_cols=None,
    label1=None,
    label2=None,
    normalize=False,
    show_value_counts=False,
    color_schema=None,
    save_plots=False,
    image_path_png=None,
    image_path_svg=None,
    string=None,
):
    """
    Generates crosstab bar plots visualizing the relationship between an outcome
    variable and multiple categorical variables.

    Color control (via color_schema):
      - None: default two-tone plus extras.
      - list/tuple: applies to all subplots, repeating if needed.
      - dict: map column names to specific color lists.

    Parameters:
    - df: pandas.DataFrame to plot.
    - list_name: list of str, categorical columns to include.
    - outcome: str, name of the outcome column.
    - bbox_to_anchor: tuple, legend anchor coordinates.
    - w_pad, h_pad: floats, padding for tight_layout.
    - figsize: tuple, figure size.
    - label_fontsize: int, font size for axis labels, titles, legend.
    - tick_fontsize: int, font size for tick labels.
    - n_rows, n_cols: ints, optional grid rows/columns; auto-calculated if omitted.
    - label1, label2: str, custom x-axis labels for outcome levels.
    - normalize: bool, False for raw counts, True for normalized proportions.
    - show_value_counts: bool, append counts or percentages to legend entries.
    - color_schema: None, list, or dict for color customization.
    - save_plots: bool, save figures if True.
    - image_path_png, image_path_svg: str paths for saving.
    - string: base filename for saving (sanitized).
    """
    n_vars = len(list_name)

    # Determine grid size
    if n_rows is None and n_cols is None:
        n_cols = math.ceil(math.sqrt(n_vars))
        n_rows = math.ceil(n_vars / n_cols)
    elif n_rows is None:
        n_rows = math.ceil(n_vars / n_cols)
    elif n_cols is None:
        n_cols = math.ceil(n_vars / n_rows)

    # Default color schemes
    default_colors = ["#00BFC4", "#F8766D"]
    extra_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for item, ax in zip(list_name, axes[:n_vars]):
        # Compute crosstab
        if not normalize:
            ylabel = "Count"
            ctab = pd.crosstab(df[outcome], df[item])
        else:
            ylabel = "Percentage"
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%")
            )
            ctab = pd.crosstab(df[outcome], df[item], normalize="index")

        # Select colors
        n_cats = len(ctab.columns)
        if isinstance(color_schema, dict) and item in color_schema:
            scheme = color_schema[item]
        elif isinstance(color_schema, (list, tuple)):
            scheme = list(color_schema)
        else:
            scheme = default_colors.copy()
        if len(scheme) < n_cats:
            needed = n_cats - len(scheme)
            pool = extra_colors if color_schema is None else scheme
            scheme = (scheme + pool * math.ceil(needed / len(pool)))[:n_cats]

        # Plot
        ctab.plot(kind="bar", stacked=True, rot=0, ax=ax, color=scheme)

        # Formatting
        ax.set_ylabel(ylabel, fontsize=label_fontsize)
        ax.set_xlabel("Outcome", fontsize=label_fontsize)
        ax.tick_params(labelsize=tick_fontsize)
        if label1 and label2:
            ax.set_xticklabels([label1, label2], fontsize=tick_fontsize)
        ax.set_title(f"Outcome vs. {item}", fontsize=label_fontsize)

        # Legend labels: counts or percentages
        labels = [str(col) for col in ctab.columns]
        if show_value_counts:
            if not normalize:
                counts = df[item].value_counts()
                labels = [
                    f"{lbl} (n={counts.get(col, 0)})"
                    for lbl, col in zip(labels, ctab.columns)
                ]
            else:
                proportions = df[item].value_counts(normalize=True)
                labels = [
                    f"{lbl} ({proportions.get(col, 0)*100:.0f}%)"
                    for lbl, col in zip(labels, ctab.columns)
                ]

        handles, _ = ax.get_legend_handles_labels()
        leg = ax.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=bbox_to_anchor,
            ncol=1,
            fontsize=label_fontsize,
        )
        for text in leg.get_texts():
            text.set_fontsize(label_fontsize)

    # Hide extra subplots
    for ax in axes[n_vars:]:
        ax.axis("off")

    plt.tight_layout(w_pad=w_pad, h_pad=h_pad)

    # Save logic
    if save_plots:
        base = string or "crosstab_plot"
        safe = base.replace(" ", "_").replace(":", "").lower()
        if image_path_png:
            plt.savefig(
                os.path.join(image_path_png, f"{safe}.png"), bbox_inches="tight"
            )
        if image_path_svg:
            plt.savefig(
                os.path.join(image_path_svg, f"{safe}.svg"), bbox_inches="tight"
            )

    plt.show()
