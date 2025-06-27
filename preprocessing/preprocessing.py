################################################################################
# Step 1. Load Dependencies and Configure Display Settings
################################################################################
import os
import typer
import pandas as pd
import numpy as np
import eda_toolkit
from eda_toolkit import dataframe_profiler, add_ids
import sys

from model_tuner.pickleObjects import dumpObjects  # import pickling scripts

from constants import (
    var_index,
    shape_var_thresh,
    target_outcome,
    preproc_run_name,
    exp_artifact_name,
)

# import all user-defined functions and constants
from functions import (
    terminal_width,
    mlflow_dumpArtifact,
    mlflow_loadArtifact,
    safe_to_numeric,
)

from project_functions import HealthMetrics

pd.set_option("display.max_columns", None)
pd.set_option("display.width", terminal_width)
pd.set_option("display.float_format", "{:.3f}".format)

app = typer.Typer()


@app.command()
def main(
    input_data_file: str = "./data/raw/df.parquet",
    output_eda_file: str = "./data/processed/circ_eda.parquet",
    output_data_file: str = "./data/processed/df_sans_zero.parquet",
    stage: str = "training",
    data_path: str = "./data/processed",
):
    """
    Main script execution replacing sys.argv with typer.

    Args:
        input_data_file (str): Path to the input parquet file.
        output_data_file (str): Path to save the processed parquet file.
        stage (str): Processing stage (e.g., 'training' or 'inference').
    """
    ################################################################################
    # Step 2. Display EDA Toolkit Info and Load Input Data
    ################################################################################

    print(f"\nThis project uses Python {sys.version.split()[0]}.")
    print(f"EDA Toolkit Version: {eda_toolkit.__version__}")
    print(f"EDA Toolkit Authors: {eda_toolkit.__author__}\n")

    df = pd.read_excel(input_data_file)

    ################################################################################
    # Step 3. Assign Anonymized IDs and Drop Original Index
    ################################################################################

    # Create anonymized patient ID and set as index
    df = add_ids(
        df=df,
        id_colname=var_index,
        num_digits=9,
        seed=222,
        set_as_index=True,
    )

    # Drop the original ID column after index is set
    if var_index in df.columns:
        df.drop(columns=[var_index], inplace=True)

    ################################################################################
    # Step 4. Initial String Column Diagnostics (Training Only)
    ################################################################################
    # String columns are identified and should be removed before modeling
    # because machine learning models typically require numerical inputs.
    # Keeping string columns in the dataset may lead to errors or
    # unintended behavior unless explicitly encoded.
    #
    # To ensure consistency between training and inference,
    # we save the list of string columns and track it using MLflow.

    if stage == "training":

        df_object = df.select_dtypes("object")
        print(
            "\nThe following columns have strings and should be removed from "
            "modeling: \n \n"
            f"{df_object.columns.to_list()}. There are {df_object.shape[1]} "
            f"of them.\n"
        )

    ################################################################################
    # Step 5. Column Renaming, Display, and Basic Validation
    ################################################################################

    ## inspect dataframe (first 5 rows)
    print(df.head())
    columns_list = df.columns.to_list()  ## Print the DataFrame Columns to List

    print("*" * terminal_width)
    print("\nDataFrame Columns:\n")
    for col in columns_list:
        print(col)
    print("*" * terminal_width)

    ######################### Rename Columns for Posterity #########################
    # The raw data is messy, so an initial renaming of the columns needs to take
    # place s/t dataset can read more meaningfully to the end-user.

    df.rename(
        columns={
            "Age (y)": "Age_years",
            "Height (m)": "Height_m",
            "Weight (Kg)": "Weight_kg",
            "Geographical Origin": "Geographical_Origin",
            "Cultural / Religious affiliation": "Cultural_Religious_Affiliation",
            "Preoperative drugs (antibiotic)": "Preop_drugs_antibiotic",
            "Preoperative Blood Pressure (mmHg)": "Preop_Blood_Pressure_mmHg",
            "Preoperative Heart Rate (bpm)": "Preop_Heart_Rate_bpm",
            "Preoperative Pulse Oxymetry (%)": "Preop_Pulse_Ox_Percent",
            "Surgical Technique": "Surgical_Technique",
            "Anesthesia Type": "Anesthesia_Type",
            "Intraoperative drugs": "Intraoperative_drugs",
            "Intraoperative Blood Loss (ml)": "Intraoperative_Blood_Loss_ml",
            "Intraoperative Mean Blood Pressure (mmHg)": "Intraop_Mean_Blood_Pressure_mmHg",
            "Intraoperative Mean Heart Rate (bpm)": "Intraop_Mean_Heart_Rate_bpm",
            "Intraoperative Mean Pulse Oxymetry (%)": "Intraop_Mean_Pulse_Ox_Percent",
            "Surgical Time (min)": "Surgical_Time_min",
            "Functional Outcomes (pain)": "Functional_Outcomes_Pain",
            "Functional Outcomes (Bleeding)": "Functional_Outcomes_Bleeding",
            "Functional Outcomes (Edema)": "Functional_Outcomes_Edema",
            "Functional Outcomes (Infection)": "Functional_Outcomes_Infection",
            "Functional Outcomes (Fast Recovery)": "Functional_Outcomes_Fast_Recovery",
            "Functional Outcomes (Cosmetic Satisfaction)": "Functional_Outcomes_Cosmetic_Satisfaction",
            "Cost of Procedure (€)": "Cost_of_Procedure_euros",
            "Cost Type": "Cost_Type",
        },
        inplace=True,
    )

    # Any binary column with the "Functional_" prefix gets distributions printed
    functional = df[[col for col in df.columns if "Functional_" in col]]

    for column in functional.columns:
        print(f"\n{functional[column].value_counts()}")

    ############################### Spelling Checks ################################
    # In the event that there are any mispelled country names, they will be corrected.

    # Correct the spelling for Morocco with error handling
    try:
        df["Geographical_Origin"] = df["Geographical_Origin"].replace(
            {"Marocco": "Morocco"}
        )
    except Exception as e:
        print(f"An error occurred while replacing values in 'Geographical_Origin': {e}")

    print(f"\nUnique Geographical Origins:\n" f"{df['Geographical_Origin'].unique()}\n")

    # check unique countries in the series.

    ################################################################################
    # Step 6. Ensure Numeric Data and Feature Engineering
    ################################################################################
    # Convert any possible numeric values that may have been incorrectly
    # classified as non-numeric. This avoids accidental labeling errors.
    # Perform necessary feature transformations (if and as applicable), such as:
    # - Deriving weight in pounds from kilograms
    # - Calculating height in feet using BMI and weight
    # - Dropping redundant features to prevent overfitting
    ########################################################################

    # Convert possible numeric columns to actual numeric types
    df = df.apply(lambda x: safe_to_numeric(x))

    ################################################################################
    # Step 7. BMI Calculation and Categorization
    ################################################################################

    # The original BMI column in the raw data is either missing or unreliable.
    # Here, we recalculate BMI using the standard formula:
    # BMI = weight (kg) / height (m)^2
    # We use our helper class `HealthMetrics` to encapsulate the logic and handle
    # edge cases.
    HealthMetrics.calculate_bmi(
        df=df,
        weight_col="Weight_kg",
        height_col="Height_m",
        weight_unit="kg",
        height_unit="m",
    )

    # Next, we categorize BMI into standard categories (e.g., Underweight, Normal,
    # Overweight, Obese) based on WHO guidelines or similar thresholds. This adds
    # a new `BMI_Category` column.
    HealthMetrics.calculate_bmi_category(df)

    # Print a preview of the new BMI-related fields to verify correctness
    print(f"\nFirst 5 Rows of BMI Data: \n{df[['BMI', 'BMI_Category']].head()}\n")

    ################################################################################
    # Step 8. MAP (Mean Arterial Pressure) Calculation
    ################################################################################

    # The MAP (mean arterial pressure) is not provided directly but can be computed
    # from combined systolic/diastolic BP fields formatted as strings
    # (e.g., \"120/80\").The formula used is: MAP = (SBP + 2 * DBP) / 3
    # We apply the transformation twice: once for preoperative and once for
    # intraoperative values.
    HealthMetrics.calculate_map(
        df=df,
        map_col_name="Preop_MAP",
        combined_bp_col="Preop_Blood_Pressure_mmHg",
    )

    HealthMetrics.calculate_map(
        df=df,
        map_col_name="Intraop_MAP",
        combined_bp_col="Intraop_Mean_Blood_Pressure_mmHg",
    )

    ################################################################################
    # Step 9. One-Hot Encoding
    ################################################################################
    # one hot encode insurance categories (break them out as separate cols of 0, 1)

    df = df.assign(
        **pd.get_dummies(
            df[["Anesthesia_Type", "BMI_Category"]],
            dtype=int,
        )
    )

    ################################################################################
    # Step 10. Comorbidities and Surgical Techniques
    ################################################################################

    # Create a new column for comorbidity flag (1 if comorbidities exist else 0)
    df["Comorbidity_Flag"] = df["Comorbidities"].apply(lambda x: 0 if x == 0 else 1)

    # Calculate value counts and percentages
    value_counts = df["Comorbidities"].value_counts()
    percentages = df["Comorbidities"].value_counts(1) * 100

    # Combine into a DataFrame
    combined_counts = pd.DataFrame(
        {"Count": value_counts, "Percentage": percentages}
    ).reset_index()

    combined_counts.columns = ["Comorbidity", "Count", "Percentage"]

    print(combined_counts)
    print()
    print("*" * terminal_width)

    # Encode Surgical_Technique as binary: 0 = Traditional, 1 = Other
    df["Surgical_Technique"] = df["Surgical_Technique"].apply(
        lambda x: 0 if x == "Traditional" else 1
    )

    ################################################################################
    # Step 11. Split Combined Blood Pressure Columns into Systolic and Diastolic
    ################################################################################

    # The dataset includes preoperative and intraoperative blood pressure values as
    # combined strings in the format "SBP/DBP" (e.g., "120/80").
    # To support numerical analysis and clinical metric derivation, we split these
    # into two separate numeric columns: systolic and diastolic.

    # First, split intraoperative blood pressure:
    HealthMetrics.split_bp_column(
        df=df,
        combined_bp_col="Intraop_Mean_Blood_Pressure_mmHg",
        systolic_col_name="Intraop_SBP",
        diastolic_col_name="Intraop_DBP",
    )

    # Then, split preoperative blood pressure:
    HealthMetrics.split_bp_column(
        df=df,
        combined_bp_col="Preop_Blood_Pressure_mmHg",
        systolic_col_name="Preop_SBP",
        diastolic_col_name="Preop_DBP",
    )

    # After this step, we will have four new numeric columns:
    # - Preop_SBP, Preop_DBP
    # - Intraop_SBP, Intraop_DBP
    # These can be used for modeling or to calculate MAP if needed independently.

    ################################################################################
    # Step 12. Filter Out Pediatric Patients (Age < 18)
    ################################################################################

    # For this study, we are only interested in adult patients.
    # To ensure the cohort represents an adult surgical population,
    # we exclude all records where the patient is under 18 years old.
    # This is a common preprocessing step in clinical datasets
    # to maintain consistency and avoid skewing results due to
    # pediatric physiology.

    df = df[df["Age_years"] >= 18]

    ################################################################################
    # Step 13. Flag Patients with Diabetes Based on Comorbidity Text
    ################################################################################

    # The `Comorbidities` column includes text strings that describe patient health
    # conditions.To flag patients with diabetes, we search for any mention of 'DM'
    # (Diabetes Mellitus) in the text, regardless of case. The result is stored as
    # a binary variable:
    #     1 = Diabetes present
    #     0 = No mention of diabetes

    df["Diabetes"] = (
        df["Comorbidities"].astype(str).str.contains("DM", case=False).astype(int)
    )

    # Show distribution of the new Diabetes flag
    print(f"\nDiabetes Prevalence: \n{df['Diabetes'].value_counts()} \n")

    # Additionally, show a breakdown of all comorbidity descriptions:
    value_counts = df["Comorbidities"].value_counts()
    percentages = df["Comorbidities"].value_counts(normalize=True) * 100

    # Convert comorbidity column to numeric (optional step if numeric format
    # expected downstream)
    df["Comorbidities"] = pd.to_numeric(df["Comorbidities"], errors="coerce")

    # Combine counts and percentages into a single DataFrame for inspection
    combined_counts = pd.DataFrame(
        {"Count": value_counts, "Percentage": percentages}
    ).reset_index()

    combined_counts.columns = ["Comorbidity", "Count", "Percentage"]

    print(f"{combined_counts}\n")
    print("*" * terminal_width)

    ################################################################################
    # Step 14. Check for Missing Values Across the Dataset
    ################################################################################

    # We generate a full missing-value profile using the custom
    # `dataframe_profiler()` from the `eda_toolkit`. This function summarizes
    # null counts, data types, and other key metadata for each column.

    print("DataFrame Analysis Report (`dataframe_columns`) \n")
    df_columns = dataframe_profiler(df, return_df=True)

    # Display the profiler output for visual inspection
    print(f"DataFrame Analysis:\n{df_columns}\n")

    # If there are no missing values, provide a confirmation message.
    if df_columns[df_columns["null_total"] >= 1].empty:
        print("No Missing Values to Report\n")

    print("*" * terminal_width)

    ########################################################################
    # Step 15. String Columns Handling
    ########################################################################
    # String columns are identified and should be removed before modeling
    # because machine learning models typically require numerical inputs.
    # Keeping string columns in the dataset may lead to errors or
    # unintended behavior unless explicitly encoded.
    #
    # To ensure consistency between training and inference,
    # we save the list of string columns and track it using MLflow.
    ########################################################################
    if stage == "training":

        # Extract column names to a list
        string_cols_list = df_object.columns.to_list()

        ########################################################################
        # Step 16. Save and Log String Column List
        ########################################################################
        # Save the list of string columns for consistency across training and
        # inference and log them in MLflow for reproducibility.
        # This list of string columns is dumped (stored) only to inform of what
        # the string columns are; no further action is taken; we do not need to
        # load this list into production, since it is only there for us to
        # see what the columns are.
        ########################################################################

        # Dump the string_cols_list into a pickle file for future reference
        dumpObjects(
            string_cols_list,
            os.path.join(data_path, "string_cols_list.pkl"),
        )

        # Log the string column list as an artifact in MLflow
        mlflow_dumpArtifact(
            experiment_name=exp_artifact_name,
            run_name=preproc_run_name,
            obj_name="string_cols_list",
            obj=string_cols_list,
        )

    ################################################################################
    ###################### Re-engineering Selected Features ########################
    ################################################################################

    ################################################################################
    # Step 17. Low Variance Columns
    ################################################################################

    # Select only numeric columns s/t .var() can be applied since you can only
    # call this function on numeric columns; otherwise, if you include a mix
    # (object and numeric), it will throw the following FutureWarning:
    # Dropping of nuisance columns in DataFrame reductions
    # (with 'numeric_only=None') is deprecated; in a future version this will
    # raise TypeError.  Select only valid columns before calling the reduction.

    ################################################################################

    if stage == "training":
        # Extract numeric columns to compute variance and identify
        # low-variance features; check for variance
        shape_var = (
            df.var(numeric_only=True).to_frame().rename(columns={0: "Shape_Var"})
        )

        # Assuming df is your original DataFrame and shape_var is as shown in the screenshot
        low_variance_list = shape_var[
            shape_var["Shape_Var"] < shape_var_thresh
        ].index.to_list()

        ########################################################################
        # Step 18. Save and Log Low Variance Columns List
        ########################################################################
        # Save the list of string columns for consistency across training and
        # inference and log them in MLflow for reproducibility.
        ########################################################################

        dumpObjects(
            low_variance_list,
            os.path.join(data_path, "low_variance_list.pkl"),
        )

        mlflow_dumpArtifact(
            experiment_name=exp_artifact_name,
            run_name=preproc_run_name,
            obj_name="low_variance_list",
            obj=low_variance_list,
        )

    if stage == "inference":

        ########################################################################
        # Load Previously Saved Low Variance Columns List
        ########################################################################

        # load zero_var_list
        low_variance_list = mlflow_loadArtifact(
            experiment_name=exp_artifact_name,
            run_name=preproc_run_name,
            obj_name="low_variance_list",
        )

    ################################################################################
    # Step 19. Drop Zero-Variance Features
    ################################################################################

    # Now that we’ve identified columns with near-zero variance (from earlier steps),
    # we remove them from the dataset. These features are unlikely to be predictive
    # and may introduce noise or overfitting into models.

    print(f"\nDropping the following low variance column(s): {low_variance_list}\n")

    df_sans_zero = df.drop(columns=low_variance_list)

    # Display dimensionality reduction info
    print(f"Sans Low Var Shape: {df_sans_zero.shape}")
    print(f"Original shape: {df.shape[1]} columns.")
    print(f"Reduced by {df.shape[1] - df_sans_zero.shape[1]} low variance columns.")
    print(f"Now there are {df_sans_zero.shape[1]} columns.\n")

    ################################################################################
    # Step 20. Save Copy of Original (for EDA) Before Dropping Cost and Other Fields
    ################################################################################

    # Before additional column drops (e.g., cost-related or redundant columns),
    # we create a copy of the full dataset with original features retained.
    # This copy (`circ_eda`) is meant for exploratory data analysis and
    # record-keeping. Some fields like cost will later be removed from the
    # modeling dataset but may still be of analytical value.

    circ_eda = df.copy()

    ################################################################################
    # Step 21. Drop Uninformative or Redundant Columns
    ################################################################################

    # The following columns are dropped:
    # - `Weight_kg`: We retain BMI instead, which already incorporates height
    #    and weight.
    # - `Birthday`: Considered unnecessary for modeling.
    # - `Cost_of_Procedure_euros`: May be sensitive or uninformative depending
    #    on the goal.
    # - Specific one-hot encoded anesthesia categories, MAP (now recalculated),
    #   and others.
    # - All preoperative variables prefixed with 'Preop_' are removed, as MAP
    #   and SBP/DBP are already split and captured.

    if stage == "training":
        cols_to_drop = [
            "ID",
            "Weight_kg",
            "Cost_of_Procedure_euros",
            "Birthday",
            "BMI_Category_Normal_Weight",
            "Preop_DBP",
            "Anesthesia_Type_lidocaine",
            "Anesthesia_Type_carbocaine",
            "Intraop_MAP",
            "Comorbidity_Flag",
        ] + [col for col in df_sans_zero.columns if "Preop_" in col]

        # Save the list for reproducibility
        dumpObjects(
            cols_to_drop,
            os.path.join(data_path, "cols_to_drop.pkl"),
        )

        mlflow_dumpArtifact(
            experiment_name=exp_artifact_name,
            run_name=preproc_run_name,
            obj_name="cols_to_drop",
            obj=cols_to_drop,
        )

    elif stage == "inference":
        # Load list of columns to drop from MLflow artifact
        cols_to_drop = mlflow_loadArtifact(
            experiment_name=exp_artifact_name,
            run_name=preproc_run_name,
            obj_name="cols_to" "_drop",
        )

    ################################################################################
    # Step 22. Apply Final Column Drop and Print Resulting Schema
    ################################################################################

    # Drop all selected columns from the model-ready dataset.
    df_sans_zero.drop(columns=cols_to_drop, inplace=True)

    # Display dropped columns in a clean, stacked format
    columns_to_drop_str = "\n".join(cols_to_drop)
    print(f"\nDropping the following additional columns:\n\n{columns_to_drop_str}\n")
    print("*" * terminal_width)
    print()

    # Show the cleaned DataFrame schema that’s now ready for EDA or modeling
    columns_listed = "\n".join(circ_eda.columns.to_list())
    print(
        f"The DataFrame is in a cleaner state than the original.\n"
        f"It is now ready for exploratory data analysis with the columns listed below:\n\n"
        f"{columns_listed}\n"
    )

    ################################################################################
    # Step 23. Create Composite Target Outcome and Drop Redundant Functional Fields
    ################################################################################

    # For training purposes, we create a new target outcome column by aggregating
    # binary indicators of specific complications:
    # - Bleeding, Edema, Pain, and Infection
    # If any of these occurred, we mark the composite target outcome as 1.
    # Otherwise, it remains 0.

    if stage == "training":
        df_sans_zero[target_outcome] = (
            df_sans_zero["Functional_Outcomes_Bleeding"].astype(int)
            | df_sans_zero["Functional_Outcomes_Edema"].astype(int)
            | df_sans_zero["Functional_Outcomes_Pain"].astype(int)
            | df_sans_zero["Functional_Outcomes_Infection"].astype(int)
        )

        # Print distribution of the new target variable for QA
        print(f"{df_sans_zero[target_outcome].value_counts()}\n")

        # Drop all other functional outcome columns except the new target
        functional_cols_to_drop = [
            col
            for col in df_sans_zero.columns
            if "Functional" in col and target_outcome not in col
        ]

        # Save and log these for reproducibility
        dumpObjects(
            functional_cols_to_drop,
            os.path.join(data_path, "functional_cols_to_drop.pkl"),
        )

        mlflow_dumpArtifact(
            experiment_name=exp_artifact_name,
            run_name=preproc_run_name,
            obj_name="functional_cols_to_drop",
            obj=functional_cols_to_drop,
        )

    elif stage == "inference":
        # Load previously saved list of functional columns to drop
        functional_cols_to_drop = mlflow_loadArtifact(
            experiment_name=exp_artifact_name,
            run_name=preproc_run_name,
            obj_name="functional_cols_to_drop",
        )

    ################################################################################
    # Step 24. Drop Functional Columns and Enforce Numeric-Only Dataset
    ################################################################################

    # Drop now-redundant individual functional outcome columns
    df_sans_zero.drop(columns=functional_cols_to_drop, inplace=True)

    # Retain only numeric columns in the final modeling dataset
    df_sans_zero = df_sans_zero.select_dtypes(include=np.number)

    # Show updated list of remaining features
    print(
        f"\nUpdated List of remaining features:\n" f"{df_sans_zero.columns.to_list()}"
    )

    ################################################################################
    # Step 25. Display Correlation Matrix for Numeric Features
    ################################################################################

    # To understand relationships between numerical variables, we print the
    # correlation matrix. This helps detect multicollinearity and guides
    # potential feature selection or reduction.

    print("*" * terminal_width, "\n")

    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print("Correlation Matrix:\n")
        print(df_sans_zero.corr())

    print("*" * terminal_width, "\n")

    ################################################################################
    # Step 26. Final Variance Review Before Export
    ################################################################################

    # As a final QA step, we recheck the variance across all numeric features
    # in the cleaned dataset. This serves as a sanity check to confirm that
    # no zero-variance features were accidentally retained.

    print(f"Final Variance Check: \n")
    print(df_sans_zero.var())
    print()

    ################################################################################
    # Step 27. Save Cleaned Data for Modeling and EDA
    ################################################################################

    # We now save two versions of the cleaned dataset:
    # 1. `circ_eda`: Original copy with cost and unfiltered fields preserved for exploratory analysis.
    # 2. `df_sans_zero`: Modeling-ready version with only numeric and selected features.

    print(f"\nShape of Dataset: {df_sans_zero.shape}")

    print(f"\nDataFrame (First 5 Rows): \n")
    print(f"{df_sans_zero.head()}\n")

    file_paths = [
        (circ_eda, output_eda_file),
        (df_sans_zero, output_data_file),
    ]

    for dataframe, file_path in file_paths:
        print(f"File successfully saved to: {file_path}")
        dataframe.to_parquet(file_path, index=True)


if __name__ == "__main__":
    app()
