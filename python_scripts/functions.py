################################################################################
######################### Import Requisite Libraries ###########################
################################################################################

import pandas as pd

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
    def split_bp_column(
        df,
        combined_bp_col,
        systolic_col_name="SBP",
        diastolic_col_name="DBP",
    ):
        """
        Split a combined blood pressure column in "Systolic/Diastolic" format
        into separate systolic (SBP) and diastolic (DBP) columns.

        Parameters:
            df (DataFrame): The pandas DataFrame containing the combined column.
            combined_bp_col (str): Column name of the combined BP values.
            systolic_col_name (str): Column name for the new systolic BP values.
            diastolic_col_name (str): Column name for the new diastolic BP values.

        Returns:
            None: The DataFrame is updated in place with the new columns.
        """
        # Split the combined BP column into two new columns
        df[[systolic_col_name, diastolic_col_name]] = df[combined_bp_col].str.split(
            "/", expand=True
        )

        # Convert the new columns to numeric types
        df[systolic_col_name] = pd.to_numeric(
            df[systolic_col_name],
            errors="coerce",
        )
        df[diastolic_col_name] = pd.to_numeric(
            df[diastolic_col_name],
            errors="coerce",
        )

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
                return "Normal_Weight"
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
