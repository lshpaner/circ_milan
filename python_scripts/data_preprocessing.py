######################## Standard Library Imports ##############################
import pandas as pd
import numpy as np
import os
import sys
import eda_toolkit
from eda_toolkit import ensure_directory, dataframe_columns, add_ids
from tqdm import tqdm

from python_scripts.constants import *
from python_scripts.functions import (
    count_comorbidities,
    HealthMetrics,
)

print()
print(f"This project uses Python {sys.version.split()[0]}.")
print(f"EDA Toolkit Version: {eda_toolkit.__version__}")
print(f"EDA Toolkit Authors: {eda_toolkit.__author__}")
print()

terminal_width = os.get_terminal_size().columns  # establish terminal width

############################ Read in the data ##################################

# Define your base paths
# `base_path`` represents the parent directory of your current working directory
base_path = os.getcwd()  # Set base_path to the current working directory
data_path = os.path.join(base_path, "data")  # Correct path to './data'

print(f"Data Path: {data_path}")
print()
image_path_png = os.path.join(base_path, "images", "png_images")
image_path_svg = os.path.join(base_path, "images", "svg_images")

# Ensure that each directory exists
ensure_directory(data_path)
ensure_directory(image_path_png)
ensure_directory(image_path_svg)
print()

# read in the data, set index to "ID"
df = pd.read_excel(os.path.join(data_path, filename)).set_index("ID")

######################### Create Anonymized Random IDs #########################

df = add_ids(
    df=df,
    id_colname=patient_id,
    num_digits=9,
    seed=222,
    set_as_index=True,
)

print()

## inspect dataframe (first 5 rows)

print(df.head())

##################### Print the DataFrame Columns to List ######################

columns_list = df.columns.to_list()

print("*" * terminal_width)
print("DataFrame Columns: \n")
for col in columns_list:
    print(col)
print("*" * terminal_width)

######################### Rename Columns for Posterity #########################

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

functional = df[[col for col in df.columns if "Functional_" in col]]

for column in functional.columns:
    print(f"\n{functional[column].value_counts()}")

############################### Spelling Checks ################################
# In the event that there are any mispelled country names, they will be corrected.
print("Unique Geographical Origins:")
print()
print(df["Geographical_Origin"].unique())  # check unique countries in the series.
print()

# Correct the spelling for Morocco
df["Geographical_Origin"] = df["Geographical_Origin"].replace({"Marocco": "Morocco"})

############################# Correct Data Types ###############################

df["Functional_Outcomes_Cosmetic_Satisfaction"] = df[
    "Functional_Outcomes_Cosmetic_Satisfaction"
].astype("Int64")

# There was one missing value in `Functional_Outcomes_Cosmetic_Satisfaction`
# this should actually be `0`; imputed as such.
df["Functional_Outcomes_Cosmetic_Satisfaction"] = df[
    "Functional_Outcomes_Cosmetic_Satisfaction"
].fillna(0)

#################################### BMI #######################################

# BMI calculation using height and weight as the original column was empty.
HealthMetrics.calculate_bmi(
    df=df,
    weight_col="Weight_kg",
    height_col="Height_m",
    weight_unit="kg",
    height_unit="m",
)

# Derive BMI categories
HealthMetrics.calculate_bmi_category(df)

# Check the results
print(df[["BMI", "BMI_Category"]].head())

# MAP calculation from systolic and diastolic blood pressure.

# health_metrics.calculate_map(systolic_col="SystolicBP", diastolic_col="DiastolicBP")
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

df = df[df["Age_years"] >= 18]

# df["MAP_min"] = df["Intraop_MAP"] / df["Surgical_Time_min"]

# df["BMI_Age"] = df["BMI"] / df["Age_years"]


############################### Diabetes Prevalance ############################

df["Diabetes"] = (
    df["Comorbidities"].astype(str).str.contains("DM", case=False).astype(int)
)

print(f"\n{df['Diabetes'].value_counts()} \n")

################################


# df["Genital_Comorbidities"] = (
#     df["Comorbidities"]
#     .astype(str)
#     .str.contains(r"hydrocele|varicocele|LS", case=False)
#     .astype(int)
# )

# print(f"\n{df['Genital_Comorbidities'].value_counts()} \n")

################################# One-Hot Encoding #############################

# one hot encode insurance categories (break them out as separate cols of 0, 1)
df = df.assign(
    **pd.get_dummies(
        df[["Anesthesia_Type", "BMI_Category"]],
        dtype=int,
    )
)

############################### Intraop SBP and DBP ############################

# Split the BP values into two columns: Systolic (SBP) and Diastolic (DBP)
df[["Intraop_SBP", "Intraop_DBP"]] = df["Intraop_Mean_Blood_Pressure_mmHg"].str.split(
    "/",
    expand=True,
)

# Convert the new columns to numeric
df["Intraop_SBP"] = pd.to_numeric(df["Intraop_SBP"])
df["Intraop_DBP"] = pd.to_numeric(df["Intraop_DBP"])

############################### Comorbidities ##################################

# # create a new column for comorbidity flag (1 if comorbidities exist else 0)
# df["Comorbidity_Flag"] = df["Comorbidities"].apply(lambda x: 0 if x == 0 else 1)

# Replace 0 inside "Comorbidities" column with "None"
# df["Number_of_Comorbidities"] = (
#     df["Comorbidities"].replace({0: "None Present"}).apply(count_comorbidities)
# )

# print("*" * terminal_width)
# print("Number of Comorbidities and their prevalance: \n")

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
############################ Surgical Techniques ###############################
df["Surgical_Technique"] = df["Surgical_Technique"].apply(
    lambda x: 0 if x == "Traditional" else 1
)

############################ Antibiotic Grouping ###############################

# df["Preop_drugs_antibiotic"] = df["Preop_drugs_antibiotic"].apply(
#     lambda x: 1 if x == "Cefazolina" else 0
# )

########################## Check for Missing Values ############################
print("DataFrame Analysis Report (`dataframe_columns`) \n")
df_columns = dataframe_columns(df, return_df=True)

print(f"DataFrame Analysis:\n {df_columns}")
print()
if df_columns[df_columns["null_total"] >= 1].empty:
    print("No Missing Values to Report")
print()
print("*" * terminal_width)

########################## Correct Blood Loss Logic ############################
# df["Blood_loss_rate_ml_per_min"] = (
#     df["Intraoperative_Blood_Loss_ml"] / df["Surgical_Time_min"]
# )

##################### Zero Variance Inspection and Drops #######################

## check for variance
shape_var = df.var(numeric_only=True).to_frame().rename(columns={0: "Shape_Var"})

shape_var_thresh = 0.02

# Assuming df is your original DataFrame and shape_var is as shown in the screenshot
low_variance_columns = shape_var[
    shape_var["Shape_Var"] < shape_var_thresh
].index.to_list()
print()
print(f"Dropping the following low variance column(s): {low_variance_columns}")
print()

# Now, drop these columns from df
df.drop(columns=low_variance_columns, inplace=True)
########################### Join Cost to EDA Column ############################
#### Cost will be dropped, so making a copy of df to join it later

circ_eda = df.copy()

######################### Dropping Additional Columns ##########################

df["abnormal_weight"] = (
    df[[col for col in df.columns if col.startswith("BMI_") and "Normal" not in col]]
    .any(axis=1)
    .astype(int)
)

# Dropping uninformative features like "Birthday"
# Dropping Weight since Height was dropped, and BMI will be used instead
cols_to_drop = [
    "Weight_kg",
    "Cost_of_Procedure_euros",
    "Birthday",
    "BMI_Category_Normal weight",
    # "Intraoperative_Blood_Loss_ml",
    # "Surgical_Time_min",
    "Anesthesia_Type_lidocaine",
    "Anesthesia_Type_carbocaine",
    "Intraop_MAP",
] + [col for col in df.columns if "Preop_" in col]


# Prepare the string of columns to drop, each on a new line, in advance
columns_to_drop_str = "\n".join(cols_to_drop)

df.drop(columns=cols_to_drop, inplace=True)

# Then, use this string in the print statement without directly embedding it in
# an f-string
print("Dropping the following additional columns:\n\n" + columns_to_drop_str)
print()
print("*" * terminal_width)
print()
# Prepare the column names as a stacked list string
columns_listed = "\n".join(circ_eda.columns.to_list())

print(
    f"The DataFrame is in a cleaner state than the original.\n"
    f"It is now ready for exploratory data analysis with the columns listed below:\n\n"
    f"{columns_listed}"
)

print()

# File paths and names
file_paths = [
    (circ_eda, os.path.join(data_path, "circ_eda.csv")),
    (df, os.path.join(data_path, "circ_preproc.csv")),
]

for dataframe, file_path in file_paths:
    print(f"File successfully saved to: {file_path}")
    dataframe.to_csv(file_path, index=True)

print()
