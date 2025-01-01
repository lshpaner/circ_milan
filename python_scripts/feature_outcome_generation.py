######################## Standard Library Imports ##############################
import pandas as pd
import numpy as np
import os
import sys
import eda_toolkit
from eda_toolkit import (
    ensure_directory,
    summarize_all_combinations,
    save_dataframes_to_excel,
)


from python_scripts.functions import *  # import custom functions
from python_scripts.constants import *  # import constants file

print()
print(f"This project uses Python {sys.version.split()[0]}.")
print(f"EDA Toolkit Version: {eda_toolkit.__version__}")
print(f"EDA Toolkit Authors: {eda_toolkit.__author__}")
print()

terminal_width = os.get_terminal_size().columns  # establish terminal width

############################# Read in the data #################################

# Go up one level from 'notebooks' to the parent directory, then into the 'data' folder
print(os.getcwd())

data_path = os.path.join(os.getcwd(), "data")

# Use the function to ensure the 'data' directory exists
ensure_directory(data_path)

# read in the data, set index to "ID"
df = pd.read_csv(os.path.join(data_path, preproc_filename)).set_index("patient_id")

print()
print("DataFrame (First 5 Rows): \n")
print(df.head())
print()
print("*" * terminal_width)
print("DataFrame Columns: \n")
for col in df.columns:
    print(col)
print()
print("*" * terminal_width)

############################ Save DataFrames to Excel ##########################

file_name = "df_circ_ages.xlsx"  # Name of the output Excel file
file_path = os.path.join(data_path, file_name)

# filter DataFrame to Ages 18-40
filtered_df = df[(df["Age_years"] > 18) & (df["Age_years"] < 40)]

df_dict = {
    "original_df": df,
    "ages_18_to_40": filtered_df,
}

save_dataframes_to_excel(
    file_path=file_path,
    df_dict=df_dict,
    decimal_places=0,
)

############################ Summarize All Combinations ########################

print()
summarize_all_combinations(
    df=df,
    variables=["Age_years", "Preop_Heart_Rate_bpm"],
    data_path=data_path,
    data_name="new_test.xlsx",
)

################################# One-Hot Encoding #############################

# one hot encode insurance categories (break them out as separate cols of 0, 1)
df = df.assign(
    **pd.get_dummies(
        df[["Surgical_Technique", "Anesthesia_Type"]],
        dtype=int,
    )
)

print("*" * terminal_width)
print()
print("Functional Columns in Dataset:")
print()
for col in df.columns:
    if "Functional" in col:
        print(col)
print()
print("*" * terminal_width)

############################### Variable Selection #############################

df = df.select_dtypes(np.number)  # select only numeric datatypes for modeling

print()
print("Preop Columns in Dataset:")
print()
for col in df.columns:
    if "Preop" in col:
        print(col)
print()
print("*" * terminal_width)

############################# Final Variance Check #############################
print()
print("Final Variance Check:")
print(df.var())
print()

######################### Feature and Outcome Generation #######################

X = df[[col for col in df.columns if col != outcome]]

print(f"Feature List: {X.columns.to_list()}")
print()
y = df[[outcome]]

print(f"Outcome: {outcome}")

################################# Save out X and y #############################
# File paths and names
file_paths_parquet = [
    (X, os.path.join(data_path, "X.parquet")),
    (y, os.path.join(data_path, "y.parquet")),
]

for dataframe, file_path in file_paths_parquet:
    print(f"File successfully saved to: {file_path}")
    dataframe.to_parquet(file_path, index=True)


# File paths and names
file_paths_csv = [
    (X, os.path.join(data_path, "X.csv")),
    (y, os.path.join(data_path, "y.csv")),
]

for dataframe, file_path in file_paths_csv:
    print(f"File successfully saved to: {file_path}")
    dataframe.to_csv(file_path, index=True)

print()
