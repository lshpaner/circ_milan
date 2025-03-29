################################################################################
############################### Constants ######################################
################################################################################

filename = "Laser Circumcision Excel 31.03.2024.xlsx"
preproc_filename = "circ_preproc.csv"

patient_id = "patient_id"

dependent_var = "Functional_Outcomes_Bleeding"
rstate = 222

age = ["Age_years"]

mlflow_data = "models/mlflow_exp"
var_index = "patient_id"

outcome = "Bleeding_Edema_Outcome"

model_svm = "SVC.pkl"


################################################################################
################################################################################
############################# Path Variables ###################################
################################################################################

model_output = "model_output"  # model output path

################################################################################
############################# Mlflow Variables #################################
################################################################################

mlflow_artifacts_data = "./mlruns/preprocessing"
mlflow_models_data = "./mlruns/models"
mlflow_models_copy = "./mlruns/models_copy"

artifact_data = "artifacts/"  # path to store mlflow artifacts
profile_data = "profile_data"  # path to store pandas profiles in
data_path = "data/processed/"


# One Hot Encoded Vars to Be Omitted
# Already one-hot encoded by data engineering team
cat_vars = []


################################################################################
########################## Variable/DataFrame Constants ########################
################################################################################

var_index = "ID"  # id index
patient_id = "patient_id"
age = "age"  # age
age_bin = ""  # bin of ages for stratification only
main_df = "df.parquet"  # main dataframe file name

miss_col_thresh = 60  # missingness threshold tolerated for zero-var cols
perc_below_indiv = f"perc_below_{miss_col_thresh}_indiv"
miss_row_thresh = 0.5  # missingness threshold (rows) tolerated based on dev. set
percent_miss = "percentage_missing"  # new col for percentage missing in rows
miss_indicator = "missing_indicator"  # indicator for percentage missing (0,1)

capital_gain = "capital-gain"
################################################################################

# The below artificat name is used for preprocessing alone
exp_artifact_name = "preprocessing"
preproc_run_name = "preprocessing"
artifact_run_id = "preprocessing"
artifact_name = "preprocessing"


################################################################################
############################## SHAP Constants ##################################

shap_artifact_name = "explainer"
shap_run_name = "explainer"
shap_artifacts_data = "./mlruns/explainer"


################################################################################
############################### Target Outcome #################################

target_outcome = "income"
