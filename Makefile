# Makefile
# ------------------------------------------------------------------------------
# GLOBALS
# ------------------------------------------------------------------------------
PROJECT_NAME = circ_milan
PYTHON_VERSION = 3.11
PYTHON_INTERPRETER = python
VENV_DIR = venv_circ_311
CONDA_ENV_NAME = conda_circ_311
PROJECT_DIRECTORY = .


############################## Training Globals ################################

# Define variables for looping
OUTCOMES = Bleeding_Edema_Outcome
PIPELINES = orig smote over 
# PIPELINES = smote 
SCORING = average_precision
PRETRAINED ?= 0  # 0 if you want to train the models, 1 if calibrate pretrained

############################# Production Globals ###############################

# Model outcome variable used in production 
EXPLAN_OUTCOME = Bleeding_Edema_Outcome # explainer outcome variable
PROD_OUTCOME = Bleeding_Edema_Outcome # production outcome variable


# ------------------------------------------------------------------------------
# COMMANDS
# ------------------------------------------------------------------------------
.PHONY: init_config
init_config:
	@CURRENT_DIR=$$(sed -n 's/^PROJECT_DIRECTORY = //p' Makefile); \
	\
	read -p "Enter project name: " project_name; \
	read -p "Enter Python version (e.g., 3.10.12): " python_version; \
	read -p "Enter Python interpreter (default: python): " python_interpreter; \
	read -p "Enter virtual environment directory name: " venv_dir; \
	read -p "Enter conda environment name: " conda_env; \
	python_interpreter=$${python_interpreter:-python}; \
	\
	if [ -d "$$CURRENT_DIR" ] && [ "$$CURRENT_DIR" != "$$project_name" ]; then \
		mv "$$CURRENT_DIR" "$$project_name"; \
	fi; \
	\
	# Cross-platform sed command (works on both macOS and Linux) \
	if [ "$$(uname)" = "Darwin" ]; then \
		sed -i '' \
			-e "s/^PROJECT_NAME = .*/PROJECT_NAME = $${project_name}/" \
			-e "s/^PYTHON_VERSION = .*/PYTHON_VERSION = $${python_version}/" \
			-e "s/^PYTHON_INTERPRETER = .*/PYTHON_INTERPRETER = $${python_interpreter}/" \
			-e "s/^VENV_DIR = .*/VENV_DIR = $${venv_dir}/" \
			-e "s/^CONDA_ENV_NAME = .*/CONDA_ENV_NAME = $${conda_env}/" \
			-e "s|^PROJECT_DIRECTORY = .*|PROJECT_DIRECTORY = $${project_name}|" \
			Makefile; \
	else \
		sed -i \
			-e "s/^PROJECT_NAME = .*/PROJECT_NAME = $${project_name}/" \
			-e "s/^PYTHON_VERSION = .*/PYTHON_VERSION = $${python_version}/" \
			-e "s/^PYTHON_INTERPRETER = .*/PYTHON_INTERPRETER = $${python_interpreter}/" \
			-e "s/^VENV_DIR = .*/VENV_DIR = $${venv_dir}/" \
			-e "s/^CONDA_ENV_NAME = .*/CONDA_ENV_NAME = $${conda_env}/" \
			-e "s|^PROJECT_DIRECTORY = .*|PROJECT_DIRECTORY = $${project_name}|" \
			Makefile; \
	fi; \
	\
	# Replace project name in Python files and other text files only \
	if [ "$$(uname)" = "Darwin" ]; then \
		find "./$$project_name" -type f \( -name "*.py" -o -name "*.txt" -o -name "*.md" -o -name "*.yaml" -o -name "*.json" \) -exec sed -i '' "s/$$CURRENT_DIR/$$project_name/g" {} \;; \
	else \
		find "./$$project_name" -type f \( -name "*.py" -o -name "*.txt" -o -name "*.md" -o -name "*.yaml" -o -name "*.json" \) -exec sed -i "s/$$CURRENT_DIR/$$project_name/g" {} \;; \
	fi; \
	\
	echo "Configuration updated successfully. Folder '$$CURRENT_DIR' -> '$$project_name'."

.PHONY: check_vars
check_vars:
	@echo "Dummy configuration detected."
	@echo ""
	@echo "Please update the following variables in your Makefile before proceeding:"
	@echo " - PROJECT_NAME"
	@echo " - PYTHON_VERSION"
	@echo " - VENV_DIR"
	@echo " - CONDA_ENV_NAME"
	@echo " - OUTCOMES"
	@echo " - PIPELINES"
	@echo " - SCORING"
	@echo " - EXPLAN_OUTCOME"
	@echo " - PROD_OUTCOME"
	@echo ""
	@echo "Once you've replaced the dummy values, you can run your full pipeline commands safely."

## Set up python interpreter environment
create_conda_env:
	@echo "Run 'conda create -n $(CONDA_ENV_NAME) python=$(PYTHON_VERSION)' to create conda environment"

# Target to create a virtual environment
create_venv:
	# Create the virtual environment using the specified Python version
	$(PYTHON_INTERPRETER) -m venv $(VENV_DIR)
	@echo "Virtual environment created with $(PYTHON_INTERPRETER)$(PYTHON_VERSION)"

# Target to activate the virtual environment (Unix-based systems)
activate_venv:
	@echo "Run 'conda deactivate' to deactivate the $(CONDA_ENV_NAME) conda environment"
	@echo "Run 'source $(VENV_DIR)/bin/activate' to activate the virtual environment"

# Target to clean the virtual environment
clean_venv:
	rm -rf $(VENV_DIR)
	@echo "Virtual environment removed"

## Install Python Dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

.PHONY: mlflow_ui
mlflow_ui:
	mlflow ui --backend-store-uri mlruns --host 0.0.0.0 --port 5501

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

################################################################################
####################### Preprocessing (+) Dataprep Pipeline ####################
################################################################################
# clean directories
clean_dir:
	@echo "Cleaning directory..."
	rm -rf data/

################################################################################
################################ Folder Creation  ##############################
################################################################################

.PHONY: create_folders
create_folders:
# Create data subdirectories
	mkdir -p data/external data/interim data/processed data/raw data/processed/inference
	mkdir -p models/results models/eval
	mkdir -p modeling preprocessing
	touch data/interim/.gitkeep
	touch data/processed/.gitkeep
	touch data/processed/inference/.gitkeep
	touch models/results/.gitkeep
	touch models/eval/.gitkeep
	touch /modeling/__init__.py
	touch /preprocessing/__init__.py

# Create models subdirectories for each outcome
	@for outcome in $(OUTCOMES); do \
		mkdir -p models/results/$$outcome; \
		mkdir -p models/eval/$$outcome; \
	done

## Replace <your_project_dir> with actual path to your preprocessing script
## Replace <raw_input_filename> with the name of the raw input file
## Replace <cleaned_output_filename> with the name of the cleaned output file
.PHONY: data_prep_preprocessing_training
data_prep_preprocessing_training:
	$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/preprocessing/preprocessing.py \
		--input-data-file ./data/raw/Laser_Circumcision_Excel_31.03.2024.xlsx \
		--output-eda-file ./data/processed/circ_eda.parquet \
		--output-data-file ./data/processed/df_sans_zero.parquet \
		--stage training \
		--data-path ./data/processed \
	2>&1 | tee data/processed/preproccessing.txt

.PHONY: feat_gen_training
feat_gen_training:
	$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/preprocessing/feat_gen.py \
	--input-data-file ./data/processed/df_sans_zero.parquet \
	--stage training \
	--data-path ./data/processed


preproc_pipeline: data_prep_preprocessing_training feat_gen_training

################################################################################
################################# Training #####################################
#################### Supportive Care RFE, Imb Learn Models #####################
################################################################################

train_logistic_regression:
	@echo "Pretrained is set to: $(PRETRAINED)"
	@for outcome in $(OUTCOMES); do \
		for pipeline in $(PIPELINES); do \
			mkdir -p models/results/$$outcome; \
			"$(PYTHON_INTERPRETER)" $(PROJECT_DIRECTORY)/modeling/train.py \
				--model-type lr \
				--pipeline-type "$$pipeline" \
				--features-path ./data/processed/X.parquet \
				--labels-path ./data/processed/y_$$outcome.parquet \
				--outcome "$$outcome" \
				--pretrained "$(PRETRAINED)" \
				--scoring "$(SCORING)" \
				2>&1 | tee models/results/$$outcome/lr_$$pipeline$$( [ "$(PRETRAINED)" -eq 1 ] && echo "_prefit" ).txt; \
		done; \
	done

train_random_forest:
	@echo "Pretrained is set to: $(PRETRAINED)"
	@for outcome in $(OUTCOMES); do \
		for pipeline in $(PIPELINES); do \
			mkdir -p models/results/$$outcome; \
			"$(PYTHON_INTERPRETER)" $(PROJECT_DIRECTORY)/modeling/train.py \
				--model-type rf \
				--pipeline-type "$$pipeline" \
				--features-path ./data/processed/X.parquet \
				--labels-path ./data/processed/y_$$outcome.parquet \
				--outcome "$$outcome" \
				--pretrained "$(PRETRAINED)" \
				--scoring "$(SCORING)" \
				2>&1 | tee models/results/$$outcome/rf_$$pipeline$$( [ "$(PRETRAINED)" -eq 1 ] && echo "_prefit" ).txt; \
		done; \
	done

train_svm:
	@echo "Pretrained is set to: $(PRETRAINED)"
	@for outcome in $(OUTCOMES); do \
		for pipeline in $(PIPELINES); do \
			mkdir -p models/results/$$outcome; \
			"$(PYTHON_INTERPRETER)" $(PROJECT_DIRECTORY)/modeling/train.py \
				--model-type svm \
				--pipeline-type "$$pipeline" \
				--features-path ./data/processed/X.parquet \
				--labels-path ./data/processed/y_$$outcome.parquet \
				--outcome "$$outcome" \
				--pretrained "$(PRETRAINED)" \
				--scoring "$(SCORING)" \
				2>&1 | tee models/results/$$outcome/svm_$$pipeline$$( [ "$(PRETRAINED)" -eq 1 ] && echo "_prefit" ).txt; \
		done; \
	done

train_all_models: train_logistic_regression train_random_forest train_svm

################################################################################
############################## Model Evaluation ################################
################################################################################

eval_logistic_regression:
	@for outcome in $(OUTCOMES); do \
		for pipeline in $(PIPELINES); do \
			$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/modeling/evaluation.py \
			--model-type lr \
			--pipeline-type $$pipeline \
			--features-path ./data/processed/X.parquet \
			--labels-path ./data/processed/y_$$outcome.parquet \
			--outcome $$outcome \
			--scoring $(SCORING) 2>&1 | tee models/eval/$$outcome/lr_eval_$$pipeline.txt; \
		done; \
	done

# Loop through each outcome for Random Forest
eval_random_forest:
	@for outcome in $(OUTCOMES); do \
		for pipeline in $(PIPELINES); do \
			$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/modeling/evaluation.py \
			--model-type rf \
			--pipeline-type $$pipeline \
			--features-path ./data/processed/X.parquet \
			--labels-path ./data/processed/y_$$outcome.parquet \
			--outcome $$outcome \
			--scoring $(SCORING) 2>&1 | tee models/eval/$$outcome/rf_eval_$$pipeline.txt; \
		done; \
	done

# Loop through each outcome for XGBoost
eval_svm:
	@for outcome in $(OUTCOMES); do \
		for pipeline in $(PIPELINES); do \
			$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/modeling/evaluation.py \
			--model-type svm \
			--pipeline-type $$pipeline \
			--features-path ./data/processed/X.parquet \
			--labels-path ./data/processed/y_$$outcome.parquet \
			--outcome $$outcome \
			--scoring $(SCORING) 2>&1 | tee models/eval/$$outcome/svm_eval_$$pipeline.txt; \
		done; \
	done


eval_all_models: eval_logistic_regression eval_random_forest eval_svm 

################################################################################
########## Preprocessing, Feature Generation, Training and Evaluation ##########
################################################################################

# This pipeline is to run consecutively the full preprocessing, training, and 
# evaluation pipeline in one command

preproc_train_eval: preproc_pipeline train_all_models eval_all_models


################################################################################
#################### Best Model Explainer and Explanations #####################
################################################################################

.PHONY: model_explainer
model_explainer:
	@for outcome in $(EXPLAN_OUTCOME); do \
		$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/modeling/explainer.py \
			--input-data-file ./data/processed/X.parquet \
			--outcome $$outcome \
			--metric-name "K-Fold Average Precision" \
			--mode max; \
	done

.PHONY: model_explanations_training
model_explanations_training:
	@for outcome in $(EXPLAN_OUTCOME); do \
		$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/modeling/explanations_training.py \
			--features-path ./data/processed/X.parquet \
			--labels-path ./data/processed/y_$$outcome.parquet \
			--outcome $$outcome \
			--metric-name "K-Fold AUC ROC" \
			--mode max \
			--top-n 5 \
			--shap-val-flag 1 \
			--explanations-path ./data/processed/shap_predictions_$$outcome.csv; \
	done

model_explaining_training: model_explainer model_explanations_training

.PHONY: model_explanations_inference
model_explanations_inference:
	@for outcome in $(EXPLAN_OUTCOME); do \
		$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/modeling/explanations_inference.py \
			--features-path ./data/processed/inference/X.parquet \
			--outcome $$outcome \
			--metric-name "K-Fold AUC ROC" \
			--mode max \
			--top-n 5 \
			--shap-val-flag 1 \
			--explanations-path ./data/processed/inference/shap_predictions_$$outcome.csv; \
	done

################################################################################
################################# Production ###################################
############################### Model Predict ##################################
################################################################################

.PHONY: data_prep_preprocessing_inference
data_prep_preprocessing_inference:
	$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/preprocessing/preprocessing.py \
	--input-data-file ./data/raw/Laser_Circumcision_Excel_31.03.2024.xlsx \
	--output-data-file ./data/processed/inference/df_inference_process.parquet \
	--stage inference \
	--data-path ./data/processed

.PHONY: feat_gen_inference
feat_gen_inference: 
	$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/preprocessing/feat_gen.py \
	--input-data-file ./data/processed/inference/df_inference_process.parquet \
	--stage inference \
	--data-path ./data/processed/inference

.PHONY: predict
predict:
	@for outcome in $(PROD_OUTCOME); do \
		$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/modeling/predict.py \
			--input-data-file data/processed/inference/X.parquet \
			--predictions-path ./data/processed/inference/predictions_$$outcome.csv \
			--outcome $$outcome \
			--metric-name "K-Fold Average Precision" \
			--mode max; \
	done


.PHONY: preproc_pipeline_inf
preproc_pipeline_inf: data_prep_preprocessing_inference feat_gen_inference
#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)