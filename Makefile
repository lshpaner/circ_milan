################################################################################
# GLOBALS                                                                      #
################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')
PROFILE = default
PROJECT_NAME = Circumcision Outcomes Milan
PYTHON_INTERPRETER = python3


ifeq (,$(shell which conda))
	HAS_CONDA=False
else
	HAS_CONDA=True
endif

### general usage notes
### 2>&1 | tee ==>pipe operation to save model output from terminal to .txt file

################################################################################
# COMMANDS                                                                     #
################################################################################

################################################################################
############## Setting up a Virtual Environment and Dependencies ###############
################################################################################
# virtual environment set-up (local)
venv:
	$(PYTHON_INTERPRETER) -m venv equi_venv
	source equi_venv/bin/activate

## Install Python Dependencies
requirements_local:	
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel

venv_dep_setup_local: venv requirements_local	# for local set-up
venv_dep_setup_gpu: venv requirements_gpu     # for server/gpu set-up

################################################################################
###########################  Dataset Script Generation #########################
################################################################################
# clean directories
clean_dir:
	@echo "Cleaning directory..."
	rm -rf public_data/
	rm -rf data/
	rm -rf data_output/
	rm -rf images/

data_preprocessing:
	$(PYTHON_INTERPRETER) -m python_scripts.data_preprocessing

feature_outcome_generation:
	$(PYTHON_INTERPRETER) -m python_scripts.feature_outcome_generation

################################################################################
##########################  Modeling Script Generation #########################
################################################################################

## Make Logistic Regression
.PHONY: logistic_regression
logistic_regression: 
	$(PYTHON_INTERPRETER) \
	python_scripts/train.py \
	--model-type lr \
	--exp-name logistic_regression \
	2>&1 | tee models/results/logistic_regression.txt

## Make Decision Tree Classifier
.PHONY: knn_neighbors
knn_neighbors: 
	$(PYTHON_INTERPRETER) \
	python_scripts/train.py \
	--model-type knn \
	--exp-name knn_neighbors \
	2>&1 | tee models/results/knn_neighbors.txt

## Make Naive Bayes
.PHONY: naive_bayes
naive_bayes: 
	$(PYTHON_INTERPRETER) \
	python_scripts/train.py \
	--model-type nb \
	--exp-name naive_bayes \
	2>&1 | tee models/results/naive_bayes.txt

## Make Support Vector Machines
.PHONY: svm
svm: 
	$(PYTHON_INTERPRETER) \
	python_scripts/train.py \
	--model-type svm \
	--exp-name svm \
	2>&1 | tee models/results/svm.txt

## Make Support Vector Machines
.PHONY: lda
lda: 
	$(PYTHON_INTERPRETER) \
	python_scripts/train.py \
	--model-type lda \
	--exp-name lda \
	2>&1 | tee models/results/lda.txt

## Make Support Vector Machines
.PHONY: qda
qda: 
	$(PYTHON_INTERPRETER) \
	python_scripts/train.py \
	--model-type qda \
	--exp-name qda \
	2>&1 | tee models/results/qda.txt
		

## Make Support Vector Machines
.PHONY: mlp
mlp: 
	$(PYTHON_INTERPRETER) \
	python_scripts/train.py \
	--model-type mlp \
	--exp-name mlp \
	2>&1 | tee models/results/mlp.txt
		

all_models: logistic_regression lda svm 