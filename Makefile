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
#################### Adult Income Dataset Script Generation ####################
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

