import typer
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from model_tuner import Model, dumpObjects
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from sklearn.base import clone
from pathlib import Path

# Add the parent directory to sys.path to access 'functions.py'
print(os.path.join(os.pardir))
sys.path.append(os.path.join(os.pardir))
sys.path.append(".")

from python_scripts.model_params import (
    model_definitions,
    stratify_list,
    bin_ages,
    rstate,
)
from python_scripts.functions import (
    # create_stratified_other_column,
    metrics_report,
    log_mlflow_experiment,
    plot_roc,
    plot_precision_recall,
    plot_confusion_matrix,
)
from python_scripts.constants import age, mlflow_data

app = typer.Typer()

PROCESSED_DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
MODELS_DIR = Path("models")


@app.command()
def main(
    model_type: str = "lr",
    exp_name: str = "logistic_regression",
    features_path: Path = PROCESSED_DATA_DIR / "X.parquet",
    labels_path: Path = PROCESSED_DATA_DIR / "y.parquet",
    results: Path = MODELS_DIR / "results",
):

    X = pd.read_parquet(features_path)
    y = pd.read_parquet(labels_path)

    print(f"y:{y}")

    clc = model_definitions[model_type]["clc"]
    estimator_name = model_definitions[model_type]["estimator_name"]

    # Set the parameters
    tuned_parameters = model_definitions[model_type]["tuned_parameters"]
    rand_grid = model_definitions[model_type]["randomized_grid"]
    early_stop = model_definitions[model_type]["early"]

    best_model = {}
    best_score = 0

    model_dict = {}
    metrics = {}

    for sampler in [
        None,
        SMOTE(random_state=rstate),
        ADASYN(random_state=rstate),
        RandomOverSampler(random_state=rstate),
    ]:
        print()
        print("Sampler", sampler)

        # Create a dataframe that includes the specified stratification columns
        # (age bins (if provided), sex, and race_ethnicity).
        # stratify_df = create_stratified_other_column(
        #     X=X,
        #     stratify_list=stratify_list,
        #     age=age,
        #     age_bin=age_bin,
        #     bin_ages=bin_ages,
        # )

        print()
        print(f"Circumcision Outcome Columns:")
        print(y.columns.to_list())
        print("*" * 80)
        print()

        pipeline = [
            ("StandardScalar", StandardScaler()),
            ("Preprocessor", SimpleImputer()),
        ]

        for m in y.columns[:2]:
            print()
            print("=" * 60)
            print(f"{m}")
            print("=" * 60)

            model_dict[m] = Model(
                pipeline_steps=pipeline,
                name=estimator_name,
                model_type="classification",
                estimator_name=estimator_name,
                calibrate=True,
                estimator=clone(clc),
                kfold=True,
                grid=tuned_parameters,
                n_jobs=2,
                randomized_grid=False,
                scoring=["average_precision"],
                random_state=rstate,
                # stratify_cols=stratify_df,
                stratify_y=True,
                boost_early=early_stop,
                imbalance_sampler=sampler,
            )

            ############################################################################
            ######################### Extract Split Data Subsets #######################
            ############################################################################

            model_dict[m].grid_search_param_tuning(X, y[m], f1_beta_tune=True)

            ############################################################################

            model_dict[m].fit(
                X,
                y[m],
                score="average_precision",
            )

            model_dict[m].return_metrics(X, y[m])

            if model_dict[m].calibrate:
                model_dict[m].calibrateModel(X, y[m], score="average_precision")

            model_dict[m].return_metrics(
                X,
                y[m],
                optimal_threshold=True,
                print_threshold=True,
                model_metrics=True,
            )

            ####################################################################
            print("=" * 80)
            cur_model = {}
            cur_model[estimator_name] = model_dict[m]

            metrics[m] = metrics_report(
                models=cur_model,
                X_valid=X,
                y_valid=y[m],
            )

            ####################################################################
            ########################### MLFlow #################################
            ####################################################################

            for m in model_dict:
                fig_1 = plot_roc(
                    models={estimator_name: model_dict[m]},
                    X_valid=X,
                    y_valid=y[m],
                    custom_name=estimator_name,
                    show=False,
                )

                fig_2 = plot_precision_recall(
                    models={estimator_name: model_dict[m]},
                    X_valid=X,
                    y_valid=y[m],
                    custom_name=estimator_name,
                    show=False,
                )

                fig_3 = plot_confusion_matrix(
                    models={estimator_name: model_dict[m]},
                    X_valid=X,
                    y_valid=y[m],
                    threshold=next(iter(model_dict[m].threshold.values())),
                    custom_name=estimator_name,
                    show=False,
                )

            print("^^^" * 30)
            # print(model_dict[m][estimator_name])
            # quit()

            print(f"{estimator_name}_{m}")
            print(metrics[m][estimator_name])
            # print(f"roc_auc_{m}_year.png")
            print(log_mlflow_experiment)

            log_mlflow_experiment(
                mlflow_data=mlflow_data,
                experiment_name=f"Circ_Milan_{exp_name}",
                model_name=f"{estimator_name}_{sampler}_{m}_{'Original'}",
                best_params=model_dict[m].best_params_per_score,
                metrics=metrics[m][estimator_name],
                images={
                    f"roc_auc_{m}.png": fig_1,
                    f"pr_{m}.png": fig_2,
                    f"cm_{m}.png": fig_3,
                },
            )

        if metrics[m].loc["AUC ROC", estimator_name] > best_score:
            best_score = metrics[m].loc["AUC ROC", estimator_name]
            best_model = model_dict.copy()

    print(os.getcwd())
    dumpObjects(
        {
            "model": model_dict,  # Trained model
        },
        results / f"{str(clc).split('(')[0]}.pkl",
    )


if __name__ == "__main__":
    app()
