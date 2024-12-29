from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import os
from model_tuner import Model, dumpObjects
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_selection import RFE
from sklearn.linear_model import ElasticNet
from sklearn.base import clone


from python_scripts.model_params import (
    model_definitions,
    stratify_list,
    bin_ages,
    rstate,
)
from python_scripts.functions import (
    create_stratified_other_column,
    metrics_report,
    log_mlflow_experiment,
    plot_roc,
)
from python_scripts.constants import age, age_bin, mlflow_data


@app.command()
def main(
    model_type: str = "lr",
    exp_name: str = "orig_models",
    features_path: Path = PROCESSED_DATA_DIR / "X.parquet",
    labels_path: Path = PROCESSED_DATA_DIR / "y.parquet",
    model_path: Path = MODELS_DIR,
    processed_data: Path = PROCESSED_DATA_DIR,
    model_results: Path = RESULTS_DIR,
    results: Path = PROCESSED_DATA_DIR,
):

    model_results_path = model_results / exp_name

    X = pd.read_parquet(features_path)
    y = pd.read_parquet(labels_path)

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
        RandomUnderSampler(random_state=rstate),
    ]:
        print()
        print("Sampler", sampler)

        X_train, X_valid, X_test, y_train, y_test, y_valid = {}, {}, {}, {}, {}, {}

        # Create a dataframe that includes the specified stratification columns
        # (age bins (if provided), sex, and race_ethnicity).
        stratify_df = create_stratified_other_column(
            X=X,
            stratify_list=stratify_list,
            age=age,
            age_bin=age_bin,
            bin_ages=bin_ages,
        )

        print()
        print(f"Circumcision Outcome Columns:")
        print(y.columns.to_list())
        print("-" * 60)
        print(f"Outcome Columns Currently Being Modeled:")
        print(f"{y.columns[:2].to_list()}")
        print("-" * 60)
        print()

        if model_type in {"xgb"}:
            pipeline = [("RFE", rfe)]
        else:
            pipeline = [
                ("StandardScalar", StandardScaler()),
                ("Preprocessor", SimpleImputer()),
                ("RFE", rfe),
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
                kfold=False,
                grid=tuned_parameters,
                n_jobs=2,
                randomized_grid=False,
                scoring=["average_precision"],
                random_state=rstate,
                stratify_cols=stratify_df,
                stratify_y=True,
                boost_early=early_stop,
                imbalance_sampler=sampler,
                feature_selection=True,
            )

            ############################################################################
            ######################### Extract Split Data Subsets #######################
            ############################################################################

            model_dict[m].grid_search_param_tuning(X, y[m], f1_beta_tune=True)

            X_train[m], y_train[m] = model_dict[m].get_train_data(X, y[m])
            X_valid[m], y_valid[m] = model_dict[m].get_valid_data(X, y[m])
            X_test[m], y_test[m] = model_dict[m].get_test_data(X, y[m])

            # Convert Series to DataFrame before saving to Parquet
            X_train[m].to_parquet(
                os.path.join(processed_data, f"X_train_{m}.parquet"),
            )
            y_train[m].to_frame().to_parquet(
                os.path.join(processed_data, f"y_train_{m}.parquet"),
            )

            X_valid[m].to_parquet(
                os.path.join(processed_data, f"X_valid_{m}.parquet"),
            )
            y_valid[m].to_frame().to_parquet(
                os.path.join(processed_data, f"y_valid_{m}.parquet"),
            )

            X_test[m].to_parquet(
                os.path.join(processed_data, f"X_test_{m}.parquet"),
            )
            y_test[m].to_frame().to_parquet(
                os.path.join(processed_data, f"y_test_{m}.parquet"),
            )

            ############################################################################

            if model_type in {"xgb", "cat"}:
                model_dict[m].fit(
                    X_train[m],
                    y_train[m],
                    validation_data=(X_valid[m], y_valid[m]),
                    score="average_precision",
                )
            else:
                model_dict[m].fit(
                    X_train[m],
                    y_train[m],
                    score="average_precision",
                )

            model_dict[m].return_metrics(X_valid[m], y_valid[m])

            if model_dict[m].calibrate:
                model_dict[m].calibrateModel(X, y[m], score="average_precision")

            model_dict[m].return_metrics(
                X_valid[m],
                y_valid[m],
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
                X_valid=X_valid[m],
                y_valid=y_valid[m],
            )

            metrics[m].index = ["valid " + ind for ind in metrics[m].index]

            # Create an empty row DataFrame with the same columns as the metrics
            empty_row = pd.DataFrame(
                [[""] * metrics[m].shape[1]],
                columns=metrics[m].columns,
            )
            empty_row.index = [""]  # Set an empty index for the empty row

            temp = metrics_report(
                models=cur_model,
                X_valid=X_test[m],
                y_valid=y_test[m],
            )
            temp.index = ["test " + ind for ind in temp.index]

            # Concatenate the valid metrics, empty row, and test metrics
            metrics[m] = pd.concat((metrics[m], empty_row, temp))

            temp = metrics_report(
                models=cur_model,
                X_valid=X_train[m],
                y_valid=y_train[m],
            )
            temp.index = ["train " + ind for ind in temp.index]
            empty_row.index = [" "]

            # Concatenate the train metrics, empty row, and test metrics
            metrics[m] = pd.concat((metrics[m], empty_row, temp))

            # print(metrics[m])

            ####################################################################
            ########################### MLFlow #################################
            ####################################################################

        for m in model_dict:
            fig = plot_roc(
                models={estimator_name: model_dict[m]},
                X_valid=X_valid[m],
                y_valid=y_valid[m],
                show=False,
            )

            # Close the figure to suppress display
            # plt.close(fig)

            # print( f"{estimator_name}_{m}_{'Original'}")
            # print( model_lr[m].best_params_per_score)
            # print( metrics[m][estimator_name])
            # print(f"roc_auc_{m}_year.png")
            # print(log_mlflow_experiment)

            log_mlflow_experiment(
                mlflow_data=mlflow_data,
                experiment_name=f"Circ_Milan_{exp_name}",
                # model_name=f"{estimator_name}_orig",
                model_name=f"{estimator_name}_{m}_{'Original'}",
                best_params=model_dict[m].best_params_per_score,
                metrics=metrics[m][estimator_name],
                images={f"roc_auc_{m}_year.png": fig},
            )

            plt.close(fig)

            # # print(metrics[2])
            # if (
            #     metrics[2].loc["AUC ROC", estimator_name]
            #     + metrics[5].loc["AUC ROC", estimator_name]
            #     > best_score
            # ):
            #     best_score = (
            #         metrics[2].loc["AUC ROC", estimator_name]
            #         + metrics[5].loc["AUC ROC", estimator_name]
            #     )
            #     best_model = model_lr.copy()

    print(os.getcwd())
    dumpObjects(
        {
            "model": model_dict,  # Trained model
        },
        os.path.join(
            model_results_path,
            f"{str(clc).split('(')[0]}_{exp_name}.pkl",
        ),
    )

    logger.success("Modeling training complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
