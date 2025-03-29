from unittest.mock import patch, MagicMock
from circ_milan.functions import mlflow_log_parameters_model


def test_mlflow_log_parameters_model():
    experiment_name = "test_experiment"
    run_name = "test_run"
    model_type = "lr"
    n_iter = 10
    kfold = True
    outcome = "target"
    model_name = "test_model"
    hyperparam_dict = {
        "param1": 1,  # Scalar value instead of [1, 2]
        "param2": "value",
    }

    with (
        patch("mlflow.start_run"),
        patch("mlflow.log_param"),
        patch("mlflow.sklearn.log_model"),
        patch(
            "mlflow.get_experiment_by_name",
            return_value=MagicMock(experiment_id="123"),
        ),
        patch("mlflow.set_experiment"),
    ):
        mlflow_log_parameters_model(
            model_type,
            n_iter,
            kfold,
            outcome,
            run_name,
            experiment_name,
            model_name,
            None,  # Pass `None` instead of the model
            hyperparam_dict,
        )
