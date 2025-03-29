import pytest
from unittest import mock
import pandas as pd
import matplotlib.pyplot as plt

from circ_milan.functions import log_mlflow_metrics


@pytest.fixture
def mock_mlflow():
    with (
        mock.patch("mlflow.set_tracking_uri") as mock_set_tracking_uri,
        mock.patch(
            "circ_milan.functions.set_or_create_experiment"
        ) as mock_set_or_create,
        mock.patch("circ_milan.functions.get_run_id_by_name") as mock_get_run_id,
        mock.patch("mlflow.start_run") as mock_start_run,
        mock.patch("mlflow.log_metric") as mock_log_metric,
        mock.patch("mlflow.log_figure") as mock_log_figure,
    ):
        yield {
            "set_tracking_uri": mock_set_tracking_uri,
            "set_or_create_experiment": mock_set_or_create,
            "get_run_id_by_name": mock_get_run_id,
            "start_run": mock_start_run,
            "log_metric": mock_log_metric,
            "log_figure": mock_log_figure,
        }


def test_log_mlflow_metrics_with_data(mock_mlflow):
    mock_mlflow["set_or_create_experiment"].return_value = "1234"
    mock_mlflow["get_run_id_by_name"].return_value = None  # New run

    experiment_name = "Test Experiment"
    run_name = "Test Run"
    metrics = pd.Series({"accuracy": 0.9, "loss": 0.1})
    images = {"plot.png": plt.figure()}

    log_mlflow_metrics(experiment_name, run_name, metrics, images)

    mock_mlflow["set_tracking_uri"].assert_called_once()
    mock_mlflow["set_or_create_experiment"].assert_called_once_with(experiment_name)
    mock_mlflow["get_run_id_by_name"].assert_called_once_with(experiment_name, run_name)
    mock_mlflow["start_run"].assert_called_once_with(experiment_id="1234", run_id=None)
    mock_mlflow["log_metric"].assert_any_call("accuracy", 0.9)
    mock_mlflow["log_metric"].assert_any_call("loss", 0.1)
    mock_mlflow["log_figure"].assert_called_once_with(images["plot.png"], "plot.png")


def test_log_mlflow_metrics_no_metrics(mock_mlflow):
    mock_mlflow["set_or_create_experiment"].return_value = "5678"
    mock_mlflow["get_run_id_by_name"].return_value = "run123"

    experiment_name = "Test Experiment"
    run_name = "Test Run"
    metrics = None
    images = {"chart.png": plt.figure()}

    log_mlflow_metrics(experiment_name, run_name, metrics, images)

    mock_mlflow["set_or_create_experiment"].assert_called_once_with(experiment_name)
    mock_mlflow["get_run_id_by_name"].assert_called_once_with(experiment_name, run_name)
    mock_mlflow["start_run"].assert_called_once_with(
        experiment_id="5678", run_id="run123"
    )
    mock_mlflow["log_metric"].assert_not_called()
    mock_mlflow["log_figure"].assert_called_once_with(images["chart.png"], "chart.png")


def test_log_mlflow_metrics_empty(mock_mlflow):
    mock_mlflow["set_or_create_experiment"].return_value = "7890"
    mock_mlflow["get_run_id_by_name"].return_value = None

    experiment_name = "Test Experiment"
    run_name = "Test Run"
    metrics = pd.Series()
    images = {}

    log_mlflow_metrics(experiment_name, run_name, metrics, images)

    mock_mlflow["set_or_create_experiment"].assert_called_once_with(experiment_name)
    mock_mlflow["get_run_id_by_name"].assert_called_once_with(experiment_name, run_name)
    mock_mlflow["start_run"].assert_called_once_with(experiment_id="7890", run_id=None)
    mock_mlflow["log_metric"].assert_not_called()
    mock_mlflow["log_figure"].assert_not_called()
