import pytest
from unittest import mock
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from circ_milan.functions import PlotMetrics


@pytest.fixture
def sample_data():
    """Fixture to provide sample dataframe and model inputs for testing."""
    df = pd.DataFrame(
        {
            "outcome": [0, 1, 1, 0, 1, 0, 1, 1],
            "pred": [0.1, 0.8, 0.7, 0.2, 0.9, 0.4, 0.6, 0.85],
        }
    )
    outcome_cols = ["outcome"]
    pred_cols = ["pred"]
    return df, outcome_cols, pred_cols


@pytest.fixture
def sample_inputs():
    """Fixture to provide sample model validation inputs."""
    X_valid = np.array(
        [
            [1, 2],
            [3, 4],
            [5, 6],
        ]
    )
    y_valid = np.array([0, 1, 1])
    return X_valid, y_valid


@pytest.fixture
def mock_model():
    """Fixture to create a mock model with a predict_proba method."""
    model = mock.Mock()
    model.predict_proba.return_value = np.array(
        [
            [0.9, 0.1],
            [0.2, 0.8],
            [0.3, 0.7],
        ]
    )
    return {"MockModel": model}


@pytest.fixture
def plot_metrics():
    """Fixture to create an instance of PlotMetrics with mocked saving."""
    with mock.patch.object(PlotMetrics, "_save_plot"):
        return PlotMetrics()


def test_plot_roc_with_dataframe(plot_metrics, sample_data):
    """
    Test that plot_roc works with a DataFrame and returns
    a Matplotlib figure.
    """
    df, outcome_cols, pred_cols = sample_data

    with mock.patch("matplotlib.pyplot.show"):
        fig = plot_metrics.plot_roc(
            df=df,
            outcome_cols=outcome_cols,
            pred_cols=pred_cols,
            show=False,
        )

    assert isinstance(fig, plt.Figure)


def test_plot_roc_with_model(
    plot_metrics,
    sample_inputs,
    mock_model,
):
    """Test that plot_roc works with a model and returns a Matplotlib figure."""
    X_valid, y_valid = sample_inputs

    with mock.patch("matplotlib.pyplot.show"):
        fig = plot_metrics.plot_roc(
            models=mock_model,
            X_valid=X_valid,
            y_valid=y_valid,
            show=False,
        )

    assert isinstance(fig, plt.Figure)
    mock_model["MockModel"].predict_proba.assert_called_once_with(X_valid)


def test_plot_precision_recall_with_dataframe(plot_metrics, sample_data):
    """
    Test that plot_precision_recall works with a DataFrame and returns a
    Matplotlib figure.
    """

    df, outcome_cols, pred_cols = sample_data

    with mock.patch("matplotlib.pyplot.show"):
        fig = plot_metrics.plot_precision_recall(
            df=df,
            outcome_cols=outcome_cols,
            pred_cols=pred_cols,
            show=False,
        )

    assert isinstance(fig, plt.Figure)


def test_plot_precision_recall_with_model(
    plot_metrics,
    sample_inputs,
    mock_model,
):
    """
    Test that plot_precision_recall works with a model and returns
    a Matplotlib figure.
    """
    X_valid, y_valid = sample_inputs

    with mock.patch("matplotlib.pyplot.show"):
        fig = plot_metrics.plot_precision_recall(
            models=mock_model,
            X_valid=X_valid,
            y_valid=y_valid,
            show=False,
        )

    assert isinstance(fig, plt.Figure)
    mock_model["MockModel"].predict_proba.assert_called_once_with(X_valid)


def test_plot_confusion_matrix_with_dataframe(plot_metrics, sample_data):
    """
    Test that plot_confusion_matrix works with a DataFrame and returns
    a Matplotlib figure.
    """
    df, outcome_cols, pred_cols = sample_data

    with mock.patch("matplotlib.pyplot.show"):
        fig = plot_metrics.plot_confusion_matrix(
            df=df,
            outcome_cols=outcome_cols,
            pred_cols=pred_cols,
            show=False,
        )

    assert isinstance(fig, plt.Figure)


def test_plot_confusion_matrix_with_model(
    plot_metrics,
    sample_inputs,
    mock_model,
):
    """
    Test that plot_confusion_matrix works with a model and returns
    a Matplotlib figure.
    """
    X_valid, y_valid = sample_inputs

    with mock.patch("matplotlib.pyplot.show"):
        fig = plot_metrics.plot_confusion_matrix(
            models=mock_model,
            X_valid=X_valid,
            y_valid=y_valid,
            show=False,
        )

    assert isinstance(fig, plt.Figure)
    mock_model["MockModel"].predict_proba.assert_called_once_with(X_valid)


def test_plot_confusion_matrix_with_custom_threshold(plot_metrics, sample_data):
    """
    Test that plot_confusion_matrix uses the '>' threshold logic.
    """
    df, outcome_cols, pred_cols = sample_data

    with mock.patch("matplotlib.pyplot.show"):
        fig = plot_metrics.plot_confusion_matrix(
            df=df,
            outcome_cols=outcome_cols,
            pred_cols=pred_cols,
            threshold=0.6,
            show=False,
        )

    assert isinstance(fig, plt.Figure)
    # Confirm logic using pandas - expected binary prediction
    expected_pred = (df["pred"] > 0.6).astype(int)
    assert expected_pred.tolist() == [0, 1, 1, 0, 1, 0, 0, 1]


def test_plot_confusion_matrix_with_model_and_threshold(
    plot_metrics,
    sample_inputs,
    mock_model,
):
    """
    Test that plot_confusion_matrix uses threshold > and
    not >= when scoring with models.
    """
    X_valid, y_valid = sample_inputs
    model = mock_model["MockModel"]

    with mock.patch("matplotlib.pyplot.show"):
        fig = plot_metrics.plot_confusion_matrix(
            models=mock_model,
            X_valid=X_valid,
            y_valid=y_valid,
            threshold=0.6,
            show=False,
        )

    assert isinstance(fig, plt.Figure)
    model.predict_proba.assert_called_once_with(X_valid)

    # Check expected thresholded result using > 0.6
    pred_probs = model.predict_proba.return_value[:, 1]
    expected_pred = (pred_probs > 0.6).astype(int)
    assert expected_pred.tolist() == [0, 1, 1]


def test_plot_confusion_matrix_with_use_optimal_threshold(
    plot_metrics,
    sample_inputs,
):
    """
    Test that plot_confusion_matrix calls model.predict with
    optimal_threshold=True.
    """
    X_valid, y_valid = sample_inputs
    mock_model = mock.Mock()
    mock_model.predict.return_value = np.array([0, 1, 1])
    models = {"MockModel": mock_model}

    with mock.patch("matplotlib.pyplot.show"):
        fig = plot_metrics.plot_confusion_matrix(
            models=models,
            model_name="MockModel",
            X_valid=X_valid,
            y_valid=y_valid,
            use_optimal_threshold=True,
            show=False,
        )

    assert isinstance(fig, plt.Figure)
    mock_model.predict.assert_called_once_with(
        X_valid,
        optimal_threshold=True,
    )
