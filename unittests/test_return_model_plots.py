import pytest
from unittest import mock
import numpy as np

from circ_milan.functions import return_model_plots


@pytest.fixture
def mock_model():
    """Fixture to create a mock model with a threshold attribute."""
    model = mock.Mock()
    model.threshold = {"default": 0.4}  # Mocking a threshold

    # Ensure `predict_proba` returns an array-like structure
    model.predict_proba.return_value = np.array([[0.2, 0.8], [0.7, 0.3]])

    return model


@pytest.fixture
def sample_inputs():
    """Fixture to provide sample inputs for testing."""
    return {
        "train": (np.array([[1, 2], [3, 4]]), np.array([0, 1])),
        "test": (np.array([[5, 6], [7, 8]]), np.array([1, 0])),
    }


@pytest.fixture
def mock_plotter():
    """Fixture to mock the PlotMetrics class and its methods."""
    plotter = mock.Mock()
    plotter.plot_roc.return_value = "mock_roc_plot"
    plotter.plot_confusion_matrix.return_value = "mock_cm_plot"
    plotter.plot_precision_recall.return_value = "mock_pr_plot"
    plotter.plot_metrics_vs_thresholds.return_value = "mock_threshold_plot"
    plotter.plot_calibration_curve.return_value = "mock_calibration_plot"
    return plotter


def test_return_model_plots(mock_model, sample_inputs, mock_plotter):
    """Test that return_model_plots correctly calls PlotMetrics and returns expected output."""
    estimator_name = "RandomForest"

    with mock.patch("circ_milan.functions.PlotMetrics", return_value=mock_plotter):
        result = return_model_plots(
            sample_inputs, mock_model, estimator_name, scoring="accuracy"
        )

    # Adjusted key names based on function output
    expected_keys = {
        "roc_train.png",
        "cm_train.png",
        "pr_train.png",
        "roc_test.png",
        "cm_test.png",
        "pr_test.png",
        "metrics_thresh_train.png",  # Adjusted name
        "metrics_thresh_test.png",  # Adjusted name
        "calib_train.png",  # Adjusted name
        "calib_test.png",  # Adjusted name
    }
    assert set(result.keys()) == expected_keys

    # Verify that plot methods were called correctly
    assert mock_plotter.plot_roc.call_count == 2
    assert mock_plotter.plot_confusion_matrix.call_count == 2
    assert mock_plotter.plot_precision_recall.call_count == 2
    assert mock_plotter.plot_metrics_vs_thresholds.call_count == 2
    assert mock_plotter.plot_calibration_curve.call_count == 2

    # Check that functions are called with the correct arguments, including threshold from mock_model
    for input_type in ["train", "test"]:
        mock_plotter.plot_roc.assert_any_call(
            models={estimator_name: mock_model},
            X_valid=mock.ANY,
            y_valid=mock.ANY,
            custom_name=estimator_name,
            show=False,
        )
        mock_plotter.plot_confusion_matrix.assert_any_call(
            models={estimator_name: mock_model},
            X_valid=mock.ANY,
            y_valid=mock.ANY,
            threshold=0.4,  # Ensure correct threshold is used
            custom_name=estimator_name,
            show=False,
            use_optimal_threshold=True,
        )
        mock_plotter.plot_precision_recall.assert_any_call(
            models={estimator_name: mock_model},
            X_valid=mock.ANY,
            y_valid=mock.ANY,
            custom_name=estimator_name,
            show=False,
        )
        mock_plotter.plot_metrics_vs_thresholds.assert_any_call(
            models={estimator_name: mock_model},
            X_valid=mock.ANY,
            y_valid=mock.ANY,
            custom_name=f"{estimator_name} - Precision, Recall, F1 Score, Specificity vs. Thresholds",
            scoring="accuracy",
            show=False,
        )
        mock_plotter.plot_calibration_curve.assert_any_call(
            models={estimator_name: mock_model},
            X_valid=mock.ANY,
            y_valid=mock.ANY,
            custom_name=f"{estimator_name} - Calibration Curve",  # Match actual function call
            show=False,
        )

    # Verify return values
    assert all(
        value
        in [
            "mock_roc_plot",
            "mock_cm_plot",
            "mock_pr_plot",
            "mock_threshold_plot",
            "mock_calibration_plot",
        ]
        for value in result.values()
    )
