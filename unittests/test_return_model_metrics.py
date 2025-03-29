import pytest
from unittest import mock
import pandas as pd
import numpy as np

from circ_milan.functions import return_model_metrics


@pytest.fixture
def mock_model():
    """Fixture to create a mock model with a return_metrics method."""
    model = mock.Mock()
    model.return_metrics.return_value = {
        "accuracy": 0.8753,
        "precision": 0.9231,
        "recall": 0.7892,
    }
    return model


@pytest.fixture
def sample_inputs():
    """Fixture to provide sample inputs for testing."""
    return {
        "train": (np.array([[1, 2], [3, 4]]), np.array([0, 1])),
        "test": (np.array([[5, 6], [7, 8]]), np.array([1, 0])),
    }


def test_return_model_metrics_basic(
    mock_model,
    sample_inputs,
):
    """Test basic functionality with multiple input types."""
    estimator_name = "LogisticRegression"

    with mock.patch("builtins.print") as mock_print:
        result = return_model_metrics(
            sample_inputs,
            mock_model,
            estimator_name,
        )

    # Check that print was called for each input type
    mock_print.assert_has_calls([mock.call("train"), mock.call("test")])

    # Verify return_metrics was called twice
    assert mock_model.return_metrics.call_count == 2

    # Check that the result is a DataFrame (not a Series)
    assert isinstance(result, pd.DataFrame)

    # Check the index formatting
    expected_index = [
        "train accuracy",
        "train precision",
        "train recall",
        "test accuracy",
        "test precision",
        "test recall",
    ]
    assert result.index.tolist() == expected_index

    # Check rounding to 3 decimal places
    expected_values = [0.875, 0.923, 0.789, 0.875, 0.923, 0.789]
    np.testing.assert_array_almost_equal(
        result.values.flatten(),
        expected_values,
        decimal=3,
    )


def test_return_model_metrics_single_input(mock_model):
    """Test with a single input type."""
    inputs = {"val": (np.array([[1, 2]]), np.array([0]))}
    estimator_name = "RandomForest"

    with mock.patch("builtins.print") as mock_print:
        result = return_model_metrics(
            inputs,
            mock_model,
            estimator_name,
        )

    mock_print.assert_called_once_with("val")
    mock_model.return_metrics.assert_called_once_with(
        inputs["val"][0],
        inputs["val"][1],
        optimal_threshold=True,
        print_threshold=True,
        model_metrics=True,
    )

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3
    assert result.index.tolist() == [
        "val accuracy",
        "val precision",
        "val recall",
    ]
    np.testing.assert_array_almost_equal(
        result.values.flatten(),
        [0.875, 0.923, 0.789],
        decimal=3,
    )


def test_return_model_metrics_empty_metrics(mock_model, sample_inputs):
    """Test when return_metrics returns an empty dict."""
    mock_model.return_metrics.return_value = {}
    estimator_name = "SVM"

    with mock.patch("builtins.print") as mock_print:
        result = return_model_metrics(
            sample_inputs,
            mock_model,
            estimator_name,
        )

    mock_print.assert_has_calls([mock.call("train"), mock.call("test")])
    assert mock_model.return_metrics.call_count == 2

    # Should return an empty DataFrame
    assert isinstance(result, pd.DataFrame)
    assert result.empty
    assert result.index.tolist() == []


def test_return_model_metrics_different_metrics_per_input(
    mock_model,
    sample_inputs,
):
    """
    Test when return_metrics returns different metrics for different inputs.
    """

    def side_effect(X, y, optimal_threshold, print_threshold, model_metrics):
        if np.array_equal(X, sample_inputs["train"][0]):
            return {"accuracy": 0.9124, "f1": 0.8876}
        else:
            return {"precision": 0.7654, "recall": 0.8321}

    mock_model.return_metrics.side_effect = side_effect
    estimator_name = "DecisionTree"

    with mock.patch("builtins.print"):
        result = return_model_metrics(sample_inputs, mock_model, estimator_name)

    assert isinstance(result, pd.DataFrame)
    expected_index = [
        "train accuracy",
        "train f1",
        "test precision",
        "test recall",
    ]
    assert result.index.tolist() == expected_index
    expected_values = [0.912, 0.888, 0.765, 0.832]
    np.testing.assert_array_almost_equal(
        result.values.flatten(),
        expected_values,
        decimal=3,
    )


def test_return_model_metrics_invalid_input_type():
    """Test with invalid input type (non-dict)."""
    invalid_inputs = "not_a_dict"
    mock_model = mock.Mock()
    estimator_name = "KNN"

    with pytest.raises(AttributeError):
        return_model_metrics(
            invalid_inputs,
            mock_model,
            estimator_name,
        )


def test_return_model_metrics_no_rounding_needed(
    mock_model,
    sample_inputs,
):
    """Test when metrics are already rounded."""
    mock_model.return_metrics.return_value = {
        "accuracy": 0.9,
        "precision": 1.0,
        "recall": 0.8,
    }
    estimator_name = "NaiveBayes"

    with mock.patch("builtins.print"):
        result = return_model_metrics(
            sample_inputs,
            mock_model,
            estimator_name,
        )

    expected_values = [0.9, 1.0, 0.8, 0.9, 1.0, 0.8]
    np.testing.assert_array_almost_equal(
        result.values.flatten(),
        expected_values,
        decimal=3,
    )
