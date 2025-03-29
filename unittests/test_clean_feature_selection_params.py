from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from circ_milan.functions import clean_feature_selection_params


def test_clean_feature_selection_params_with_rfe():
    """Test when RFE is present in the pipeline."""
    pipeline_steps = [
        ("scaler", None),
        ("feature_selection", RFE(estimator=LogisticRegression())),
        ("classifier", LogisticRegression()),
    ]
    tuned_parameters = [
        {
            "feature_selection__n_features_to_select": [5, 10],
            "classifier__C": [0.1, 1.0, 10.0],
        }
    ]

    clean_feature_selection_params(pipeline_steps, tuned_parameters)

    # Ensure feature_selection-related params remain
    assert "feature_selection__n_features_to_select" in tuned_parameters[0]
    assert "classifier__C" in tuned_parameters[0]


def test_clean_feature_selection_params_without_rfe():
    """Test when RFE is not present in the pipeline."""
    pipeline_steps = [("scaler", None), ("classifier", RandomForestClassifier())]
    tuned_parameters = [
        {
            "feature_selection__n_features_to_select": [5, 10],
            "classifier__n_estimators": [100, 200],
        }
    ]

    clean_feature_selection_params(pipeline_steps, tuned_parameters)

    # Ensure feature_selection-related params are removed
    assert "feature_selection__n_features_to_select" not in tuned_parameters[0]
    assert "classifier__n_estimators" in tuned_parameters[0]


def test_clean_feature_selection_params_empty_pipeline():
    """Test when pipeline_steps is empty."""
    pipeline_steps = []
    tuned_parameters = [{"feature_selection__n_features_to_select": [5, 10]}]

    clean_feature_selection_params(pipeline_steps, tuned_parameters)

    # Ensure feature_selection-related params are removed
    assert "feature_selection__n_features_to_select" not in tuned_parameters[0]


def test_clean_feature_selection_params_empty_tuned_parameters():
    """Test when tuned_parameters is empty."""
    pipeline_steps = [("classifier", LogisticRegression())]
    tuned_parameters = [{}]

    clean_feature_selection_params(pipeline_steps, tuned_parameters)

    # Ensure tuned_parameters remains empty
    assert tuned_parameters == [{}]


def test_clean_feature_selection_params_no_feature_related_keys():
    """Test when no feature_selection-related keys are present."""
    pipeline_steps = [("classifier", LogisticRegression())]
    tuned_parameters = [{"classifier__C": [0.1, 1.0]}]

    clean_feature_selection_params(pipeline_steps, tuned_parameters)

    # Ensure no changes are made
    assert "classifier__C" in tuned_parameters[0]
