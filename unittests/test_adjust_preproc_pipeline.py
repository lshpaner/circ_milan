import pytest
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

from circ_milan.functions import adjust_preprocessing_pipeline


# Fixtures
@pytest.fixture
def numerical_cols():
    return ["age", "income"]


@pytest.fixture
def categorical_cols():
    return ["gender"]


@pytest.fixture
def base_preprocessor(numerical_cols, categorical_cols):
    numerical_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("imputer", SimpleImputer(strategy="mean")),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="missing",
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="passthrough",
    )


@pytest.fixture
def pipeline_no_rfe(base_preprocessor):
    return [("Preprocessor", base_preprocessor)]


@pytest.fixture
def pipeline_with_rfe(base_preprocessor):
    rfe = RFE(estimator=LogisticRegression(), step=0.1)
    return [("Preprocessor", base_preprocessor), ("RFE", rfe)]


@pytest.fixture
def smote_sampler():
    return SMOTE(random_state=42)


# Tests
def test_rf_no_rfe_no_smote(
    pipeline_no_rfe,
    numerical_cols,
    categorical_cols,
):
    """
    Test RandomForest without RFE or SMOTE: pipeline unchanged.
    """
    adjusted_steps = adjust_preprocessing_pipeline(
        "rf",
        pipeline_no_rfe,
        numerical_cols,
        categorical_cols,
        sampler=None,
    )
    assert adjusted_steps == pipeline_no_rfe  # Unchanged for rf
    preprocessor = adjusted_steps[0][1]
    num_transformer = preprocessor.transformers[0][1]
    cat_transformer = preprocessor.transformers[1][1]
    assert len(num_transformer.steps) == 2  # Scaler + imputer
    assert isinstance(num_transformer.steps[0][1], StandardScaler)
    assert isinstance(num_transformer.steps[1][1], SimpleImputer)
    assert len(cat_transformer.steps) == 2  # Imputer + encoder
    assert isinstance(cat_transformer.steps[0][1], SimpleImputer)
    assert isinstance(cat_transformer.steps[1][1], OneHotEncoder)


def test_rf_with_rfe_no_smote(
    pipeline_with_rfe,
    numerical_cols,
    categorical_cols,
):
    """
    Test RandomForest with RFE, no SMOTE: pipeline unchanged.
    """
    adjusted_steps = adjust_preprocessing_pipeline(
        "rf",
        pipeline_with_rfe,
        numerical_cols,
        categorical_cols,
        sampler=None,
    )
    assert len(adjusted_steps) == 2
    assert adjusted_steps == pipeline_with_rfe  # Unchanged for rf
    preprocessor = adjusted_steps[0][1]
    num_transformer = preprocessor.transformers[0][1]
    cat_transformer = preprocessor.transformers[1][1]
    assert len(num_transformer.steps) == 2
    assert isinstance(num_transformer.steps[0][1], StandardScaler)
    assert isinstance(num_transformer.steps[1][1], SimpleImputer)
    assert len(cat_transformer.steps) == 2
    assert isinstance(cat_transformer.steps[0][1], SimpleImputer)
    assert isinstance(cat_transformer.steps[1][1], OneHotEncoder)


def test_xgb_no_rfe_no_smote(
    pipeline_no_rfe,
    numerical_cols,
    categorical_cols,
):
    """
    Test XGBoost without RFE or SMOTE: no scaling, no imputation.
    """
    adjusted_steps = adjust_preprocessing_pipeline(
        "xgb",
        pipeline_no_rfe,
        numerical_cols,
        categorical_cols,
        sampler=None,
    )
    preprocessor = adjusted_steps[0][1]
    num_transformer = preprocessor.transformers[0][1]
    cat_transformer = preprocessor.transformers[1][1]
    assert len(num_transformer.steps) == 1
    assert num_transformer.steps[0][0] == "passthrough"
    assert len(cat_transformer.steps) == 1
    assert isinstance(cat_transformer.steps[0][1], OneHotEncoder)


def test_xgb_no_rfe_with_smote(
    pipeline_no_rfe,
    numerical_cols,
    categorical_cols,
    smote_sampler,
):
    """
    Test XGBoost without RFE, with SMOTE: imputation, no scaling.
    """
    adjusted_steps = adjust_preprocessing_pipeline(
        "xgb",
        pipeline_no_rfe,
        numerical_cols,
        categorical_cols,
        sampler=smote_sampler,
    )
    preprocessor = adjusted_steps[0][1]
    num_transformer = preprocessor.transformers[0][1]
    cat_transformer = preprocessor.transformers[1][1]
    assert len(num_transformer.steps) == 1
    assert isinstance(num_transformer.steps[0][1], SimpleImputer)
    assert num_transformer.steps[0][1].strategy == "mean"
    assert len(cat_transformer.steps) == 2
    assert isinstance(cat_transformer.steps[0][1], SimpleImputer)
    assert cat_transformer.steps[0][1].strategy == "constant"
    assert cat_transformer.steps[0][1].fill_value == "missing"
    assert isinstance(cat_transformer.steps[1][1], OneHotEncoder)


def test_cat_with_rfe_no_smote(
    pipeline_with_rfe,
    numerical_cols,
    categorical_cols,
):
    """
    Test CatBoost with RFE, no SMOTE: imputation, no scaling.
    """
    adjusted_steps = adjust_preprocessing_pipeline(
        "cat",
        pipeline_with_rfe,
        numerical_cols,
        categorical_cols,
        sampler=None,
    )
    preprocessor = adjusted_steps[0][1]
    num_transformer = preprocessor.transformers[0][1]
    cat_transformer = preprocessor.transformers[1][1]
    assert len(num_transformer.steps) == 1
    assert isinstance(num_transformer.steps[0][1], SimpleImputer)
    assert len(cat_transformer.steps) == 2
    assert isinstance(cat_transformer.steps[0][1], SimpleImputer)
    assert isinstance(cat_transformer.steps[1][1], OneHotEncoder)


def test_cat_with_rfe_with_smote(
    pipeline_with_rfe,
    numerical_cols,
    categorical_cols,
    smote_sampler,
):
    """
    Test CatBoost with RFE and SMOTE: imputation, no scaling.
    """
    adjusted_steps = adjust_preprocessing_pipeline(
        "cat",
        pipeline_with_rfe,
        numerical_cols,
        categorical_cols,
        sampler=smote_sampler,
    )
    preprocessor = adjusted_steps[0][1]
    num_transformer = preprocessor.transformers[0][1]
    cat_transformer = preprocessor.transformers[1][1]
    assert len(num_transformer.steps) == 1
    assert isinstance(num_transformer.steps[0][1], SimpleImputer)
    assert len(cat_transformer.steps) == 2
    assert isinstance(cat_transformer.steps[0][1], SimpleImputer)
    assert isinstance(cat_transformer.steps[1][1], OneHotEncoder)


def test_lr_no_rfe_no_smote(
    pipeline_no_rfe,
    numerical_cols,
    categorical_cols,
    base_preprocessor,
):
    """
    Test Logistic Regression without RFE or SMOTE: pipeline unchanged.
    """
    adjusted_steps = adjust_preprocessing_pipeline(
        "lr", pipeline_no_rfe, numerical_cols, categorical_cols, sampler=None
    )
    assert adjusted_steps == pipeline_no_rfe
    assert adjusted_steps[0][1] is base_preprocessor


def test_lr_with_rfe_with_smote(
    pipeline_with_rfe,
    numerical_cols,
    categorical_cols,
    base_preprocessor,
    smote_sampler,
):
    """
    Test Logistic Regression with RFE and SMOTE: pipeline unchanged.
    """
    adjusted_steps = adjust_preprocessing_pipeline(
        "lr",
        pipeline_with_rfe,
        numerical_cols,
        categorical_cols,
        sampler=smote_sampler,
    )
    assert adjusted_steps == pipeline_with_rfe
    assert adjusted_steps[0][1] is base_preprocessor
    assert isinstance(adjusted_steps[1][1], RFE)
