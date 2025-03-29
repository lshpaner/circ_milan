import pytest
import pandas as pd
import numpy as np
from circ_milan.functions import create_stratified_other_column


# Fixture to create sample data with PatientID as index
@pytest.fixture
def sample_df():
    data = {
        "PatientID": [1, 2, 3, 4],
        "race_white": [1, 0, 0, 1],
        "race_black": [0, 1, 0, 0],
        "race_asian": [0, 0, 1, 0],
        "race_other": [0, 0, 0, 0],
        "age": [25, 35, 45, 55],
    }
    df = pd.DataFrame(data)
    return df.set_index("PatientID")  # Set PatientID as index


def test_basic_functionality(sample_df):
    """Test basic functionality with stratification list only"""
    result = create_stratified_other_column(
        X=sample_df,
        stratify_list=["race_white", "race_black"],
        patient_id="PatientID",
    )
    assert isinstance(result, pd.DataFrame)
    assert set(["race_white", "race_black"]).issubset(result.columns)
    assert len(result) == len(sample_df)


def test_other_column_creation(sample_df):
    """Test creation and combination of other_column"""
    result = create_stratified_other_column(
        X=sample_df,
        stratify_list=["race_white"],
        other_columns=["race_asian", "race_black"],
        other_column="race_other",
        patient_id="PatientID",
    )
    assert "race_other" in result.columns
    expected = sample_df[["race_asian", "race_black"]].max(axis=1)
    pd.testing.assert_series_equal(
        result["race_other"],
        expected,
        check_names=False,
    )


def test_age_binning(sample_df):
    """Test age binning functionality"""
    bin_ages = [0, 30, 40, 100]
    result = create_stratified_other_column(
        X=sample_df,
        stratify_list=["race_white"],
        age="age",
        age_bin="age_group",
        bin_ages=bin_ages,
        patient_id="PatientID",
    )
    assert "age_group" in result.columns
    assert result["age_group"].nunique() > 1
    assert all(result["age_group"].notna())


def test_missing_age_handling(sample_df):
    """Test handling of missing age values"""
    df_with_na = sample_df.copy()
    df_with_na.loc[1, "age"] = np.nan  # Use index 1 since PatientID is now index
    bin_ages = [0, 30, 40, 100]
    result = create_stratified_other_column(
        X=df_with_na,
        stratify_list=["race_white"],
        age="age",
        age_bin="age_group",
        bin_ages=bin_ages,
        patient_id="PatientID",
    )
    assert "age_group" in result.columns
    assert result["age_group"].notna().all()


def test_empty_stratify_list(sample_df):
    """Test behavior with empty stratify list"""
    result = create_stratified_other_column(
        X=sample_df,
        stratify_list=[],
        other_columns=["race_asian"],
        other_column="race_other",
        patient_id="PatientID",
    )
    assert isinstance(result, pd.DataFrame)
    assert "race_other" in result.columns
    assert len(result) == len(sample_df)


def test_no_other_column_modification(sample_df):
    """Test when other_columns is empty"""
    original = sample_df["race_other"].copy()
    result = create_stratified_other_column(
        X=sample_df,
        stratify_list=["race_white"],
        other_columns=[],
        other_column="race_other",
        patient_id="PatientID",
    )
    pd.testing.assert_series_equal(result["race_other"], original)


def test_invalid_input():
    """Test error handling for invalid input"""
    with pytest.raises(AttributeError):
        create_stratified_other_column(
            X=None,
            stratify_list=["race_white"],
            patient_id="PatientID",
        )


def test_column_not_in_dataframe(sample_df):
    """Test behavior when requested column doesn't exist"""
    with pytest.raises(KeyError):
        create_stratified_other_column(
            X=sample_df,
            stratify_list=["non_existent_column"],
            patient_id="PatientID",
        )


@pytest.mark.parametrize(
    "bin_ages,expected_bins",
    [
        ([0, 30, 60], 2),
        ([0, 20, 40, 60], 3),
    ],
)
def test_different_age_bins(sample_df, bin_ages, expected_bins):
    """Test different age bin configurations"""
    result = create_stratified_other_column(
        X=sample_df,
        stratify_list=["race_white"],
        age="age",
        age_bin="age_group",
        bin_ages=bin_ages,
        patient_id="PatientID",
    )
    assert result["age_group"].nunique() <= expected_bins


if __name__ == "__main__":
    pytest.main()
