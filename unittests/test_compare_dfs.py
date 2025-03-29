import pandas as pd
from circ_milan.functions import compare_dataframes

# Helper function to capture print output
from io import StringIO
import sys


def capture_output(func, *args, **kwargs):
    """Helper function to capture print statements from compare_dataframes."""
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    try:
        func(*args, **kwargs)
    finally:
        sys.stdout = old_stdout
    return mystdout.getvalue()


# Test cases
def test_identical_dataframes():
    df1 = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    df2 = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

    output = capture_output(compare_dataframes, df1, df2)
    assert "No differences found between the DataFrames!" in output


def test_different_shapes():
    df1 = pd.DataFrame({"A": [1, 2, 3]})
    df2 = pd.DataFrame({"A": [1, 2, 3, 4]})

    output = capture_output(compare_dataframes, df1, df2)
    assert "DataFrames have different shapes" in output


def test_different_columns():
    df1 = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    df2 = pd.DataFrame(
        {"A": [1, 2, 3], "C": [4, 5, 6]}
    )  # Column name 'C' instead of 'B'

    output = capture_output(compare_dataframes, df1, df2)
    assert "DataFrames have different columns" in output


def test_different_data_types():
    df1 = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    df2 = pd.DataFrame(
        {"A": [1, 2, 3], "B": [4.0, 5.0, 6.0]}
    )  # B is float instead of int

    output = capture_output(compare_dataframes, df1, df2)
    assert "DataFrames have different data types" in output


def test_different_content():
    df1 = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    df2 = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 9]})  # Last value changed to 9

    output = capture_output(compare_dataframes, df1, df2)
    assert "DataFrames have different content" in output
