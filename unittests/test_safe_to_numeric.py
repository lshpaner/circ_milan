from circ_milan.functions import safe_to_numeric
import pandas as pd


def test_safe_to_numeric_numeric():
    series = pd.Series([1, 2, 3, 4.5, 6.7])
    result = safe_to_numeric(series)
    expected = pd.Series([1, 2, 3, 4.5, 6.7])
    pd.testing.assert_series_equal(result, expected)


def test_safe_to_numeric_mixed():
    series = pd.Series(["1", "2", "three", "4.5", "NaN"])
    result = safe_to_numeric(series)
    expected = pd.Series(["1", "2", "three", "4.5", "NaN"])
    pd.testing.assert_series_equal(result, expected)


def test_safe_to_numeric_invalid():
    series = pd.Series(["apple", "banana", "cherry"])
    result = safe_to_numeric(series)
    expected = pd.Series(["apple", "banana", "cherry"])
    pd.testing.assert_series_equal(result, expected)


def test_safe_to_numeric_nan():
    series = pd.Series(["1", "2", None, "4.5"])
    result = safe_to_numeric(series)
    expected = pd.Series([1.0, 2.0, None, 4.5])
    pd.testing.assert_series_equal(result, expected)


def test_safe_to_numeric_empty():
    series = pd.Series([])
    result = safe_to_numeric(series)
    expected = pd.Series(dtype="int")
    print(result, expected)
    pd.testing.assert_series_equal(result, expected)


def test_safe_to_numeric_boolean():
    series = pd.Series([True, False, True])
    result = safe_to_numeric(series)
    expected = pd.to_numeric(series, downcast="integer")  # Ensures dtype consistency
    pd.testing.assert_series_equal(result, expected)
