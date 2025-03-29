from circ_milan.functions import top_n
import pandas as pd


def test_top_n():
    series = pd.Series(
        [
            "apple",
            "banana",
            "apple",
            "cherry",
            "apple",
            "banana",
            "banana",
            "cherry",
            "cherry",
            "cherry",
            "date",
            "fig",
        ]
    )
    result = top_n(series, n=3)
    expected = {"apple", "banana", "cherry"}  # The three most common values
    assert result == expected


def test_top_n_smaller_than_n():
    series = pd.Series(["grape", "melon", "kiwi"])
    result = top_n(series, n=5)
    expected = {"grape", "melon", "kiwi"}  # All unique values since n is larger
    assert result == expected


def test_top_n_empty():
    series = pd.Series([])
    result = top_n(series, n=3)
    expected = set()
    assert result == expected
