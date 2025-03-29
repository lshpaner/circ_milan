from circ_milan.functions import clean_dataframe
import pandas as pd


def test_clean_dataframe():

    df = pd.DataFrame(
        {
            "col1": ["2,300", "-200", "-2,450"],
            "col2": ["", 400, "--"],
            "col3": ["..", "-9..", "-400"],
            "col4": [60, 20, -90],
        }
    )
    cols_with_thousand_separators = ["col1"]
    df_out = clean_dataframe(df, cols_with_thousand_separators)

    print(df_out.dtypes)
    print(df_out)

    assert df_out.dtypes.iloc[0] == int
    assert df_out.dtypes.iloc[1] == float
    assert df_out["col3"].isna().sum() == 2
    assert df_out["col1"].iloc[0] == 2300
