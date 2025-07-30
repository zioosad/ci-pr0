import os
import pandas as pd
import pytest
from src.data.preprocess import load_data, clean_data, encode_features

# def test_load_data():
#     df = load_data("data/raw/churn_data.csv")
#     assert not df.empty
#     assert 'churn' in df.columns


def test_load_data():
    print("Current working directory:", os.getcwd())
    print("Attempting to load: data/raw/churn_data.csv")

    df = load_data("data/raw/churn_data.csv")
    # df = load_data("../../data/raw/churn_data.csv")
    print("Loaded DataFrame:")
    print(df)
    print("Columns:", df.columns.tolist())

    assert not df.empty
    assert "churn" in df.columns


def test_clean_data():
    df = pd.DataFrame({"A": [1, 2, None, 4], "B": [5, None, None, 8]})
    cleaned = clean_data(df)
    assert cleaned.shape[0] == 2  # Only 2 non-NaN rows


def test_encode_features():
    df = pd.DataFrame({"gender": ["Male", "Female", "Male"], "age": [25, 30, 35]})
    encoded = encode_features(df)
    assert encoded["gender"].dtype == "int64"
    assert encoded["gender"].nunique() <= 2
