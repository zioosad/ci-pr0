import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    file_size = os.path.getsize(path)
    if file_size < 100:  # Too small for real data
        raise ValueError(
            f"File too small ({file_size} bytes): likely corrupted: {path}"
        )

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Loaded DataFrame is empty")
    if "churn" not in df.columns:
        raise ValueError("Missing 'churn' column")

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna()
    df = df.drop_duplicates()
    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    le = LabelEncoder()
    categorical_cols = ["gender", "subscription_type"]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))
    return df
