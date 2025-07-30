# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import train_test_split
# import joblib
# import pandas as pd

# def train_model(data_path: str, model_path: str):
#     df = pd.read_csv(data_path)
#     df = clean_data(df)           # Reuse from preprocess
#     df = encode_features(df)
    
#     X = df.drop('churn', axis=1)
#     y = df['churn']

#     # Sample for CI speed (use full in prod)
#     X = X.sample(n=min(1000, len(X)))
#     y = y.loc[X.index]

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     model = RandomForestClassifier(n_estimators=10, random_state=42)
#     model.fit(X_train, y_train)

#     preds = model.predict(X_test)
#     acc = accuracy_score(y_test, preds)

#     # Validate minimum accuracy
#     assert acc > 0.6, f"Model accuracy {acc:.2f} below threshold 0.6"

#     joblib.dump(model, model_path)
#     print(f"Model saved to {model_path}, Accuracy: {acc:.2f}")
#     return acc


# src/models/train.py
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

def clean_data(df):
    return df.dropna().drop_duplicates()

def encode_features(df):
    le = LabelEncoder()
    df['gender'] = le.fit_transform(df['gender'].astype(str))
    df['subscription_type'] = le.fit_transform(df['subscription_type'].astype(str))
    return df

if __name__ == "__main__":
    # Load data
    df = pd.read_csv("data/raw/churn_data.csv")
    df = clean_data(df)
    df = encode_features(df)

    X = df.drop("churn", axis=1)
    y = df["churn"]

    # Use only numeric columns
    X = X.select_dtypes(include=["number"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
    model.fit(X_train, y_train)

    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/latest_model.joblib")
    print("✅ Model trained and saved to models/latest_model.joblib")