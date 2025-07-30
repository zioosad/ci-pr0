# src/models/train.py
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os
import joblib  # ✅ Add joblib


def train_model(data_path: str, model_path: str = "models/latest_model.joblib"):
    # ❌ Remove autolog
    # mlflow.sklearn.autolog()

    df = pd.read_csv(data_path)
    X = df.drop("churn", axis=1)
    y = df["churn"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run():
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 5)
        mlflow.log_artifact(data_path, "training_data")

        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X_train, y_train)

        # ✅ Log model to MLflow (as artifact)
        mlflow.sklearn.log_model(model, "model")

        # ✅ Save local copy as .joblib file
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)  # This creates the file

    print(f"✅ Model saved to {model_path}")
    return model