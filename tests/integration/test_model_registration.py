# tests/integration/test_model_registration.py
import mlflow
import pytest
import subprocess
import time

@pytest.mark.integration
def test_model_is_registered(sample_csv):
    # Start MLflow server if not running
    # (Or assume it's already up in CI)

    mlflow.set_tracking_uri("http://localhost:5000")

    # Train model (assumes it registers)
    train_model(sample_csv)

    # Check registry
    client = mlflow.MlflowClient()
    try:
        versions = client.search_model_versions("name='ChurnModel'")
        assert len(versions) >= 1
        assert versions[0].status == "READY"
    except mlflow.exceptions.MlflowException:
        pytest.fail("Model not registered")