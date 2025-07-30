import pytest
from unittest import mock
import pandas as pd
import os
from src.models.train import train_model

@pytest.fixture
def sample_csv(tmp_path):
    """Create a temporary CSV file."""
    df = pd.DataFrame({
        'age': [25, 30, 35],
        'salary': [50000, 60000, 70000],
        'churn': [0, 1, 0]
    })
    csv_path = tmp_path / "test_data.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)

@mock.patch("mlflow.start_run")
@mock.patch("mlflow.log_param")
@mock.patch("mlflow.log_artifact")
@mock.patch("mlflow.sklearn.log_model")
@mock.patch("mlflow.sklearn.save_model")
def test_train_model_logs_to_mlflow(
    mock_save_model,
    mock_log_model,
    mock_log_artifact,
    mock_log_param,
    mock_start_run,
    sample_csv,
    tmp_path
):
    # Arrange
    model_output = tmp_path / "test_model.joblib"

    # Act
    train_model(sample_csv, str(model_output))

    # Assert: MLflow was called correctly
    assert mock_start_run.call_count == 1
    assert mock_log_param.call_count >= 3
   
    mock_log_param.assert_any_call("model_type", "RandomForest")
    mock_log_param.assert_any_call("n_estimators", 100)
    mock_log_param.assert_any_call("max_depth", 5)

    # Check data was logged
    mock_log_artifact.assert_any_call(sample_csv, "training_data")

    # Check model was logged and saved
    mock_log_model.assert_called()
    mock_save_model.assert_called_with(mock.ANY, str(model_output))

    # Inside test function
    model_output = tmp_path / "test_model" 
    train_model(sample_csv, str(model_output))
    assert os.path.exists(model_output)
    assert os.path.isdir(model_output)
    # assert os.path.exists(model_output)