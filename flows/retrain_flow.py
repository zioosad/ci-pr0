# flows/retrain_flow.py
from prefect import flow, task
from evidently.report import Report
from evidently.metrics import DataDriftPreset
import pandas as pd
import subprocess
import requests

@task
def fetch_new_data():
    subprocess.run(["aws", "s3", "cp", "s3://my-mlops-data/feedback_data.csv", "data/feedback_data.csv"])
    return pd.read_csv("data/feedback_data.csv")

@task
def detect_drift(current_df):
    reference_df = pd.read_csv("data/raw/churn_data.csv")  # Or last training set
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_df, current_data=current_df)
    result = report.as_dict()
    return result['metrics'][0]['result']['dataset_drift']

@task
def trigger_retraining():
    repo = "zioosad/ci-pr0"
    token = "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"

    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    data = {
        "event_type": "retrain-trigger",
        "client_payload": {
            "source": "prefect-flow",
            "data_date": pd.Timestamp.now().strftime("%Y-%m-%d")
        }
    }
    r = requests.post(f"https://api.github.com/repos/{repo}/dispatches", json=data, headers=headers)
    r.raise_for_status()
    print("Retraining triggered on GitHub")

@flow(name="Continuous Training Pipeline")
def continuous_training():
    new_data = fetch_new_data()
    drift = detect_drift(new_data)
    
    if drift:
        print("⚠️ Data drift detected! Retraining model...")
        trigger_retraining()
    else:
        print("✅ No drift. Skipping retraining.")

# Run locally
if __name__ == "__main__":
    continuous_training()