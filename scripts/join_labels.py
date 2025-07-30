import pandas as pd
import json

with open("logs/predictions.log") as f:
    preds = [json.loads(line) for line in f]

df_preds = pd.DataFrame(preds)
df_preds['timestamp'] = pd.to_datetime(df_preds['timestamp'])

# Load actual churn data (from CRM, DB, etc.)
df_actual = pd.read_csv("data/actual_churn.csv")
df_actual['churn_date'] = pd.to_datetime(df_actual['churn_date'])

# Join: Match prediction with actual outcome after delay
df_actual['join_date'] = df_actual['churn_date'] - pd.Timedelta(days=1)
df_joined = pd.merge_asof(
    df_preds.sort_values('timestamp'),
    df_actual[['user_id', 'churn', 'join_date']].sort_values('join_date'),
    left_on='timestamp',
    right_on='join_date',
    tolerance=pd.Timedelta(days=1),
    direction='nearest'
)
df_joined.to_csv("data/feedback_data.csv", index=False)