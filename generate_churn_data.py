import pandas as pd
import numpy as np
import os

np.random.seed(42)

n = 1000  # Number of customers

data = {
    "customer_id": np.arange(10001, 10001 + n),
    "age": np.random.randint(18, 80, size=n),
    "gender": np.random.choice(["Male", "Female"], size=n),
    "tenure_months": np.random.randint(0, 72, size=n),
    "subscription_type": np.random.choice(["Basic", "Standard", "Premium"], size=n, p=[0.5, 0.3, 0.2]),
    "monthly_bill": np.round(np.random.uniform(20, 150, size=n), 2),
    "total_usage_gb": np.round(np.random.uniform(10, 1000, size=n), 1),
    "support_calls": np.random.poisson(2, size=n),
    "account_balance": np.round(np.random.uniform(-50, 500, size=n), 2),
    "payment_delay_days": np.random.poisson(3, size=n),
    "satisfaction_score": np.random.randint(1, 11, size=n),
}

df = pd.DataFrame(data)

# Simulate churn logic
churn_prob = (
    (df["tenure_months"] < 6) * 0.5 +
    (df["support_calls"] > 4) * 0.4 +
    (df["payment_delay_days"] > 5) * 0.4 +
    (df["satisfaction_score"] < 4) * 0.6 +
    np.random.normal(0, 0.05, size=n)
)
df["churn"] = (np.clip(churn_prob, 0, 1) > 0.5).astype(int)

# Ensure directory exists
os.makedirs("data/raw", exist_ok=True)

# Save real CSV
df.to_csv("data/raw/churn_data.csv", index=False)

print("✅ Real churn dataset generated!")
print(f"📊 Shape: {df.shape}")
print(f"📈 Churn distribution:\n{df['churn'].value_counts()}")
print(f"💾 File size: {os.path.getsize('data/raw/churn_data.csv')} bytes")