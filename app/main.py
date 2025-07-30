# app/main.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse  # ✅ Correct import
import joblib
import pandas as pd
from pydantic import BaseModel
import os

app = FastAPI(title="Churn Prediction API", version="1.0")

MODEL_PATH = "models/latest_model.joblib"
model = None

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded from {MODEL_PATH}")
else:
    print(f"⚠️ Model not found at {MODEL_PATH}")

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <h1>🎯 Churn Prediction API</h1>
    <p><strong>Model loaded:</strong> """ + ("Yes" if model is not None else "No") + """</p>
    <h3>✅ Endpoints:</h3>
    <ul>
        <li><a href="/health"><code>GET /health</code></a></li>
        <li><code>POST /predict</code></li>
    </ul>
    """

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

class ChurnInput(BaseModel):
    gender: str
    age: int
    subscription_type: str
    monthly_bill: float
    tenure: int

class ChurnOutput(BaseModel):
    prediction: int
    probability: float

@app.post("/predict", response_model=ChurnOutput)
def predict_churn( ChurnInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train first.")

    try:
        gender_encoded = 1 if data.gender.lower() == 'male' else 0
        sub_encoded = 1 if data.subscription_type == 'Premium' else 0

        input_df = pd.DataFrame([{
            'gender': gender_encoded,
            'age': data.age,
            'subscription_type': sub_encoded,
            'monthly_bill': data.monthly_bill,
            'tenure': data.tenure,
        }])

        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0].max()

        return {"prediction": int(pred), "probability": float(proba)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))