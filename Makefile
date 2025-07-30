train:
    python src/models/train.py data/raw/churn_data.csv

serve:
    uvicorn app.main:app --reload --port 8000

predict-test:
    curl -X POST http://127.0.0.1:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"gender":"Male","age":35,"subscription_type":"Premium","monthly_bill":89.99,"tenure":24}'

dev: train serve