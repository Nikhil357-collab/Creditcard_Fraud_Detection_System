from fastapi import FastAPI
from schema import Transaction, BatchTransaction
from predictor import predict_single, predict_batch
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# ✅ CORS (required)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Load model once
model = joblib.load("fraud_xgb_model.pkl")
scaler = joblib.load("scaler.pkl")

THRESHOLD = 0.87


class mainTransaction(BaseModel):
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float


@app.get("/")
def home():
    return {"message": "API running"}


@app.post("/score")
def score(transaction: Transaction):
    df = pd.DataFrame([transaction.dict()])

    df["Amount"] = scaler.transform(df[["Amount"]])

    prob = model.predict_proba(df)[:, 1][0]
    pred = int(prob > THRESHOLD)

    return {
        "fraud_probability": float(prob),
        "is_fraud": pred
    }
# =========================
# BATCH SCORING
# =========================
@app.post("/score/batch")
def score_batch(batch: BatchTransaction):
    transactions = [t.dict() for t in batch.transactions]
    result = predict_batch(transactions)
    return result

# =========================
# STREAM SIMULATION
# =========================
@app.post("/stream")
def stream(transaction: Transaction):
    result = predict_single(transaction.dict())

    # Simulate alert trigger
    if result["is_fraud"] == 1:
        alert = "🚨 FRAUD ALERT TRIGGERED"
    else:
        alert = "✅ Normal transaction"

    return {
        "result": result,
        "alert": alert
    }