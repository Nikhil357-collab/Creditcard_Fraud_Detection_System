import joblib
import pandas as pd

MODEL_PATH = "fraud_detectionupdated/fraud_xgb_model.pkl"
SCALER_PATH = "fraud_detectionupdated/scaler.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

THRESHOLD = 0.87

def predict_single(transaction: dict):
    df = pd.DataFrame([transaction])

    # Scale Amount
    df['Amount'] = scaler.transform(df[['Amount']])

    prob = model.predict_proba(df)[:, 1][0]
    pred = int(prob > THRESHOLD)

    return {
        "fraud_probability": float(prob),
        "is_fraud": pred
    }


def predict_batch(transactions: list):
    df = pd.DataFrame(transactions)
    df['Amount'] = scaler.transform(df[['Amount']])

    probs = model.predict_proba(df)[:, 1]
    preds = (probs > THRESHOLD).astype(int)

    return [
        {
            "fraud_probability": float(p),
            "is_fraud": int(pred)
        }
        for p, pred in zip(probs, preds)
    ]