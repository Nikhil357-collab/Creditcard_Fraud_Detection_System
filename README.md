# Creditcard_Fraud_Detection_System
An end-to-end Machine Learning system to detect fraudulent transactions in near real-time using imbalanced classification techniques, deployed with FastAPI and visualized 
# 💳 Credit Card Fraud Detection System

An end-to-end Machine Learning system to detect fraudulent transactions in near real-time using imbalanced classification techniques, deployed with FastAPI and visualized using a Next.js dashboard.
-

## 🚀 Project Highlights

* 🔍 Fraud detection using XGBoost (handles class imbalance)
* ⚖️ Imbalanced learning with `scale_pos_weight`
* 🎯 Threshold tuning for precision-recall trade-off
* ⚡ FastAPI backend for real-time scoring
* 🌐 Next.js setup
* 🚨 Fraud alert system (stream simulation) not yet

## 🧠 Model Details

* Algorithm: XGBoost Classifier
* Dataset: European Credit Card Fraud Dataset
* Imbalance Handling: `scale_pos_weight`
* Evaluation Metrics:

  * ROC-AUC: **0.9799**
  * Fraud Recall: **0.83**
  * Precision: **0.92**

## 📊 Confusion Matrix

|              | Predicted Legit | Predicted Fraud |
| ------------ | --------------- | --------------- |
| Actual Legit | 56857           | 7               |
| Actual Fraud | 17              | 81              |
-

## ⚙️ Tech Stack

### Backend

* Python
* FastAPI
* XGBoost
* Scikit-learn

### Frontend

* Next.js (React + TypeScript)
---

## 📁 Project Structure

```
fraud-detection-system/
│
├── fraud.py
│─ main.py
│─ model.pkl
│── scaler.pkl
│--Predictor.py
|--schema.py
├── dashboard/
│   ├── app/
│   ├── lib/api.ts
│
└── README.md
```

## 🔌 API Endpoints

### POST `/score`

Predict single transaction

### POST `/score/batch`

Batch predictions

### POST `/stream`

## ▶️ Run Locally

### Backend

```
cd backend
uvicorn main:app --reload
```

### Frontend

```
cd frontend
npm install
npm run dev
```
---

## 🧪 Sample Prediction Output
{
  "fraud_probability": 0.0001,
  "is_fraud": 0
}
``--

## 🎯 Key Learnings

* Handling extreme class imbalance in real-world datasets
* Threshold tuning for business-driven decisions
* Building production-ready ML APIs
* Full-stack ML system integration

## 🚀 Future Improvements

* SHAP explainability
* Kafka real-time streaming
* Docker + Cloud deployment
* Advanced fraud analytics dashboard

## 👨‍💻 Author

Nikhil
Aspiring Data Scientist | ML Engineer

---
