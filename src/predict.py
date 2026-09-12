"""
Inference module for Telco Customer Churn Prediction.
Loads serialized champion pipeline, performs preprocessing & feature engineering,
and produces churn probabilities, risk levels, and actionable explanations.
"""

import os
import logging
from typing import Dict, Any, Union, List
import pandas as pd
import numpy as np
import joblib

from data_prep import clean_data, engineer_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH = "model/churn_model.pkl"
_cached_pipeline = None


def get_pipeline():
    """Load and cache the trained champion model pipeline."""
    global _cached_pipeline
    if _cached_pipeline is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model file not found at '{MODEL_PATH}'. Run 'python src/train.py' first."
            )
        logger.info("Loading serialized pipeline from %s", MODEL_PATH)
        _cached_pipeline = joblib.load(MODEL_PATH)
    return _cached_pipeline


def generate_actionable_insights(record: Dict[str, Any], proba: float) -> List[str]:
    """Generate business insights and retention drivers for the customer."""
    insights = []

    contract = str(record.get("Contract", "")).lower()
    tenure = float(record.get("tenure", 0))
    monthly = float(record.get("MonthlyCharges", 0))
    internet = str(record.get("InternetService", "")).lower()
    payment = str(record.get("PaymentMethod", "")).lower()
    tech_support = str(record.get("TechSupport", "")).lower()
    online_sec = str(record.get("OnlineSecurity", "")).lower()

    if "month" in contract:
        insights.append("Month-to-month contract increases cancellation flexibility.")
    if tenure <= 12:
        insights.append(f"Early lifecycle stage ({int(tenure)} months tenure) represents highest churn vulnerability.")
    if "fiber" in internet and monthly > 70:
        insights.append(f"High monthly spend (${monthly:.2f}) on Fiber Optic without long-term commitment.")
    if "electronic check" in payment:
        insights.append("Electronic check payment correlates with higher payment friction and churn.")
    if tech_support in ["no", "no internet service"]:
        insights.append("Absence of Tech Support increases likelihood of unresolved service frustration.")
    if online_sec in ["no", "no internet service"]:
        insights.append("Lack of Online Security add-on correlates with lower account stickiness.")

    if not insights:
        insights.append("Account metrics indicate stable tenure and balanced service usage.")

    return insights


def recommend_retention_action(proba: float, record: Dict[str, Any]) -> str:
    """Recommend high-ROI business intervention based on churn probability and profile."""
    contract = str(record.get("Contract", "")).lower()
    payment = str(record.get("PaymentMethod", "")).lower()

    if proba >= 0.65:
        if "month" in contract:
            return "Urgent: Offer 15% discount on an annual contract upgrade with complimentary Tech Support."
        return "Urgent: Assign proactive account manager and conduct satisfaction audit."
    elif proba >= 0.35:
        if "electronic check" in payment:
            return "Medium: Promote $5/mo autopay credit for switching to automated bank/card payments."
        return "Medium: Enroll in loyalty rewards program and offer bundled security add-on."
    else:
        return "Low: Standard retention health; target for cross-selling and referral rewards."


def predict_churn(customer_input: Union[Dict[str, Any], pd.DataFrame, List[Dict[str, Any]]]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Predict churn probability and classification for one or more customer profiles.
    
    Parameters:
        customer_input: dict or DataFrame containing customer attributes.
        
    Returns:
        Dictionary with prediction results, probability, risk level, and insights.
    """
    pipeline = get_pipeline()

    if isinstance(customer_input, dict):
        df_input = pd.DataFrame([customer_input])
        is_single = True
    elif isinstance(customer_input, list):
        df_input = pd.DataFrame(customer_input)
        is_single = False
    elif isinstance(customer_input, pd.DataFrame):
        df_input = customer_input.copy()
        is_single = len(df_input) == 1
    else:
        raise ValueError("Input must be a dict, list of dicts, or pandas DataFrame.")

    # Apply data cleaning & feature engineering
    cleaned_df = clean_data(df_input)
    featured_df = engineer_features(cleaned_df)

    # Predict probabilities and labels
    probabilities = pipeline.predict_proba(featured_df)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    results = []
    records = df_input.to_dict(orient="records")

    for i, (pred, proba) in enumerate(zip(predictions, probabilities)):
        proba_val = float(np.round(proba, 4))

        if proba_val >= 0.65:
            risk = "High"
        elif proba_val >= 0.35:
            risk = "Medium"
        else:
            risk = "Low"

        record_dict = records[i]
        insights = generate_actionable_insights(record_dict, proba_val)
        recommendation = recommend_retention_action(proba_val, record_dict)

        results.append({
            "churn_prediction": int(pred),
            "churn_label": "Yes" if pred == 1 else "No",
            "churn_probability": proba_val,
            "risk_level": risk,
            "key_factors": insights,
            "recommended_action": recommendation
        })

    return results[0] if is_single else results


if __name__ == "__main__":
    # Test with sample high-risk customer profile
    sample_customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.35,
        "TotalCharges": 140.7
    }
    res = predict_churn(sample_customer)
    print("\n--- Smoke Test Prediction Result ---")
    for k, v in res.items():
        print(f"{k}: {v}")
