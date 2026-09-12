"""
FastAPI REST application for customer churn prediction inference.
Provides single customer scoring, batch inference, and health check endpoints.
"""

import sys
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

# Add src and api directories to system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from predict import get_pipeline, predict_churn
try:
    from api.schema import (
        CustomerInput,
        PredictionResponse,
        BatchCustomerInput,
        BatchPredictionResponse,
        HealthResponse
    )
except ImportError:
    from schema import (
        CustomerInput,
        PredictionResponse,
        BatchCustomerInput,
        BatchPredictionResponse,
        HealthResponse
    )

MODEL_NAME = "LogisticRegression Champion"
API_VERSION = "1.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model pipeline into memory on server startup."""
    try:
        get_pipeline()
        print("✅ Champion model pipeline loaded successfully into memory.")
    except Exception as e:
        print(f"⚠️ Model not loaded on startup: {e}")
    yield


app = FastAPI(
    title="Customer Churn Prediction API",
    description="Production-ready REST API for real-time customer churn scoring, risk stratification, and retention plays.",
    version=API_VERSION,
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["General"])
def root():
    return {
        "message": "Customer Churn Prediction API is active.",
        "documentation": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check():
    """System health check and model status."""
    is_loaded = False
    try:
        get_pipeline()
        is_loaded = True
    except Exception:
        is_loaded = False

    return HealthResponse(
        status="healthy" if is_loaded else "degraded",
        model_loaded=is_loaded,
        champion_model=MODEL_NAME,
        version=API_VERSION
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inference"]
)
def predict_single(customer: CustomerInput):
    """Score a single customer profile for churn probability and retention actions."""
    try:
        data = customer.model_dump()
        if data.get("TotalCharges") is None:
            data["TotalCharges"] = round(data["MonthlyCharges"] * max(data["tenure"], 1), 2)

        result = predict_churn(data)
        return PredictionResponse(**result)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(exc)}"
        )


@app.post(
    "/predict-batch",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inference"]
)
def predict_batch(payload: BatchCustomerInput):
    """Score multiple customer profiles in bulk."""
    try:
        records = []
        for cust in payload.customers:
            item = cust.model_dump()
            if item.get("TotalCharges") is None:
                item["TotalCharges"] = round(item["MonthlyCharges"] * max(item["tenure"], 1), 2)
            records.append(item)

        results = predict_churn(records)
        high_risk_count = sum(1 for r in results if r["risk_level"] == "High")

        return BatchPredictionResponse(
            total_customers=len(results),
            high_risk_count=high_risk_count,
            predictions=[PredictionResponse(**r) for r in results]
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference error: {str(exc)}"
        )
