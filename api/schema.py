"""
Pydantic schemas for the Telco Churn Prediction FastAPI service.
Provides data validation, swagger docs examples, and typing.
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field, ConfigDict


class CustomerInput(BaseModel):
    gender: Literal["Male", "Female"] = Field(..., description="Customer gender")
    SeniorCitizen: Literal[0, 1] = Field(..., description="Whether customer is 65 or older (1: Yes, 0: No)")
    Partner: Literal["Yes", "No"] = Field(..., description="Whether customer has a partner")
    Dependents: Literal["Yes", "No"] = Field(..., description="Whether customer has dependents")
    tenure: int = Field(..., ge=0, le=120, description="Months customer has stayed with company")
    PhoneService: Literal["Yes", "No"] = Field(..., description="Whether customer has phone service")
    MultipleLines: Literal["No phone service", "No", "Yes"] = Field(..., description="Multiple phone lines status")
    InternetService: Literal["DSL", "Fiber optic", "No"] = Field(..., description="Internet service provider type")
    OnlineSecurity: Literal["No internet service", "No", "Yes"] = Field(..., description="Online security add-on")
    OnlineBackup: Literal["No internet service", "No", "Yes"] = Field(..., description="Online backup add-on")
    DeviceProtection: Literal["No internet service", "No", "Yes"] = Field(..., description="Device protection add-on")
    TechSupport: Literal["No internet service", "No", "Yes"] = Field(..., description="Premium tech support add-on")
    StreamingTV: Literal["No internet service", "No", "Yes"] = Field(..., description="Streaming TV add-on")
    StreamingMovies: Literal["No internet service", "No", "Yes"] = Field(..., description="Streaming movies add-on")
    Contract: Literal["Month-to-month", "One year", "Two year"] = Field(..., description="Contract term")
    PaperlessBilling: Literal["Yes", "No"] = Field(..., description="Whether paperless billing is enabled")
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ] = Field(..., description="Payment method used")
    MonthlyCharges: float = Field(..., ge=0.0, description="Monthly recurring charge in USD")
    TotalCharges: Optional[float] = Field(None, ge=0.0, description="Total charges billed (optional)")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
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
                "TotalCharges": 140.70
            }
        }
    )


class PredictionResponse(BaseModel):
    churn_prediction: int = Field(..., description="Binary churn prediction (1: Churn, 0: Retained)")
    churn_label: str = Field(..., description="Prediction ('Yes' or 'No')")
    churn_probability: float = Field(..., description="Model probability score between 0.0 and 1.0")
    risk_level: Literal["Low", "Medium", "High"] = Field(..., description="Risk tier classification")
    key_factors: List[str] = Field(..., description="Top drivers identified for this profile")
    recommended_action: str = Field(..., description="Suggested business retention intervention")


class BatchCustomerInput(BaseModel):
    customers: List[CustomerInput] = Field(..., description="List of customer profiles to score")


class BatchPredictionResponse(BaseModel):
    total_customers: int
    high_risk_count: int
    predictions: List[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    champion_model: str
    version: str
