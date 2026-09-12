"""
Data preparation, cleaning, feature engineering, and preprocessing pipeline
for the Telco Customer Churn dataset.
"""

import os
import logging
from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RAW_DATA_PATH = "data/raw/Telco-Customer-Churn.csv"
PROCESSED_TRAIN_PATH = "data/processed/train.csv"
PROCESSED_TEST_PATH = "data/processed/test.csv"
PREPROCESSOR_PATH = "model/preprocessor.pkl"

TARGET = "Churn"

NUMERICAL_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "num_services",
    "charges_ratio",
    "monthly_tenure_ratio"
]

CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "tenure_group"
]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw dataset:
    - Drop customerID (arbitrary identifier that causes leakage)
    - Fix blank whitespace strings in TotalCharges (impute with 0.0 for tenure=0)
    - Map binary target Churn ('Yes'/'No') to 1/0
    """
    df = df.copy()

    # Drop identifier
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Fix TotalCharges whitespace bug discovered during EDA
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

    # Encode binary target if present
    if TARGET in df.columns and df[TARGET].dtype == "object":
        df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0}).astype(int)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create domain-specific features:
    1. tenure_group: Discretize customer lifecycle stages
    2. num_services: Count of active value-added subscriptions
    3. charges_ratio: MonthlyCharges / (TotalCharges + 1)
    4. monthly_tenure_ratio: TotalCharges / (tenure + 1)
    """
    df = df.copy()

    # 1. Lifecycle Tenure Groups
    bins = [-1, 12, 24, 48, 72]
    labels = ["0-12m", "13-24m", "25-48m", "49-72m"]
    df["tenure_group"] = pd.cut(df["tenure"], bins=bins, labels=labels).astype(str)

    # 2. Count of active add-on services
    services = [
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies"
    ]
    existing_services = [s for s in services if s in df.columns]
    df["num_services"] = df[existing_services].apply(
        lambda row: sum(1 for val in row if str(val).strip().lower() == "yes"), axis=1
    )

    # 3. Ratio Features
    df["charges_ratio"] = df["MonthlyCharges"] / (df["TotalCharges"] + 1.0)
    df["monthly_tenure_ratio"] = df["TotalCharges"] / (df["tenure"] + 1.0)

    # Ensure SeniorCitizen is treated as categorical string for OneHotEncoder
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)

    return df


def build_preprocessor() -> ColumnTransformer:
    """
    Build scikit-learn ColumnTransformer:
    - Numerical: Median imputation + StandardScaler (zero mean, unit variance)
    - Categorical: Mode imputation + OneHotEncoder (handles unseen categories gracefully)
    """
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUMERICAL_FEATURES),
            ("cat", cat_pipeline, CATEGORICAL_FEATURES)
        ],
        remainder="drop"
    )
    return preprocessor


def prepare_and_split_data(
    raw_path: str = RAW_DATA_PATH,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    End-to-end execution: Load -> Clean -> Engineer -> Split -> Fit Preprocessor -> Save.
    """
    logger.info("Loading raw dataset from %s", raw_path)
    raw_df = pd.read_csv(raw_path)

    logger.info("Cleaning data and engineering features...")
    cleaned_df = clean_data(raw_df)
    featured_df = engineer_features(cleaned_df)

    # Stratified split to preserve the 26.5% churn ratio in both train and test
    train_df, test_df = train_test_split(
        featured_df,
        test_size=test_size,
        random_state=random_state,
        stratify=featured_df[TARGET]
    )

    logger.info("Train shape: %s (Churn rate: %.2f%%)", train_df.shape, train_df[TARGET].mean() * 100)
    logger.info("Test shape:  %s (Churn rate: %.2f%%)", test_df.shape, test_df[TARGET].mean() * 100)

    # Save processed datasets
    os.makedirs(os.path.dirname(PROCESSED_TRAIN_PATH), exist_ok=True)
    train_df.to_csv(PROCESSED_TRAIN_PATH, index=False)
    test_df.to_csv(PROCESSED_TEST_PATH, index=False)
    logger.info("Saved train split to %s and test split to %s", PROCESSED_TRAIN_PATH, PROCESSED_TEST_PATH)

    # Fit preprocessor on TRAINING DATA ONLY (Prevents Data Leakage)
    preprocessor = build_preprocessor()
    X_train = train_df.drop(columns=[TARGET])
    preprocessor.fit(X_train)

    os.makedirs(os.path.dirname(PREPROCESSOR_PATH), exist_ok=True)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    logger.info("Fitted preprocessor saved to %s", PREPROCESSOR_PATH)

    return train_df, test_df


if __name__ == "__main__":
    prepare_and_split_data()
