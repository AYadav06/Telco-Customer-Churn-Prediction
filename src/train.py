"""
Model training and MLflow tracking pipeline for Telco Customer Churn.
Trains Logistic Regression, Random Forest, and XGBoost models,
logs parameters, metrics, and models to MLflow, and serializes the champion pipeline.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
import joblib

import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from data_prep import (
    PROCESSED_TRAIN_PATH,
    PROCESSED_TEST_PATH,
    TARGET,
    build_preprocessor
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR = "model"
CHAMPION_MODEL_PATH = os.path.join(MODEL_DIR, "churn_model.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "model_metrics.json")
FEATURE_NAMES_PATH = os.path.join(MODEL_DIR, "feature_names.json")
EXPERIMENT_NAME = "telco-customer-churn"


def load_data():
    """Load pre-split train and test datasets."""
    if not os.path.exists(PROCESSED_TRAIN_PATH) or not os.path.exists(PROCESSED_TEST_PATH):
        from data_prep import prepare_and_split_data
        logger.info("Processed data not found. Running data preparation...")
        prepare_and_split_data()

    train_df = pd.read_csv(PROCESSED_TRAIN_PATH)
    test_df = pd.read_csv(PROCESSED_TEST_PATH)

    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET].values
    X_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET].values

    return X_train, y_train, X_test, y_test


def evaluate_model(model, X_test, y_test):
    """Compute standard classification metrics."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }
    return metrics, y_pred, y_proba


def train_and_track():
    """Train multiple models, log runs to MLflow, and export champion pipeline."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Configure MLflow to log locally in ./mlruns (allow filesystem store in MLflow 3.x)
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, y_train, X_test, y_test = load_data()

    # Build fresh preprocessor
    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)

    # Save feature names after OneHotEncoding for explainability in Phase 5
    cat_features = preprocessor.named_transformers_["cat"].named_steps["ohe"].get_feature_names_out()
    num_features = preprocessor.transformers[0][2]
    all_feature_names = list(num_features) + list(cat_features)
    with open(FEATURE_NAMES_PATH, "w") as f:
        json.dump(all_feature_names, f, indent=2)
    logger.info("Saved %d feature names to %s", len(all_feature_names), FEATURE_NAMES_PATH)

    # Compute imbalance weight for XGBoost
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    scale_pos_weight = neg_count / max(pos_count, 1)
    logger.info("Class imbalance ratio (scale_pos_weight): %.2f", scale_pos_weight)

    # Define our 3 candidate models
    models_config = {
        "LogisticRegression": {
            "model": LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                C=0.5,
                solver="lbfgs",
                random_state=42
            ),
            "params": {"C": 0.5, "max_iter": 1000, "class_weight": "balanced"}
        },
        "RandomForest": {
            "model": RandomForestClassifier(
                n_estimators=150,
                max_depth=8,
                min_samples_split=5,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            ),
            "params": {"n_estimators": 150, "max_depth": 8, "min_samples_split": 5, "class_weight": "balanced"}
        },
        "XGBoost": {
            "model": XGBClassifier(
                n_estimators=180,
                max_depth=4,
                learning_rate=0.06,
                scale_pos_weight=scale_pos_weight,
                subsample=0.85,
                colsample_bytree=0.85,
                eval_metric="logloss",
                random_state=42
            ),
            "params": {
                "n_estimators": 180,
                "max_depth": 4,
                "learning_rate": 0.06,
                "scale_pos_weight": float(scale_pos_weight),
                "subsample": 0.85,
                "colsample_bytree": 0.85
            }
        }
    }

    results = {}
    best_model_name = None
    best_roc_auc = -1.0
    best_pipeline = None

    for name, config in models_config.items():
        logger.info("--- Training: %s ---", name)
        clf = config["model"]

        # Bundle preprocessor + model into ONE Pipeline
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", clf)
        ])

        with mlflow.start_run(run_name=name):
            pipeline.fit(X_train, y_train)

            metrics, y_pred, y_proba = evaluate_model(pipeline, X_test, y_test)
            logger.info("%s Test Metrics: Accuracy=%.4f, Precision=%.4f, Recall=%.4f, F1=%.4f, ROC-AUC=%.4f",
                        name, metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1_score"], metrics["roc_auc"])

            # 1. Log Hyperparameters
            mlflow.log_params(config["params"])
            mlflow.log_param("model_type", name)

            # 2. Log Test Metrics
            for m_key, m_val in metrics.items():
                mlflow.log_metric(f"test_{m_key}", m_val)

            # 3. Log Model Artifact
            mlflow.sklearn.log_model(
                pipeline,
                name=f"model_{name.lower()}",
                serialization_format="cloudpickle"
            )

            results[name] = metrics

            # Champion selection based on ROC-AUC
            if metrics["roc_auc"] > best_roc_auc:
                best_roc_auc = metrics["roc_auc"]
                best_model_name = name
                best_pipeline = pipeline

    logger.info("🏆 Champion Model: %s with ROC-AUC = %.4f", best_model_name, best_roc_auc)

    # Save the champion pipeline
    joblib.dump(best_pipeline, CHAMPION_MODEL_PATH)
    logger.info("Saved champion model pipeline to %s", CHAMPION_MODEL_PATH)

    # Export comparison metrics to JSON
    comparison_summary = {
        "champion_model": best_model_name,
        "champion_roc_auc": best_roc_auc,
        "models": results
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(comparison_summary, f, indent=2)
    logger.info("Saved metrics comparison to %s", METRICS_PATH)

    return comparison_summary


if __name__ == "__main__":
    train_and_track()
