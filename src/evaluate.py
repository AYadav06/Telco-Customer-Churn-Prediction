"""
Evaluation and explainability module.
Generates ROC curves, Precision-Recall curves, Confusion Matrices,
and SHAP summary/feature importance visualizations.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/script environments
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    confusion_matrix
)

from data_prep import PROCESSED_TEST_PATH, TARGET

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

FIGURES_DIR = "reports/figures"
ASSETS_DIR = "assets"
MODEL_PATH = "model/churn_model.pkl"
FEATURE_NAMES_PATH = "model/feature_names.json"


def ensure_dirs():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(ASSETS_DIR, exist_ok=True)


def plot_confusion_matrix(y_true, y_pred, output_path: str):
    """Plot publication-quality confusion matrix with counts and percentages."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(6, 5))
    annot = [
        [f"{cm[i, j]}\n({cm_norm[i, j]:.1%})" for j in range(cm.shape[1])]
        for i in range(cm.shape[0])
    ]
    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        cbar=False,
        xticklabels=["Retained (0)", "Churned (1)"],
        yticklabels=["Retained (0)", "Churned (1)"],
        ax=ax,
        linewidths=1.5,
        linecolor="white",
        annot_kws={"size": 13, "weight": "bold"}
    )
    ax.set_title("Confusion Matrix - Champion Model", fontsize=14, pad=12, weight="bold")
    ax.set_xlabel("Predicted Label", fontsize=11, labelpad=8)
    ax.set_ylabel("True Label", fontsize=11, labelpad=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info("Saved confusion matrix to %s", output_path)


def plot_roc_pr_curves(pipeline, X_test, y_test, roc_path: str, pr_path: str):
    """Plot ROC and Precision-Recall curves."""
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="#2b5c8f", lw=2.5, label=f"Champion Model (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="#999999", lw=1.5, linestyle="--", label="Random Baseline (AUC = 0.500)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curve - Telco Churn Predictor", fontsize=14, weight="bold", pad=12)
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(roc_path, dpi=300)
    plt.close()
    logger.info("Saved ROC curve to %s", roc_path)

    # 2. PR Curve
    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(rec, prec)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(rec, prec, color="#d9534f", lw=2.5, label=f"Champion Model (PR-AUC = {pr_auc:.3f})")
    baseline = np.mean(y_test)
    ax.plot([0, 1], [baseline, baseline], color="#999999", lw=1.5, linestyle="--", label=f"Baseline ({baseline:.1%})")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Precision-Recall Curve", fontsize=14, weight="bold", pad=12)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(pr_path, dpi=300)
    plt.close()
    logger.info("Saved PR curve to %s", pr_path)


def compute_shap_explanations(pipeline, X_test, summary_path: str, bar_path: str):
    """Compute SHAP values and save summary and bar plots."""
    preprocessor = pipeline.named_steps["preprocessor"]
    clf = pipeline.named_steps["classifier"]

    # Transform test set using fitted preprocessor
    X_test_transformed = preprocessor.transform(X_test)

    # Load feature names
    if os.path.exists(FEATURE_NAMES_PATH):
        with open(FEATURE_NAMES_PATH, "r") as f:
            feature_names = json.load(f)
    else:
        cat_features = preprocessor.named_transformers_["cat"].named_steps["ohe"].get_feature_names_out()
        num_features = preprocessor.transformers[0][2]
        feature_names = list(num_features) + list(cat_features)

    # Clean prefix names for pretty plotting
    clean_feature_names = [
        f.replace("remainder__", "").replace("cat__", "").replace("num__", "")
        for f in feature_names
    ]

    logger.info("Computing SHAP values on test set (%d samples)...", len(X_test))

    # Generic Explainer that automatically detects Tree vs Linear model
    try:
        explainer = shap.Explainer(clf, X_test_transformed)
        shap_values = explainer(X_test_transformed)
        shap_vals_matrix = shap_values.values
    except Exception:
        # Fallback to TreeExplainer
        explainer = shap.TreeExplainer(clf)
        shap_vals_matrix = explainer.shap_values(X_test_transformed)
        if isinstance(shap_vals_matrix, list):
            shap_vals_matrix = shap_vals_matrix[1]

    # 1. SHAP Summary (Beeswarm) Plot
    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_vals_matrix,
        X_test_transformed,
        feature_names=clean_feature_names,
        max_display=15,
        show=False
    )
    plt.title("SHAP Feature Importance (Impact on Churn Risk)", fontsize=14, weight="bold", pad=14)
    plt.tight_layout()
    plt.savefig(summary_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved SHAP summary plot to %s", summary_path)

    # 2. SHAP Bar Plot (Global Importance)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_vals_matrix,
        X_test_transformed,
        feature_names=clean_feature_names,
        plot_type="bar",
        max_display=15,
        show=False
    )
    plt.title("Top 15 Global Predictive Features (Mean |SHAP Value|)", fontsize=14, weight="bold", pad=14)
    plt.tight_layout()
    plt.savefig(bar_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved SHAP bar plot to %s", bar_path)


def run_evaluation():
    ensure_dirs()
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Champion model not found at {MODEL_PATH}. Train models first.")

    pipeline = joblib.load(MODEL_PATH)
    test_df = pd.read_csv(PROCESSED_TEST_PATH)

    X_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET].values

    y_pred = pipeline.predict(X_test)

    # Target filepaths
    cm_path = os.path.join(FIGURES_DIR, "confusion_matrix.png")
    roc_path = os.path.join(FIGURES_DIR, "roc_curve.png")
    pr_path = os.path.join(FIGURES_DIR, "pr_curve.png")
    shap_summary_path = os.path.join(FIGURES_DIR, "shap_summary.png")
    shap_bar_path = os.path.join(FIGURES_DIR, "shap_bar.png")

    plot_confusion_matrix(y_test, y_pred, cm_path)
    plot_roc_pr_curves(pipeline, X_test, y_test, roc_path, pr_path)
    compute_shap_explanations(pipeline, X_test, shap_summary_path, shap_bar_path)

    # Copy to assets/ for Streamlit and README
    import shutil
    for fig_file in ["confusion_matrix.png", "roc_curve.png", "pr_curve.png", "shap_summary.png", "shap_bar.png"]:
        shutil.copyfile(os.path.join(FIGURES_DIR, fig_file), os.path.join(ASSETS_DIR, fig_file))

    logger.info("Evaluation complete! Visualizations exported to %s and %s", FIGURES_DIR, ASSETS_DIR)


if __name__ == "__main__":
    run_evaluation()
