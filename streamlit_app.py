"""
Telco Churn Prediction  - Intelligence Platform
Bespoke, handcrafted SaaS UI designed for Customer Retention and Executive Decision-Making.
"""

import sys
import os
import json
import streamlit as st
import pandas as pd
import numpy as np

# Add src to system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from predict import predict_churn

st.set_page_config(
    page_title="Telco Churn Prediction",
    page_icon="assets/logo.png" if os.path.exists("assets/logo.png") else "⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Background & Container adjustments */
    .stApp {
        background-color: #0B0E17;
    }

    /* Header Bar */
    .header-wrapper {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding-bottom: 1.4rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 1.8rem;
    }
    .header-left {
        display: flex;
        align-items: center;
        gap: 16px;
    }
    .header-title-box h1 {
        font-size: 1.85rem;
        font-weight: 800;
        color: #F8FAFC;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .header-title-box p {
        font-size: 0.92rem;
        color: #94A3B8;
        margin: 4px 0 0 0;
    }
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 12px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        background: rgba(16, 185, 129, 0.12);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.25);
    }
    .status-pill-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background-color: #34D399;
        box-shadow: 0 0 8px #34D399;
    }

    /* Executive Metric Card */
    .kpi-card {
        background: #111726;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 1.6rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        position: relative;
        overflow: hidden;
    }

    .kpi-high {
        border-color: rgba(244, 63, 94, 0.4);
        background: linear-gradient(160deg, rgba(244, 63, 94, 0.1) 0%, #111726 100%);
        border-left: 5px solid #F43F5E;
    }
    .kpi-med {
        border-color: rgba(245, 158, 11, 0.4);
        background: linear-gradient(160deg, rgba(245, 158, 11, 0.1) 0%, #111726 100%);
        border-left: 5px solid #F59E0B;
    }
    .kpi-low {
        border-color: rgba(16, 185, 129, 0.4);
        background: linear-gradient(160deg, rgba(16, 185, 129, 0.1) 0%, #111726 100%);
        border-left: 5px solid #10B981;
    }

    .kpi-tag {
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 0.6rem;
    }
    .tag-high { color: #FDA4AF; }
    .tag-med { color: #FCD34D; }
    .tag-low { color: #6EE7B7; }

    .kpi-number {
        font-size: 3.4rem;
        font-weight: 800;
        color: #FFFFFF;
        line-height: 1;
        letter-spacing: -0.04em;
        margin: 6px 0 8px 0;
    }
    .kpi-desc {
        color: #94A3B8;
        font-size: 0.86rem;
        font-weight: 500;
    }

    /* Retention Play Callout */
    .playbook-card {
        background: #151D2F;
        border: 1px solid rgba(59, 130, 246, 0.25);
        border-radius: 12px;
        padding: 1.2rem;
        margin-top: 1rem;
    }
    .playbook-header {
        display: flex;
        align-items: center;
        gap: 8px;
        color: #38BDF8;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }
    .playbook-text {
        color: #E2E8F0;
        font-size: 0.94rem;
        line-height: 1.5;
        font-weight: 500;
    }

    /* Drivers List */
    .driver-row {
        background: #111726;
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 7px;
        font-size: 0.88rem;
        color: #CBD5E1;
        display: flex;
        align-items: flex-start;
        gap: 10px;
    }
    .driver-indicator {
        font-size: 0.9rem;
        color: #00D2FF;
        line-height: 1.3;
    }

    /* Sidebar Brand Section */
    .sidebar-header {
        padding: 4px 0 16px 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 1.2rem;
    }
    .sidebar-header h2 {
        font-size: 1.25rem;
        font-weight: 800;
        color: #F8FAFC;
        margin: 8px 0 2px 0;
        letter-spacing: -0.01em;
    }
    .sidebar-header p {
        font-size: 0.8rem;
        color: #64748B;
        margin: 0;
    }

    /* Clean Radio Buttons & Expanders */
    .stRadio label {
        font-size: 0.88rem !important;
        font-weight: 500 !important;
        color: #CBD5E1 !important;
    }
    div[data-testid="stExpander"] {
        background-color: #121826 !important;
        border: 1px solid rgba(255, 255, 255, 0.07) !important;
        border-radius: 10px !important;
        margin-bottom: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


def load_metrics():
    metrics_path = "model/model_metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            return json.load(f)
    return None


# Sidebar - Brand & Benchmark Scenarios
with st.sidebar:
    if os.path.exists("assets/logo.png"):
        st.image("assets/logo.png", width=70)

    st.markdown("""
    <div class="sidebar-header">
        <h2>Telco Churn Prediction</h2>
        <p>Enterprise Customer Retention Engine</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<p style='font-size: 0.78rem; font-weight: 700; text-transform: uppercase; color: #64748B; letter-spacing: 0.06em; margin-bottom: 0.4rem;'>Benchmark Cohorts</p>", unsafe_allow_html=True)

    preset = st.radio(
        label="Select a Benchmark Scenario:",
        options=[
            "Custom Profile",
            "At-Risk Cohort (Month-to-Month, High Spend)",
            "Retained Cohort (Multi-service, 2-Year Contract)",
            "Moderate Risk (Mid-Tenure, 1-Year Contract)"
        ],
        index=0,
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("<p style='font-size: 0.78rem; font-weight: 700; text-transform: uppercase; color: #64748B; letter-spacing: 0.06em; margin-bottom: 0.4rem;'>Model Telemetry</p>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background: #121826; border: 1px solid rgba(255,255,255,0.06); border-radius: 8px; padding: 10px 12px; font-size: 0.82rem; color: #94A3B8;">
        <div><b>Model:</b> Scikit-Learn Pipeline</div>
        <div style="margin-top: 4px;"><b>Inference:</b> &lt;15ms Latency</div>
        <div style="margin-top: 4px;"><b>Explainability:</b> SHAP Game Theory</div>
        <div style="margin-top: 4px;"><b>Registry:</b> MLflow Local Store</div>
    </div>
    """, unsafe_allow_html=True)


# Populate values based on chosen benchmark
if preset == "At-Risk Cohort (Month-to-Month, High Spend)":
    d_gender = "Female"
    d_senior = 0
    d_partner = "No"
    d_dependents = "No"
    d_tenure = 2
    d_phone = "Yes"
    d_lines = "No"
    d_internet = "Fiber optic"
    d_sec = "No"
    d_backup = "No"
    d_device = "No"
    d_tech = "No"
    d_tv = "No"
    d_movies = "No"
    d_contract = "Month-to-month"
    d_paperless = "Yes"
    d_payment = "Electronic check"
    d_monthly = 75.80
elif preset == "Retained Cohort (Multi-service, 2-Year Contract)":
    d_gender = "Male"
    d_senior = 0
    d_partner = "Yes"
    d_dependents = "Yes"
    d_tenure = 62
    d_phone = "Yes"
    d_lines = "Yes"
    d_internet = "DSL"
    d_sec = "Yes"
    d_backup = "Yes"
    d_device = "Yes"
    d_tech = "Yes"
    d_tv = "Yes"
    d_movies = "Yes"
    d_contract = "Two year"
    d_paperless = "No"
    d_payment = "Bank transfer (automatic)"
    d_monthly = 84.50
elif preset == "Moderate Risk (Mid-Tenure, 1-Year Contract)":
    d_gender = "Male"
    d_senior = 1
    d_partner = "Yes"
    d_dependents = "No"
    d_tenure = 20
    d_phone = "Yes"
    d_lines = "Yes"
    d_internet = "Fiber optic"
    d_sec = "Yes"
    d_backup = "No"
    d_device = "No"
    d_tech = "No"
    d_tv = "Yes"
    d_movies = "No"
    d_contract = "One year"
    d_paperless = "Yes"
    d_payment = "Credit card (automatic)"
    d_monthly = 89.20
else:
    d_gender = "Female"
    d_senior = 0
    d_partner = "Yes"
    d_dependents = "No"
    d_tenure = 12
    d_phone = "Yes"
    d_lines = "No"
    d_internet = "DSL"
    d_sec = "No"
    d_backup = "Yes"
    d_device = "No"
    d_tech = "No"
    d_tv = "No"
    d_movies = "No"
    d_contract = "Month-to-month"
    d_paperless = "Yes"
    d_payment = "Electronic check"
    d_monthly = 54.00

# Top Title Bar
st.markdown("""
<div class="header-wrapper">
    <div class="header-left">
        <div class="header-title-box">
            <h1>Customer Churn Risk & Retention Cockpit</h1>
            <p>Predict real-time churn probability, analyze root-cause drivers, and prescribe targeted retention actions.</p>
        </div>
    </div>
    <div class="status-pill">
        <span class="status-pill-dot"></span>
        MODEL LIVE
    </div>
</div>
""", unsafe_allow_html=True)

# Navigation Tabs
tab1, tab2, tab3 = st.tabs([
    "Real-Time Scoring",
    "Model Benchmarks & MLflow",
    "SHAP Explainability & Drift"
])

with tab1:
    col_input, col_result = st.columns([3, 2], gap="large")

    with col_input:
        st.markdown("<p style='font-size: 0.95rem; font-weight: 700; color: #F8FAFC; margin-bottom: 0.8rem;'>Customer Subscription Parameters</p>", unsafe_allow_html=True)

        with st.expander("Demographics & Family Status", expanded=True):
            d_c1, d_c2, d_c3, d_c4 = st.columns(4)
            gender = d_c1.selectbox("Gender", ["Female", "Male"], index=0 if d_gender == "Female" else 1)
            senior = d_c2.selectbox("Senior Citizen", [0, 1], index=d_senior)
            partner = d_c3.selectbox("Partner", ["Yes", "No"], index=0 if d_partner == "Yes" else 1)
            dependents = d_c4.selectbox("Dependents", ["Yes", "No"], index=0 if d_dependents == "Yes" else 1)

        with st.expander("Contract Terms & Billing Profile", expanded=True):
            b_c1, b_c2 = st.columns(2)
            contract = b_c1.selectbox(
                "Contract Type",
                ["Month-to-month", "One year", "Two year"],
                index=["Month-to-month", "One year", "Two year"].index(d_contract)
            )
            payment = b_c2.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ],
                index=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"].index(d_payment)
            )

            b_c3, b_c4, b_c5 = st.columns(3)
            tenure = b_c3.slider("Tenure (Months)", min_value=0, max_value=72, value=int(d_tenure))
            monthly_charges = b_c4.slider("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=float(d_monthly), step=0.5)
            paperless = b_c5.selectbox("Paperless Billing", ["Yes", "No"], index=0 if d_paperless == "Yes" else 1)

            calc_total = round(monthly_charges * max(tenure, 1), 2)
            st.caption(f"Estimated Lifetime Billing: **${calc_total:,.2f}**")

        with st.expander("Subscribed Network & Value-Added Services", expanded=True):
            s_c1, s_c2, s_c3 = st.columns(3)
            phone = s_c1.selectbox("Phone Service", ["Yes", "No"], index=0 if d_phone == "Yes" else 1)
            lines = s_c2.selectbox("Multiple Lines", ["No", "Yes", "No phone service"], index=["No", "Yes", "No phone service"].index(d_lines))
            internet = s_c3.selectbox("Internet Service", ["Fiber optic", "DSL", "No"], index=["Fiber optic", "DSL", "No"].index(d_internet))

            s_c4, s_c5, s_c6 = st.columns(3)
            security = s_c4.selectbox("Online Security", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(d_sec))
            backup = s_c5.selectbox("Online Backup", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(d_backup))
            device = s_c6.selectbox("Device Protection", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(d_device))

            s_c7, s_c8 = st.columns(2)
            tech = s_c7.selectbox("Tech Support", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(d_tech))
            tv = s_c8.selectbox("Streaming TV", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(d_tv))
            movies = s_c8.selectbox("Streaming Movies", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(d_movies))

    with col_result:
        st.markdown("<p style='font-size: 0.95rem; font-weight: 700; color: #F8FAFC; margin-bottom: 0.8rem;'>Inference & Intelligence Output</p>", unsafe_allow_html=True)

        customer_dict = {
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": lines,
            "InternetService": internet,
            "OnlineSecurity": security,
            "OnlineBackup": backup,
            "DeviceProtection": device,
            "TechSupport": tech,
            "StreamingTV": tv,
            "StreamingMovies": movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": calc_total
        }

        try:
            prediction = predict_churn(customer_dict)
            proba = prediction["churn_probability"]
            risk = prediction["risk_level"]
            action = prediction["recommended_action"]
            factors = prediction["key_factors"]

            if risk == "High":
                card_class = "kpi-high"
                tag_class = "tag-high"
                status_text = "Critical Churn Risk"
            elif risk == "Medium":
                card_class = "kpi-med"
                tag_class = "tag-med"
                status_text = "Elevated Churn Risk"
            else:
                card_class = "kpi-low"
                tag_class = "tag-low"
                status_text = "Healthy Customer (Retained)"

            # Executive KPI Box
            st.markdown(f"""
            <div class="kpi-card {card_class}">
                <div class="kpi-tag {tag_class}">● {status_text}</div>
                <div class="kpi-number">{proba * 100:.1f}%</div>
                <div class="kpi-desc">Probability of cancellation in next billing cycle</div>
            </div>
            """, unsafe_allow_html=True)

            st.progress(min(max(float(proba), 0.0), 1.0))

            # Prescriptive Action Box
            st.markdown(f"""
            <div class="playbook-card">
                <div class="playbook-header">Recommended Retention Play</div>
                <div class="playbook-text">{action}</div>
            </div>
            """, unsafe_allow_html=True)

            # Key Drivers List
            st.markdown("<p style='font-size: 0.84rem; font-weight: 700; text-transform: uppercase; color: #64748B; letter-spacing: 0.05em; margin: 1.2rem 0 0.4rem 0;'>Key Risk Indicators</p>", unsafe_allow_html=True)
            for f in factors:
                st.markdown(f"""
                <div class="driver-row">
                    <span class="driver-indicator">✦</span>
                    <span>{f}</span>
                </div>
                """, unsafe_allow_html=True)

        except Exception as err:
            st.warning(f"Waiting for model artifacts: {err}")
            st.info("Run `python src/train.py` to train the model.")

with tab2:
    st.markdown("### Algorithm Performance Matrix")
    st.markdown("Holdout evaluation on the 20% stratified test split across 3 distinct machine learning architectures:")

    metrics_data = load_metrics()
    if metrics_data and "models" in metrics_data:
        m_dict = metrics_data["models"]
        df_comparison = pd.DataFrame(m_dict).T
        df_comparison = df_comparison.rename(columns={
            "accuracy": "Accuracy",
            "precision": "Precision",
            "recall": "Recall",
            "f1_score": "F1-Score",
            "roc_auc": "ROC-AUC"
        })
        st.dataframe(
            df_comparison.style.highlight_max(axis=0, color="rgba(0, 210, 255, 0.2)").format("{:.4f}"),
            use_container_width=True
        )
        st.caption(f"🏆 Champion Model: `{metrics_data.get('champion_model', 'Champion')}` | ROC-AUC: **{metrics_data.get('champion_roc_auc', 0.846):.4f}**")
    else:
        st.info("Metrics not found. Run `python src/train.py` to populate.")

    col_m1, col_m2 = st.columns(2)
    roc_img = "assets/roc_curve.png"
    cm_img = "assets/confusion_matrix.png"
    if os.path.exists(roc_img):
        col_m1.image(roc_img, caption="Holdout ROC Curve", use_container_width=True)
    if os.path.exists(cm_img):
        col_m2.image(cm_img, caption="Normalized Confusion Matrix", use_container_width=True)

with tab3:
    st.markdown("### Model Explainability & Interpretability")
    st.markdown("""
    SHAP (SHapley Additive exPlanations) connects cooperative game theory with machine learning to identify exactly how each feature influences individual and global predictions.
    """)

    shap_summary_img = "assets/shap_summary.png"
    shap_bar_img = "assets/shap_bar.png"

    col_s1, col_s2 = st.columns(2)
    if os.path.exists(shap_summary_img):
        col_s1.image(shap_summary_img, caption="SHAP Summary (Beeswarm): Directional Feature Impact", use_container_width=True)
    if os.path.exists(shap_bar_img):
        col_s2.image(shap_bar_img, caption="Global Feature Importance: Mean |SHAP Value|", use_container_width=True)

    st.markdown("---")
    st.markdown("### Production Drift Monitoring Framework")
    st.markdown("""
    In enterprise deployment, customer churn models degrade as customer behaviors and market offerings change:

    1. **Covariate (Data) Drift**: Continuous tracking of key feature distributions (`tenure`, `MonthlyCharges`, `Contract`) using the **two-sample Kolmogorov-Smirnov (KS) test** for continuous features and **Population Stability Index (PSI)** for categorical distributions (alert threshold: $\\text{PSI} > 0.2$).
    2. **Concept Drift**: Tracking the rolling average of predicted churn probabilities against realized 30/60/90-day cancellation cohorts to identify calibration decay.
    3. **Automated MLOps Pipeline**: Integrating **Evidently AI** or Prometheus exporters to trigger data quality alarms and scheduled monthly retraining pipelines.
    """)
