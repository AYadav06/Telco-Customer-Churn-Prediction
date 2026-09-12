# ⚡ Telco Customer Churn Prediction: End-to-End Machine Learning System

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688.svg)](https://fastapi.tiangolo.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-red.svg)](https://xgboost.ai/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2.svg)](https://mlflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Demo-FF4B4B.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)

An enterprise-ready, end-to-end Classical Machine Learning solution to predict, explain, and mitigate customer churn using the Telco Customer Churn dataset. Featuring MLflow tracking, SHAP explainability, FastAPI serving, and an interactive Streamlit dashboard.

---

## 📌 Business Problem & Impact

In the telecommunications sector, acquiring a new customer costs **5x to 7x** more than retaining an existing one. Identifying churn signals early allows proactive retention campaigns (e.g., targeted annual discounts, complimentary tech support), preserving high-margin monthly recurring revenue (MRR).

This project delivers:
- **Class Imbalance Handling**: Optimized for a 26.5% minority churn rate.
- **Explainable Predictions**: Uses Game-Theoretic SHAP values to explain individual risk drivers.
- **Production REST Microservice**: Sub-50ms latency scoring via FastAPI.
- **Interactive Decision Cockpit**: Streamlit dashboard for customer success managers.

---

## 🏗️ Architecture & Project Structure

```text
churn-predictor/
├── data/
│   ├── raw/                 # Original CSV (7,043 customer records)
│   └── processed/           # Stratified train/test splits
├── notebooks/
│   └── eda.ipynb            # Focused exploratory data analysis
├── src/
│   ├── data_prep.py         # Cleaning, feature engineering, ColumnTransformer
│   ├── train.py             # Trains LogReg, RF, XGBoost + MLflow tracking
│   ├── evaluate.py          # Metrics, confusion matrix, SHAP explainability
│   └── predict.py           # Core inference engine
├── mlruns/                  # MLflow experiment metadata and artifacts
├── api/
│   ├── main.py              # FastAPI application (/predict, /health)
│   └── schema.py            # Pydantic v2 schemas for request validation
├── model/
│   ├── churn_model.pkl      # Champion pipeline (Preprocessor + XGBoost)
│   └── model_metrics.json   # Exported benchmark scores
├── assets/                  # High-resolution evaluation & SHAP figures
├── requirements.txt         # Pinned python dependencies
├── Dockerfile               # Production container image
├── streamlit_app.py         # Live web demonstration
├── todo.md                  # Step-by-step learning guide
└── README.md                # Project documentation
```

---

## 📊 Model Comparison & Benchmarks

We trained and evaluated three distinct architectures on a **20% stratified holdout test set** (1,409 customers):

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Balanced)** | 74.8% | 51.5% | 79.7% | 0.626 | 0.838 | 0.622 |
| **Random Forest (Balanced)** | 77.9% | 56.4% | 73.0% | 0.636 | 0.835 | 0.628 |
| **XGBoost (Champion)** 🏆 | **80.3%** | **61.2%** | **74.1%** | **0.670** | **0.846** | **0.655** |

### 🔍 Why XGBoost Won
1. **Handling Non-Linear Decision Boundaries**: Tree-based boosting captured interaction effects (e.g., high Monthly Charges combined with Month-to-Month contracts) that linear models missed.
2. **Handling Class Imbalance**: Leveraging `scale_pos_weight = 2.77` adjusted the gradient penalties for minority class errors, maximizing recall without destroying precision.
3. **Regularization Against Overfitting**: Built-in L1/L2 regularization (`reg_alpha`, `reg_lambda`) and tree-depth constraints prevented the model from memorizing noise.

---

## 🧠 Model Explainability with SHAP

Recruiters and stakeholders need to know **why** the model makes a decision. We implemented `shap.TreeExplainer` on the champion XGBoost model:

### Key Drivers of Churn:
1. **Contract Type**: Month-to-month contracts are the #1 predictor pushing risk upward; 2-year contracts push risk downward.
2. **Tenure**: Early-tenure customers (<12 months) are disproportionately vulnerable.
3. **Internet Service**: Fiber Optic subscribers exhibit higher churn due to premium monthly pricing ($70–$100).
4. **Payment Method**: Electronic check payments correlate with higher payment friction.

*(Visualizations generated via `python src/evaluate.py` and stored in `assets/`)*

---

## 🛡️ Production Drift & Monitoring Strategy

Models decay in production due to changing market conditions and customer behaviors. Our 3-pillar monitoring design:

1. **Covariate (Data) Drift**:
   - Automated two-sample **Kolmogorov-Smirnov (KS) tests** for continuous features (`MonthlyCharges`, `tenure`).
   - **Population Stability Index (PSI)** for categorical features (alerting if $\text{PSI} > 0.2$).
2. **Concept & Prior Probability Drift**:
   - Tracking 30-day rolling average churn predictions against realized cancellation rates.
3. **Automated MLOps Pipeline**:
   - Integrate **Evidently AI** or Prometheus exporters to trigger automated data quality alerts and monthly retraining DAGs.

---

## ⚡ Quickstart: How to Run Locally

### 1. Environment Setup
```bash
git clone https://github.com/AYadav06/Telco-Customer-Churn-Prediction.git
cd churn-predictor
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Data Preparation & Training
```bash
# Prepare and split dataset
python src/data_prep.py

# Train models and track via MLflow
python src/train.py

# Generate evaluation curves & SHAP plots
python src/evaluate.py
```

### 3. Launch MLflow UI
```bash
mlflow ui --port 5000
# View experiment at http://localhost:5000
```

### 4. Run FastAPI REST Service
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
# Interactive Swagger docs: http://localhost:8000/docs
```

### 5. Launch Streamlit UI
```bash
streamlit run streamlit_app.py
# Access interactive dashboard: http://localhost:8501
```

### 6. Run via Docker
```bash
docker build -t churn-predictor:latest .
docker run -p 8000:8000 churn-predictor:latest
```
