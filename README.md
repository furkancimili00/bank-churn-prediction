# 🏦 Modernized Bank Customer Churn Prediction: An Explainable AI (XAI) Approach

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-Model-1B9CFC.svg)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-9b59b6.svg)
![Flask](https://img.shields.io/badge/Flask-REST_API-000000.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B.svg)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-20bf6b.svg)

## 📌 Abstract
This project presents a production-ready, highly accurate, and explainable Machine Learning pipeline designed to predict bank customer churn. Addressing the inherent challenge of imbalanced datasets and algorithmic fairness, the system employs advanced resampling techniques (SMOTE/ADASYN) coupled with an optimized **XGBoost** classifier.

The architecture strictly decouples the predictive engine from the user interface. A highly performant **Flask REST API** serves the ML model (achieving sub-second SLA response times), while a modernized **Streamlit** frontend acts purely as a client dashboard for branch managers and portfolio executives. Crucially, the system integrates **SHAP (SHapley Additive exPlanations)** to provide transparent, feature-level insights into every prediction, ensuring the model's decisions are interpretable and compliant with ethical AI guidelines (targeting a Disparate Impact Ratio < 1.2). The entire lifecycle is governed by automated **CI/CD pipelines via GitHub Actions**, strictly enforcing performance (F1 > 0.85, ROC-AUC > 0.90) and ethical KPIs before deployment.

---

## 🏗️ System Architecture
![Architecture](https://via.placeholder.com/800x400?text=Architecture+Diagram+Placeholder)

The system is designed as a microservice architecture:
1. **Data Pipeline (`src/`)**: Modular scripts for data ingestion, robust preprocessing (`ColumnTransformer`, `OneHotEncoder`, `StandardScaler`), resampling, hyperparameter tuning, and KPI evaluation.
2. **Inference Engine (`api/`)**: A Flask-based RESTful API exposing `/predict` (single instance) and `/predict_batch` (bulk CSV) endpoints, calculating SHAP values on the fly.
3. **Client Application (`dashboard.py`)**: A Streamlit dashboard providing single-customer risk analysis, "What-If" scenario simulations, and financial impact (CLTV) batch assessments.

---

## 🚀 Local Execution Guide (Non-Docker)
Due to stringent enterprise IT policies that may block Docker, the system is fully capable of running via a local Python virtual environment.

### 1. Environment Setup
Open a terminal in the project root and create a virtual environment:

**Windows**
```powershell
python -m virtualenv venv
.\venv\Scripts\activate
```

**Linux/Mac**
```bash
python3 -m virtualenv venv
source venv/bin/activate
```

Install the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Configure Credentials (First Time Only)
Create a `.streamlit` folder and a `secrets.toml` file to configure secure access to the dashboard:
```bash
mkdir .streamlit
echo '[auth]' > .streamlit/secrets.toml
echo 'username = "admin"' >> .streamlit/secrets.toml
echo 'password = "supersecretpassword123"' >> .streamlit/secrets.toml
```

### 3. Start the Flask REST API (Terminal 1)
In your active virtual environment, start the backend server:
```bash
python api/app.py
```
*The API will start running on `http://localhost:8000`.*

### 4. Start the Streamlit Dashboard (Terminal 2)
Open a **new terminal window**, navigate to the project root, activate your virtual environment again, set the API environment variable, and run Streamlit:

**Windows (PowerShell):**
```powershell
.\venv\Scripts\activate
$env:API_URL="http://localhost:8000"
streamlit run dashboard.py
```

**Linux/Mac (Bash):**
```bash
source venv/bin/activate
export API_URL="http://localhost:8000"
streamlit run dashboard.py
```
*The dashboard will automatically open in your browser at `http://localhost:8501`.*

---

## 🧠 Model Training Instructions
If you need to retrain the XGBoost model from scratch or update the dataset, execute the modular pipeline sequentially from the project root. Ensure your virtual environment is active.

1. **Ingest Data:** Fetches the Kaggle Bank Customer Churn dataset.
   ```bash
   python src/ingest.py
   ```
2. **Preprocess Data:** Cleans data, applies One-Hot Encoding, scaling, and SMOTE resampling (strictly post-split to prevent data leakage). Saves artifacts to `data/`.
   ```bash
   python src/preprocess.py
   ```
3. **Train Model:** Runs hyperparameter optimization via GridSearchCV to train the XGBoost classifier. Saves the `.pkl` model.
   ```bash
   python src/train.py
   ```
4. **Evaluate KPIs:** Mathematically verifies the F1-score, ROC-AUC, and calculates the Disparate Impact Ratio using SHAP to ensure fairness compliance.
   ```bash
   python src/evaluate.py
   ```

---
*Developed for Modernized Bank Customer Churn Prediction Thesis.*
