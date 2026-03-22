from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import shap
import time
import os

app = Flask(__name__)

# Load the model and preprocessor (Global, loaded once on startup)
MODEL_PATH = os.getenv("MODEL_PATH", "data/xgboost_churn_model.pkl")
PREPROCESSOR_PATH = os.getenv("PREPROCESSOR_PATH", "data/preprocessor.pkl")

try:
    print(f"Loading model from {MODEL_PATH}...")
    model = joblib.load(MODEL_PATH)
    print(f"Loading preprocessor from {PREPROCESSOR_PATH}...")
    preprocessor_artifact = joblib.load(PREPROCESSOR_PATH)
    transformer = preprocessor_artifact['transformer']
    expected_features = preprocessor_artifact['features']
    print("Model and preprocessor loaded successfully.")

    # Initialize SHAP explainer
    explainer = shap.TreeExplainer(model)
    print("SHAP explainer initialized.")
except Exception as e:
    print(f"Error loading model/preprocessor: {e}")
    model = None
    transformer = None
    expected_features = None
    explainer = None

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint to verify the API is running."""
    if model is None:
        return jsonify({"status": "error", "message": "Model not loaded."}), 500
    return jsonify({"status": "success", "message": "Flask Churn Prediction API is running."})

def process_data_and_predict(df_input: pd.DataFrame):
    """Helper method to transform data and run inference."""
    # Drop irrelevant columns if they exist
    for col in ['RowNumber', 'CustomerId', 'Surname']:
        if col in df_input.columns:
            df_input = df_input.drop(columns=[col])

    # Apply sklearn ColumnTransformer
    transformed_data = transformer.transform(df_input)
    transformed_df = pd.DataFrame(transformed_data, columns=expected_features)

    # Predict
    churn_probabilities = model.predict_proba(transformed_df)[:, 1]
    churn_predictions = (churn_probabilities >= 0.5).astype(int)

    # SHAP Explainability
    shap_values_raw = explainer.shap_values(transformed_df)

    if isinstance(shap_values_raw, list):
        shap_vals_matrix = shap_values_raw[1]
    else:
        if len(shap_values_raw.shape) == 3:
            shap_vals_matrix = shap_values_raw[:, :, 1]
        else:
            shap_vals_matrix = shap_values_raw

    return churn_predictions, churn_probabilities, shap_vals_matrix

@app.route("/predict", methods=["POST"])
def predict():
    """
    Main single prediction endpoint.
    SLA: < 1 second response time.
    """
    start_time = time.time()

    if model is None:
        return jsonify({"error": "Machine learning model is not available."}), 500

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided."}), 400

        df_input = pd.DataFrame([data])

        preds, probs, shap_matrix = process_data_and_predict(df_input)

        churn_prediction = int(preds[0])
        churn_probability = float(probs[0])
        shap_vals = shap_matrix[0]

        if churn_probability >= 0.70:
            risk_level = "Çok Yüksek Riskli - Acil İletişime Geçilmeli"
        elif churn_probability >= 0.40:
            risk_level = "Orta Riskli - Kampanya Önerilebilir"
        else:
            risk_level = "Düşük Riskli - Sadık Müşteri"

        shap_dict = {feat: float(val) for feat, val in zip(expected_features, shap_vals)}
        sorted_shap = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
        top_shap = {k: v for k, v in sorted_shap[:5]}

        end_time = time.time()
        response_time = end_time - start_time

        if response_time > 1.0:
            app.logger.warning(f"SLA Violation: Prediction took {response_time:.4f} seconds.")

        return jsonify({
            "churn_tahmini": churn_prediction,
            "churn_ihtimali": round(churn_probability, 4),
            "risk_seviyesi": risk_level,
            "shap_degerleri": shap_dict,
            "top_nedenler": top_shap,
            "response_time_sec": round(response_time, 4),
            "mesaj": "Tahmin başarıyla hesaplandı."
        })

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 400

@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    """
    Batch prediction endpoint. Expects a JSON list of customer records.
    Significantly more efficient for large CSV uploads.
    """
    if model is None:
        return jsonify({"error": "Machine learning model is not available."}), 500

    try:
        data = request.get_json()
        if not isinstance(data, list) or len(data) == 0:
            return jsonify({"error": "Input data must be a non-empty JSON array."}), 400

        df_input = pd.DataFrame(data)
        customer_ids = df_input.get("CustomerId", pd.Series(range(len(df_input)))).tolist()
        balances = df_input.get("Balance", pd.Series([0]*len(df_input))).astype(float).tolist()
        salaries = df_input.get("EstimatedSalary", pd.Series([0]*len(df_input))).astype(float).tolist()

        preds, probs, _ = process_data_and_predict(df_input)

        results = []
        for i in range(len(preds)):
            prob = float(probs[i])
            if prob >= 0.70:
                risk_level = "Yüksek Riskli"
            elif prob >= 0.40:
                risk_level = "Orta Riskli"
            else:
                risk_level = "Düşük Riskli"

            c_value = balances[i] + (salaries[i] * 0.20)
            exp_loss = c_value * prob

            results.append({
                "Müşteri ID": customer_ids[i],
                "Risk (%)": round(prob * 100, 2),
                "Risk Seviyesi": risk_level,
                "Müşteri Değeri (€)": round(c_value, 2),
                "Beklenen Kayıp (€)": round(exp_loss, 2)
            })

        return jsonify({"status": "success", "results": results})

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
