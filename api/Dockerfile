FROM python:3.9-slim

WORKDIR /app

# Install system dependencies required for XGBoost/Scikit-Learn/SHAP
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# Copy necessary files
COPY api/ /app/api/
# Copy the trained model and preprocessor artifacts
COPY data/ /app/data/

# Set environment variables for model paths
ENV MODEL_PATH=/app/data/xgboost_churn_model.pkl
ENV PREPROCESSOR_PATH=/app/data/preprocessor.pkl

EXPOSE 8000

# Use Gunicorn for production server
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8000", "api.app:app"]
