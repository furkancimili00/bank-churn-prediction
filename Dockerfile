FROM python:3.9-slim

WORKDIR /app

# The frontend doesn't need heavy ML libraries like XGBoost or SHAP.
# We create a lighter requirements file just for Streamlit and requests.
RUN echo "streamlit\nplotly\npandas\nrequests" > frontend_requirements.txt
RUN pip install --no-cache-dir -r frontend_requirements.txt

COPY dashboard.py .
# Optional: Add .streamlit/secrets.toml if needed for production
# COPY .streamlit/ .streamlit/

EXPOSE 8501

CMD ["streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
