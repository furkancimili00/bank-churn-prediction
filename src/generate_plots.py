import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
import os

def generate_shap_plots(model_path: str = "data/xgboost_churn_model.pkl",
                        data_dir: str = "data",
                        output_dir: str = "outputs"):
    """
    Loads the trained XGBoost model and preprocessed X_test data.
    Generates and saves high-resolution SHAP Summary and Dependence plots.
    """
    print(f"Loading model from {model_path}...")
    try:
        model = joblib.load(model_path)
    except FileNotFoundError as e:
        print(f"Model not found. Please run the training pipeline first. Error: {e}")
        return

    print(f"Loading test data from {data_dir}/X_test.csv...")
    try:
        X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    except FileNotFoundError as e:
        print(f"Test data not found. Please run the preprocessing pipeline first. Error: {e}")
        return

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    print("Initializing SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(model)

    print("Calculating SHAP values for the test set...")
    shap_values_raw = explainer.shap_values(X_test)

    # Handle SHAP output format depending on XGBoost objective
    if isinstance(shap_values_raw, list):
        shap_values = shap_values_raw[1] # For binary classification returning list
    else:
        if len(shap_values_raw.shape) == 3:
            shap_values = shap_values_raw[:, :, 1]
        else:
            shap_values = shap_values_raw

    # 1. Generate SHAP Summary Plot
    print("Generating SHAP Summary Plot...")
    plt.figure(figsize=(10, 8))
    # We use show=False so we can save the figure instead of rendering it in the GUI
    shap.summary_plot(shap_values, X_test, show=False)
    summary_plot_path = os.path.join(output_dir, "shap_summary_plot.png")
    plt.savefig(summary_plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved SHAP Summary Plot to {summary_plot_path}")
    plt.close()

    # 2. Generate SHAP Dependence Plot for 'Age'
    # Need to verify if 'Age' exists in X_test columns (might be num__Age depending on transformer)
    age_col = None
    for col in X_test.columns:
        if 'Age' in col:
            age_col = col
            break

    if age_col:
        print(f"Generating SHAP Dependence Plot for feature: {age_col}...")
        plt.figure(figsize=(10, 8))
        shap.dependence_plot(age_col, shap_values, X_test, show=False)
        dependence_plot_path = os.path.join(output_dir, f"shap_dependence_plot_{age_col}.png")
        plt.savefig(dependence_plot_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved SHAP Dependence Plot to {dependence_plot_path}")
        plt.close()
    else:
        print("❌ 'Age' feature not found in X_test columns. Skipping Dependence Plot.")

    print("SHAP Plot Generation Complete.")

if __name__ == "__main__":
    generate_shap_plots()
