import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import joblib
import os

def train_model(data_dir: str = "data", output_path: str = "data/xgboost_churn_model.pkl") -> None:
    """
    Trains an XGBoost model.
    Passes scale_pos_weight based on exact class imbalance ratio of raw data.
    """
    print(f"Loading preprocessed training data from {data_dir}...")
    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv"))

    # --- BOOST F1: DYNAMIC CLASS WEIGHTS ---
    try:
        raw_df = pd.read_csv("data/raw_data.csv")
        num_neg = (raw_df['Exited'] == 0).sum()
        num_pos = (raw_df['Exited'] == 1).sum()
    except FileNotFoundError:
        print("Raw data not found for true ratio calculation. Defaulting to 1.0")
        num_neg, num_pos = 1, 1

    if num_pos > 0:
        imbalance_ratio = num_neg / num_pos
    else:
        imbalance_ratio = 1.0

    print(f"Calculated true class imbalance ratio (negative/positive): {imbalance_ratio:.4f}")

    xgb_clf = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42,
        scale_pos_weight=imbalance_ratio
    )

    param_grid = {
        'n_estimators': [200, 300],
        'max_depth': [5, 7, 9],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    print("Running GridSearchCV to optimize hyperparameters...")
    grid_search = GridSearchCV(
        estimator=xgb_clf,
        param_grid=param_grid,
        scoring='f1',
        cv=3,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train.values.ravel())

    best_model = grid_search.best_estimator_
    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best cross-validated F1 score: {grid_search.best_score_:.4f}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(best_model, output_path)
    print(f"Model successfully saved to {output_path}")

if __name__ == "__main__":
    train_model()
