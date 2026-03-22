import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
import joblib
import os

def preprocess_data(input_path: str = "data/raw_data.csv", output_dir: str = "data") -> None:
    """
    Reads raw data, cleans, uses OneHotEncoder, performs train-test split,
    applies SMOTE on the training set (preventing data leakage), and saves processed data.
    """
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)

    # Drop irrelevant columns
    columns_to_drop = ['RowNumber', 'CustomerId', 'Surname']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

    # Define features and target
    X = df.drop(columns=['Exited'])
    y = df['Exited']

    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Train-test split before transforming to prevent data leakage
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Create the column transformer with OneHotEncoder and StandardScaler
    # drop='first' avoids multicollinearity dummy variable trap
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols)
        ]
    )

    # Fit and transform training data, transform test data
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # Get feature names after transformation
    num_features = numerical_cols
    cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols).tolist()
    feature_names = num_features + cat_features

    X_train_scaled = pd.DataFrame(X_train_transformed, columns=feature_names)
    X_test_scaled = pd.DataFrame(X_test_transformed, columns=feature_names)

    print("Applying SMOTE to training data to handle class imbalance...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

    print(f"Original training shape: {X_train_scaled.shape}, Resampled shape: {X_train_resampled.shape}")

    os.makedirs(output_dir, exist_ok=True)

    # Save processed data
    X_train_resampled.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    y_train_resampled.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    X_test_scaled.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)

    # Save the ColumnTransformer (which contains both scaler and encoder) and feature names
    preprocessor_artifact = {
        'transformer': preprocessor,
        'features': feature_names,
        'categorical_cols': categorical_cols,
        'numerical_cols': numerical_cols
    }
    joblib.dump(preprocessor_artifact, os.path.join(output_dir, "preprocessor.pkl"))
    print("Preprocessing completed. Artifacts saved in data/ directory.")

if __name__ == "__main__":
    preprocess_data()
