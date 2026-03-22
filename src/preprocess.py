import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
import os

def preprocess_data(input_path: str = "data/raw_data.csv", output_dir: str = "data") -> None:
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)

    columns_to_drop = ['RowNumber', 'CustomerId', 'Surname']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

    if 'Gender' in df.columns:
        df = df.drop(columns=['Gender'])
        print("Dropped 'Gender' feature for Fairness through Unawareness.")

    raw_gender = pd.read_csv(input_path)['Gender']

    X = df.drop(columns=['Exited'])
    y = df['Exited']

    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    indices = np.arange(len(X))
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, indices, test_size=0.2, random_state=42, stratify=y
    )

    test_genders = raw_gender.iloc[idx_test]
    test_genders.to_csv(os.path.join(output_dir, "test_genders.csv"), index=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols)
        ]
    )

    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    num_features = numerical_cols
    cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols).tolist()
    feature_names = num_features + cat_features

    X_train_scaled = pd.DataFrame(X_train_transformed, columns=feature_names)
    X_test_scaled = pd.DataFrame(X_test_transformed, columns=feature_names)

    # We remove SMOTE here because scale_pos_weight in XGBoost is superior for maintaining F1 metrics
    # Double-correcting class imbalance (SMOTE + scale_pos_weight) degrades precision heavily.

    os.makedirs(output_dir, exist_ok=True)
    X_train_scaled.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    X_test_scaled.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)

    preprocessor_artifact = {
        'transformer': preprocessor,
        'features': feature_names,
        'categorical_cols': categorical_cols,
        'numerical_cols': numerical_cols
    }
    joblib.dump(preprocessor_artifact, os.path.join(output_dir, "preprocessor.pkl"))

if __name__ == "__main__":
    preprocess_data()
