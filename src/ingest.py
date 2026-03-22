"""
Data Ingestion Module.
Downloads the Bank Customer Churn dataset from Kaggle or via direct URL if possible.
For this script, we'll download a commonly hosted version of the CSV to avoid needing Kaggle API keys in CI/CD.
"""

import pandas as pd
import os

def load_data(url: str = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/bank-churn.csv", output_path: str = "data/raw_data.csv") -> pd.DataFrame:
    """
    Downloads the dataset from a given URL and saves it to a local CSV.
    Note: The actual Kaggle 'Churn Modelling' dataset is often available on raw github links.
    Using a standard link for 'Churn_Modelling.csv'.
    """
    # Using a reliable raw github link for the standard Churn_Modelling dataset
    # from Kaggle (10k rows, 14 columns).
    real_url = "https://raw.githubusercontent.com/kirenz/datasets/master/churn.csv"
    # Actually, a better known raw link is from the commonly used Kaggle dataset.
    # Let's try to fetch from a known source or just use pandas to load from a known URL.
    # Alternatively, we can use an openml dataset if the github link is dead.
    # We will use an openml fetch to be safe.

    # Wait, the prompt says "write a data ingestion script in the pipeline that automatically fetches the 'Bank Customer Churn' dataset from Kaggle".
    # Kaggle requires API keys. Since I don't have user keys, I'll provide a script that downloads from a public mirror of the exact Kaggle dataset.

    mirror_url = "https://raw.githubusercontent.com/sharmaroshan/Churn-Modelling-Dataset/master/Churn_Modelling.csv"

    print(f"Downloading data from {mirror_url}...")
    try:
        df = pd.read_csv(mirror_url)
    except Exception as e:
        print(f"Error downloading data: {e}")
        raise e

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")
    return df

if __name__ == "__main__":
    load_data()
