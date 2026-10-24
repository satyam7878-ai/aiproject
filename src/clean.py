import os
import pandas as pd


def load_and_clean_data():
    # Always use a relative path so the TA's fresh-clone test passes
    data_path = os.path.join("data", "winequality-red.csv")

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found at '{data_path}'. Please ensure winequality-red.csv is inside the data/ folder."
        )

    # Read CSV (automatically handles comma or semicolon separators)
    df = pd.read_csv(data_path)
    if df.shape[1] == 1:
        df = pd.read_csv(data_path, sep=";")

    print(f"Loaded {data_path}: {df.shape[0]} rows, {df.shape[1]} columns")

    # Drop any missing values
    df_clean = df.dropna().reset_index(drop=True)

    # Dependent variable (target) is 'alcohol'; all other columns are features
    target_col = "alcohol"
    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]

    return df_clean, X, y
