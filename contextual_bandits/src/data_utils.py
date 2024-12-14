import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(file_path: str):
    """
    Load data from a CSV (or other source).
    The data should have columns: user_features..., action, reward
    """
    df = pd.read_csv(file_path)
    return df


def prepare_data(df: pd.DataFrame, user_features_cols: list, test_size: float = 0.2):
    """
    Split data into train/val sets and extract features, actions, and rewards.
    """
    train_df, val_df = train_test_split(
        df, test_size=test_size, random_state=42
    )

    X_train = train_df[user_features_cols]
    A_train = train_df['action']
    R_train = train_df['reward']

    X_val = val_df[user_features_cols]
    A_val = val_df['action']
    R_val = val_df['reward']

    return X_train, A_train, R_train, X_val, A_val, R_val
