"""
Preprocessing pipeline: encoding, scaling, train/test split.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


# Columns to drop before modelling
_DROP_COLS = ["customerID"]

# Feature groups
BINARY_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling",
]
ORDINAL_BINARY = ["SeniorCitizen"]   # already 0/1

MULTI_CATEGORICAL_COLS = [
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaymentMethod",
]
NUMERICAL_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

TARGET_COL = "Churn"


def load_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=_DROP_COLS, errors="ignore").copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"])
    return df


def encode_target(series: pd.Series) -> pd.Series:
    le = LabelEncoder()
    return pd.Series(le.fit_transform(series), index=series.index, name=series.name)


def build_preprocessor() -> ColumnTransformer:
    binary_pipe = Pipeline([
        ("ohe", OneHotEncoder(drop="if_binary", sparse_output=False, handle_unknown="ignore")),
    ])
    multi_cat_pipe = Pipeline([
        ("ohe", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")),
    ])
    num_pipe = Pipeline([
        ("scaler", StandardScaler()),
    ])

    return ColumnTransformer(
        transformers=[
            ("binary", binary_pipe, BINARY_COLS),
            ("multi_cat", multi_cat_pipe, MULTI_CATEGORICAL_COLS),
            ("numerical", num_pipe, NUMERICAL_COLS),
        ],
        remainder="passthrough",
    )


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.20,
    val_size: float = 0.10,
    random_state: int = 42,
):
    """Returns (X_train, X_val, X_test, y_train, y_val, y_test)."""
    df = load_and_clean(df)
    X = df.drop(columns=[TARGET_COL])
    y = encode_target(df[TARGET_COL])

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    val_frac = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_frac, stratify=y_temp, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
