import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE


def preprocess_data(
    input_path="data/raw/transactions.csv",
    output_path="data/processed/processed_data.pkl"
):
    os.makedirs("data/processed", exist_ok=True)

    df = pd.read_csv(input_path)

    X = df.drop(columns=["transaction_id", "is_fraud"])
    y = df["is_fraud"]

    numerical_cols = [
        "amount",
        "hour",
        "day_of_week",
        "previous_transactions",
        "avg_transaction_amount",
        "is_international"
    ]

    categorical_cols = [
        "merchant_category",
        "transaction_type",
        "location_risk"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ]
    )

    X_processed = preprocessor.fit_transform(X)

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_processed, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled,
        y_resampled,
        test_size=0.2,
        random_state=42,
        stratify=y_resampled
    )

    joblib.dump(
        {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "preprocessor": preprocessor,
            "columns": X.columns.tolist()
        },
        output_path
    )

    print("Data preprocessing completed successfully.")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")

    return X_train, X_test, y_train, y_test, preprocessor


if __name__ == "__main__":
    preprocess_data()