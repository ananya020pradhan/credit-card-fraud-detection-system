import os
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def train_model(
    processed_path="data/processed/processed_data.pkl",
    model_path="models/fraud_model.pkl"
):
    os.makedirs("models", exist_ok=True)

    data = joblib.load(processed_path)

    X_train = data["X_train"]
    y_train = data["y_train"]
    preprocessor = data["preprocessor"]
    columns = data["columns"]

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    joblib.dump(
        {
            "model": model,
            "preprocessor": preprocessor,
            "columns": columns
        },
        model_path
    )

    print("Model trained successfully.")
    print(f"Model saved at: {model_path}")

    return model


if __name__ == "__main__":
    train_model()