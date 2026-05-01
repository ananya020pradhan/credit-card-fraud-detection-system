import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


def evaluate_model(
    processed_path="data/processed/processed_data.pkl",
    model_path="models/fraud_model.pkl"
):
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("images", exist_ok=True)

    data = joblib.load(processed_path)
    bundle = joblib.load(model_path)

    X_test = data["X_test"]
    y_test = data["y_test"]
    model = bundle["model"]

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    report = classification_report(y_test, y_pred)

    with open("outputs/classification_report.txt", "w") as f:
        f.write("Credit Card Fraud Detection Model Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"ROC AUC: {roc_auc:.4f}\n\n")
        f.write(report)

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not Fraud", "Fraud"],
        yticklabels=["Not Fraud", "Fraud"]
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("images/confusion_matrix.png")
    plt.close()

    print("Evaluation completed successfully.")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")

    return accuracy, precision, recall, f1, roc_auc


if __name__ == "__main__":
    evaluate_model()