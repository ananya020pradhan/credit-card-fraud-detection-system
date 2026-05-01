import os
import numpy as np
import pandas as pd


def generate_transaction_data(n_samples=10000, save_path="data/raw/transactions.csv"):
    np.random.seed(42)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    transaction_id = [f"TXN{i:06d}" for i in range(1, n_samples + 1)]

    amount = np.random.exponential(scale=120, size=n_samples).round(2)
    hour = np.random.randint(0, 24, size=n_samples)
    day_of_week = np.random.randint(0, 7, size=n_samples)

    merchant_category = np.random.choice(
        ["Grocery", "Electronics", "Fuel", "Restaurant", "Travel", "Luxury", "Online"],
        size=n_samples,
        p=[0.25, 0.15, 0.15, 0.18, 0.08, 0.04, 0.15]
    )

    transaction_type = np.random.choice(
        ["POS", "Online", "ATM", "Contactless"],
        size=n_samples,
        p=[0.4, 0.35, 0.1, 0.15]
    )

    location_risk = np.random.choice(
        ["Low", "Medium", "High"],
        size=n_samples,
        p=[0.7, 0.22, 0.08]
    )

    previous_transactions = np.random.poisson(lam=5, size=n_samples)
    avg_transaction_amount = np.random.normal(loc=100, scale=40, size=n_samples).round(2)
    avg_transaction_amount = np.maximum(avg_transaction_amount, 10)

    is_international = np.random.choice([0, 1], size=n_samples, p=[0.88, 0.12])

    fraud_probability = (
        (amount > 500).astype(int) * 0.20 +
        ((hour >= 0) & (hour <= 5)).astype(int) * 0.15 +
        (merchant_category == "Luxury").astype(int) * 0.15 +
        (transaction_type == "Online").astype(int) * 0.10 +
        (location_risk == "High").astype(int) * 0.20 +
        (is_international == 1).astype(int) * 0.15 +
        (previous_transactions <= 1).astype(int) * 0.05
    )

    fraud_probability = np.clip(fraud_probability, 0, 0.85)
    is_fraud = np.random.binomial(1, fraud_probability)

    fraud_indices = np.where(is_fraud == 1)[0]
    if len(fraud_indices) > int(n_samples * 0.06):
        remove_fraud = np.random.choice(
            fraud_indices,
            size=len(fraud_indices) - int(n_samples * 0.06),
            replace=False
        )
        is_fraud[remove_fraud] = 0

    df = pd.DataFrame({
        "transaction_id": transaction_id,
        "amount": amount,
        "hour": hour,
        "day_of_week": day_of_week,
        "merchant_category": merchant_category,
        "transaction_type": transaction_type,
        "location_risk": location_risk,
        "previous_transactions": previous_transactions,
        "avg_transaction_amount": avg_transaction_amount,
        "is_international": is_international,
        "is_fraud": is_fraud
    })

    df.to_csv(save_path, index=False)
    print(f"Dataset created successfully: {save_path}")
    print(f"Shape: {df.shape}")
    print(f"Fraud Rate: {df['is_fraud'].mean() * 100:.2f}%")

    return df


if __name__ == "__main__":
    generate_transaction_data()