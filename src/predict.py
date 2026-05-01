import joblib
import pandas as pd


def predict_transaction(transaction_data, model_path="models/fraud_model.pkl"):
    bundle = joblib.load(model_path)

    model = bundle["model"]
    preprocessor = bundle["preprocessor"]
    columns = bundle["columns"]

    input_df = pd.DataFrame([transaction_data])
    input_df = input_df[columns]

    processed_input = preprocessor.transform(input_df)

    prediction = model.predict(processed_input)[0]
    probability = model.predict_proba(processed_input)[0][1]

    if probability >= 0.75:
        risk_level = "High Risk"
    elif probability >= 0.40:
        risk_level = "Medium Risk"
    else:
        risk_level = "Low Risk"

    return {
        "prediction": int(prediction),
        "fraud_probability": float(probability),
        "risk_level": risk_level
    }