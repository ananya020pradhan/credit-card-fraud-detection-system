from src.data_generator import generate_transaction_data
from src.preprocessing import preprocess_data
from src.train import train_model
from src.evaluate import evaluate_model


def main():
    print("Starting Credit Card Fraud Detection System...\n")

    print("Step 1: Generating dataset...")
    generate_transaction_data()

    print("\nStep 2: Preprocessing data...")
    preprocess_data()

    print("\nStep 3: Training model...")
    train_model()

    print("\nStep 4: Evaluating model...")
    evaluate_model()

    print("\nProject pipeline completed successfully.")
    print("Now run the dashboard using:")
    print("streamlit run app.py")


if __name__ == "__main__":
    main()