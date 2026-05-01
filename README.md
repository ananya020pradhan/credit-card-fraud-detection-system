# 💳 Credit Card Fraud Detection System

🚀 An end-to-end **Machine Learning + Analytics Dashboard** that detects fraudulent credit card transactions using imbalanced classification techniques and presents results through a premium Streamlit UI.

---

## 📌 Overview

Credit card fraud is a critical challenge in the banking and fintech industry. Fraudulent transactions are rare but can cause significant financial loss.

This project builds a **complete fraud detection system** that:

✔ Detects fraudulent transactions using ML
✔ Handles imbalanced data using SMOTE
✔ Provides real-time fraud prediction
✔ Visualizes transaction patterns
✔ Generates fraud alerts with risk levels
✔ Demonstrates an industry-style ML pipeline

---

## 🎯 Objective

To design a system that:

* Identifies fraud transactions with high accuracy
* Minimizes financial loss
* Maintains a balance between **precision and recall**
* Demonstrates real-world ML problem solving

---

## 🧠 Problem Statement

Fraud detection is challenging because:

* Fraud cases are **very rare (imbalanced dataset)**
* Patterns evolve over time
* Missing fraud (False Negative) is costly
* Too many alerts (False Positive) affect user experience

---

## ⚙️ Tech Stack

### 💻 Programming

* Python

### 📊 Data & Machine Learning

* Pandas
* NumPy
* Scikit-learn
* Imbalanced-learn (SMOTE)

### 📈 Visualization

* Plotly
* Matplotlib
* Seaborn

### 🧩 Model

* Random Forest Classifier

### 🌐 Dashboard

* Streamlit (Premium UI/UX)

---

## 🏗️ System Architecture

```
Transaction Data → Preprocessing → Feature Engineering → Model → Prediction → Risk Scoring → Dashboard
```

---

## 📁 Project Structure

```
credit-card-fraud-detection-system/
│
├── app.py                      # Streamlit dashboard (UI)
├── main.py                     # End-to-end pipeline runner
├── requirements.txt           # Dependencies
├── README.md                  # Project documentation
│
├── data/
│   ├── raw/                   # Generated raw dataset (ignored in git)
│   └── processed/             # Preprocessed data (ignored in git)
│
├── models/                    # Trained model files (auto-generated / ignored in git)
│
├── outputs/
│   └── classification_report.txt   # Model evaluation report
│
├── images/
│   ├── confusion_matrix.png   # Saved confusion matrix
│   └── (other screenshots)    # Dashboard screenshots for README
│
├── src/
│   ├── data_generator.py      # Synthetic data creation
│   ├── preprocessing.py       # Cleaning + feature engineering + SMOTE
│   ├── train.py               # Model training (Random Forest)
│   ├── evaluate.py            # Metrics + confusion matrix
│   └── predict.py             # Inference (fraud probability & risk)
│
└── .gitignore                 # Ignore large/temporary files
```

---

## 🔄 Workflow

1️⃣ Generate synthetic transaction dataset
2️⃣ Preprocess and clean data
3️⃣ Handle class imbalance using SMOTE
4️⃣ Train ML model (Random Forest)
5️⃣ Evaluate performance (Precision, Recall, F1)
6️⃣ Predict fraud probability
7️⃣ Display results via interactive dashboard

---

## 📊 Key Features

* Fraud probability prediction
* Risk classification (Low / Medium / High)
* Interactive data visualization
* Real-time prediction system
* Clean and professional dashboard
* Automatic model training (no manual setup required)

---

## 📈 Model Evaluation

Metrics used:

* Accuracy
* Precision
* Recall (Most Important ⚠️)
* F1 Score
* Confusion Matrix

📌 **Why Recall matters?**
Missing a fraud transaction leads to direct financial loss, so recall is prioritized.

---

## ▶️ How to Run

### Step 1: Clone Repository

```
git clone https://github.com/your-username/credit-card-fraud-detection-system.git
cd credit-card-fraud-detection-system
```

### Step 2: Create Virtual Environment

```
python -m venv venv
source venv/bin/activate      # Mac
venv\Scripts\activate         # Windows
```

### Step 3: Install Dependencies

```
pip install -r requirements.txt
```

### Step 4: Run Project

```
streamlit run app.py
```

👉 The system will automatically:

* Generate dataset
* Preprocess data
* Train model

---

## 📊 Dashboard Features

* 📌 Executive Overview (KPIs)
* 📌 Dataset Analysis
* 📌 Model Performance
* 📌 Live Fraud Prediction
* 📌 Fraud Simulation
* 📌 Project Explanation

---

## 🚨 Output Example

* Fraud Probability (%)
* Prediction (Fraud / Not Fraud)
* Risk Level:

  * 🟢 Low Risk
  * 🟡 Medium Risk
  * 🔴 High Risk

---

## 💼 Industry Relevance

Used in:

* Banks 🏦
* Fintech Companies 💳
* Payment Gateways 💰
* Digital Wallets 📱

This project reflects real-world applications of:

* Fraud detection systems
* Risk scoring engines
* ML pipelines in finance

---

## 🔮 Future Improvements

* FastAPI deployment (real-time API)
* XGBoost / LightGBM models
* SHAP explainability
* Kafka streaming integration
* Cloud deployment (AWS/GCP)

---

## 🎓 Interview Ready Points

**Q: What problem does this solve?**
A: Detects fraudulent transactions to reduce financial loss.

**Q: Why is this difficult?**
A: Because fraud cases are rare and patterns change.

**Q: What metric is most important?**
A: Recall, to avoid missing fraud cases.

**Q: How did you handle imbalance?**
A: Using SMOTE and class balancing.

---

## 👨‍💻 Author

**Ananya Pradhan**

🔗 LinkedIn:  http://www.linkedin.com/in/ananya-pradhan-10bb462ba

---

## ⭐ If you like this project

Give it a ⭐ on GitHub and share it!
