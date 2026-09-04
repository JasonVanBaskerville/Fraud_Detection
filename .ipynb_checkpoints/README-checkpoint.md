# 💳 Fraud Detection

A machine learning project to detect fraudulent financial transactions using transaction characteristics and machine learning classification.

## 🚀 Live Demo

### 📊 Analyst Dashboard

An interactive dashboard for exploring and analyzing transaction data, fraud patterns, and model-related insights.

👉 **[Open Analyst Dashboard](https://analystdashboard-frauddetection-ecbnspyrdpwxskuyjwz3xg.streamlit.app/)**

### 💳 Fraud Detection App

An application to predict whether a financial transaction is potentially fraudulent based on its transaction characteristics.

👉 **[Open Fraud Detection App](https://frauddetection-muf4ezgnju6bbqyhizrn6w.streamlit.app/)**

---

## 📊 Features

### Analyst Dashboard

* Dataset overview
* Fraud vs. non-fraud transaction distribution
* Transaction type analysis
* Transaction amount distribution
* Fraud rate by transaction type
* Fraud analysis by transaction step
* Transaction amount vs. fraud
* Origin account balance analysis
* Destination account balance analysis
* Correlation analysis
* Interactive filtering
* Fraud pattern exploration

### Fraud Detection App

You can enter transaction information such as:

* Transaction type
* Transaction amount
* Origin account balance before transaction
* Origin account balance after transaction
* Destination account balance before transaction
* Destination account balance after transaction

The application will then generate a prediction indicating whether the transaction is **Fraud** or **Non-Fraud**.

---

## 🤖 Machine Learning Model

The project uses **Logistic Regression** as the classification model.

The preprocessing pipeline includes:

* Numerical feature scaling using `StandardScaler`
* Categorical feature encoding using `OneHotEncoder`
* Class imbalance handling using `class_weight="balanced"`

The model is implemented using a Scikit-learn `Pipeline` and saved as a `.joblib` file for use in the prediction application.

---

## 📈 Model Evaluation

The model is evaluated using several classification metrics:

* Accuracy
* Precision
* Recall
* F1-Score
* Classification Report

Because fraud transactions represent a very small proportion of the dataset, special attention is given to **Fraud Recall and Precision** rather than relying only on accuracy.

Threshold analysis is also performed to investigate the trade-off between:

* Precision
* Recall
* F1-Score

for the Fraud class.

---

## 🛠️ Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Plotly
* Streamlit
* Jupyter Notebook
* Joblib

---

## 📂 Dataset

The dataset contains financial transaction information used to identify potentially fraudulent transactions.

Main features include:

| Feature          | Description                                                          |
| ---------------- | -------------------------------------------------------------------- |
| `step`           | Time step of the transaction                                         |
| `type`           | Type of transaction                                                  |
| `amount`         | Transaction amount                                                   |
| `nameOrig`       | Origin account identifier                                            |
| `oldbalanceOrg`  | Origin account balance before transaction                            |
| `newbalanceOrig` | Origin account balance after transaction                             |
| `nameDest`       | Destination account identifier                                       |
| `oldbalanceDest` | Destination account balance before transaction                       |
| `newbalanceDest` | Destination account balance after transaction                        |
| `isFraud`        | Target variable indicating whether the transaction is fraudulent     |
| `isFlaggedFraud` | Indicates whether the transaction was flagged by the original system |

The dataset used in this study was obtained from Kaggle:
- **Source:** Kaggle
- **Dataset:** AIML Dataset
- **Link:** [Kaggle Dataset](https://www.kaggle.com/datasets/amanalisiddiqui/fraud-detection-dataset?resource=download)

Dataset Setup
1. Download the dataset from Kaggle.
2. Save the `AIML Dataset.csv` file in the project's root folder.
3. Run the notebook or application.

> The dataset used in this project may be sampled or processed to make it suitable for analysis and deployment.

---

## ⚙️ How to Run Locally

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_LINK
cd fraud-detection
```

### 2. Install Dependencies

Create or activate your Python environment, then install the required libraries:

```bash
pip install -r requirements.txt
```

### 3. Run the Jupyter Notebook

Open Jupyter Lab:

```bash
jupyter lab
```

Then open:

```text
notebook/python_fraud_detection.ipynb
```

Run all cells to perform:

* Data preprocessing
* Exploratory Data Analysis
* Feature preparation
* Model training
* Model evaluation
* Threshold analysis

### 4. Run the Analyst Dashboard

Open a terminal and run:

```bash
streamlit run analyst_dashboard.py
```

Example:

```text
Local URL: http://localhost:8501
Network URL: http://192.168.0.8:8501
```

### 5. Run the Fraud Detection App

Open another terminal and run:

```bash
streamlit run app.py
```

Example:

```text
Local URL: http://localhost:8502
Network URL: http://192.168.0.8:8502
```

---

## 🔍 Project Workflow

```text
Raw Transaction Data
        │
        ▼
Data Cleaning & Preprocessing
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Feature Selection
        │
        ▼
Train-Test Split
        │
        ▼
Data Preprocessing
(StandardScaler + OneHotEncoder)
        │
        ▼
Logistic Regression
(class_weight="balanced")
        │
        ▼
Model Evaluation
        │
        ▼
Threshold Analysis
        │
        ▼
Saved Model (.joblib)
        │
        ▼
Streamlit Fraud Detection App
```

---

## ⚠️ Class Imbalance

Fraud detection is a highly imbalanced classification problem because fraudulent transactions represent only a very small portion of all transactions.

Therefore, accuracy alone is not sufficient to evaluate the model.

This project focuses on the performance of the **Fraud class**, particularly:

* **Precision** — how many transactions predicted as fraud are actually fraud.
* **Recall** — how many actual fraudulent transactions are successfully detected.
* **F1-Score** — balance between precision and recall.

The `class_weight="balanced"` parameter is used in Logistic Regression to help the model pay more attention to the minority Fraud class.

---

## 📌 Project Goal

The main goal of this project is to build an end-to-end fraud detection system that combines:

**Data Analysis → Machine Learning → Model Evaluation → Interactive Dashboard → Prediction Application**

The project demonstrates how machine learning can be used to identify potentially fraudulent financial transactions while considering the challenges of highly imbalanced data.
