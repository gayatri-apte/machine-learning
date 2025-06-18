#1) Framingham Heart Disease Prediction using Logistic Regression

This project builds a logistic regression model using the **Framingham Heart Study dataset** to predict whether a patient is at risk of developing **coronary heart disease (CHD)** in the next 10 years.

## 📁 Dataset

- **File:** `framingham.csv`
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Framingham+Heart+Study) or Kaggle
- **Target Variable:** `TenYearCHD` (1 = risk present, 0 = no risk)

## 🔍 Features Used

- `age` – Age of the patient
- `sex_male` – Gender (1 = male, 0 = female)
- `cigsPerDay` – Number of cigarettes per day
- `totChol` – Total cholesterol
- `sysBP` – Systolic blood pressure
- `glucose` – Glucose level

## 🧪 Model and Tools

- **Model:** Logistic Regression (from `sklearn.linear_model`)
- **Libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`
- **Preprocessing:** Standardization using `StandardScaler`
- **Train-Test Split:** 70% training, 30% testing

## 📊 Evaluation Metrics

- **Accuracy Score**
- **Classification Report** (Precision, Recall, F1-Score)
- **Confusion Matrix** (with heatmap visualization)

## 📈 Visualizations

- Bar plot showing distribution of target variable
- Line plot of CHD cases
- Heatmap of the confusion matrix

