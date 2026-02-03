# EXPT NO: 2 – Regression and Optimization  

## Scenario 1: Linear Regression – Ocean Temperature Prediction  
## Scenario 2: Logistic Regression – LIC Stock Movement  

**Student:** Jeevanantham K.  
**Roll No:** 24BAD047  
**Course:** Machine Learning Laboratory  
**Semester:** IV  

---

## 📌 About the Experiment
This experiment demonstrates the application of **regression techniques** for solving real-world prediction problems.

Two different scenarios are implemented:
1. **Linear Regression** for predicting ocean temperature.
2. **Logistic Regression** for predicting stock movement (up/down).

---

## 🎯 Aim
To understand and implement:
- Linear Regression for continuous value prediction.
- Logistic Regression for classification problems.
- Model training, prediction, and evaluation.

---

## 🛠️ Technologies & Libraries
- Python 3  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- Scikit-learn  

---

## 📂 Datasets Used

### 1. CalCOFI Dataset – Ocean Temperature  
Source: Kaggle  
Link: https://www.kaggle.com/datasets/sohier/calcofi  

Used for predicting **ocean water temperature** based on environmental features.

### 2. LIC Stock Price Dataset  
Source: Kaggle  
Link: https://www.kaggle.com/datasets/debashis74017/lic-stock-price-data  

Used for predicting **stock movement (Up/Down)** using Logistic Regression.

> Note: Datasets are not included in this repository due to large file size.  
> Please download them manually from the above links.

---

## ⚙️ Implemented Tasks

### Scenario 1 – Linear Regression
- Loaded CalCOFI dataset.
- Selected relevant features.
- Handled missing values.
- Applied feature scaling.
- Trained Linear Regression model.
- Predicted ocean temperature.
- Evaluated using MSE, RMSE, and R² score.
- Visualized predicted vs actual values.

### Scenario 2 – Logistic Regression
- Loaded LIC stock dataset.
- Created binary target variable (Up/Down).
- Performed data preprocessing.
- Split dataset into training and testing sets.
- Trained Logistic Regression model.
- Predicted stock movement.
- Evaluated using:
  - Accuracy
  - Confusion Matrix
  - Classification Report

---
