# EXPT NO: 3 – Regression and Optimization  
## Multilinear Regression & Polynomial Regression  

**Student:** Jeevanantham K.  
**Roll No:** 24BAD047  
**Course:** Machine Learning Laboratory  
**Semester:** IV  
**Date:** 04.02.2026  

---

## 📌 About the Experiment
This experiment focuses on implementing and analyzing **advanced regression techniques** using real-world datasets.

Two scenarios are covered:
1. **Multilinear Regression** – Predicting student academic performance using multiple features.
2. **Polynomial Regression** – Predicting vehicle fuel efficiency using a non-linear model.

The experiment also includes **model optimization using regularization techniques** such as Ridge and Lasso.

---

## 🎯 Aim
To implement:
- Multilinear Regression for multi-feature prediction.
- Polynomial Regression for non-linear relationships.
- Model evaluation and optimization using regularization.

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

### 1. Student Performance Dataset  
Source: Kaggle (Public)  
Link: https://www.kaggle.com/datasets/spscientist/students-performance-in-exams  

**Target Variable:**  
Final Exam Score (Average of Math, Reading, Writing)

---

### 2. Auto MPG Dataset  
Source: Kaggle (Public)  
Link: https://www.kaggle.com/datasets/uciml/autompg-dataset  

**Target Variable:**  
Miles Per Gallon (MPG)

**Input Feature:**  
Engine Horsepower

---

> Note: Datasets are not included in this repository because of large file size.  
> Please download them manually from Kaggle using the above links.

---

## ⚙️ Implemented Tasks

### Scenario 1 – Multilinear Regression
- Loaded student performance dataset.
- Encoded categorical features.
- Created target variable (final score).
- Handled missing values using imputation.
- Applied feature scaling.
- Split data into training and testing sets.
- Trained Multilinear Regression model.
- Evaluated using:
  - MSE  
  - RMSE  
  - R² Score  
- Applied Ridge and Lasso regularization.
- Visualized:
  - Actual vs Predicted scores  
  - Coefficient comparison  
  - Residual distribution  

---

### Scenario 2 – Polynomial Regression
- Loaded Auto MPG dataset.
- Selected horsepower as independent variable.
- Handled missing values.
- Generated polynomial features (Degree 2, 3, 4).
- Trained Polynomial Regression models.
- Evaluated using:
  - MSE  
  - RMSE  
  - R² Score  
- Compared performance across degrees.
- Applied Ridge regression to reduce overfitting.
- Visualized:
  - Polynomial curves  
  - Overfitting vs underfitting  

---
