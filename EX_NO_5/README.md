# EXPT NO: 5  
# Classification using K-Nearest Neighbors (KNN) and Decision Tree  

**Name:** Jeevanantham K.  
**Roll No:** 24BAD047  
**Date:** 19.02.2026  
**Course:** Machine Learning Lab  

---

# 📌 OBJECTIVE

To implement and evaluate two supervised machine learning algorithms:
- K-Nearest Neighbors (KNN)
- Decision Tree Classifier

---

# 🔷 SCENARIO 1 – K-NEAREST NEIGHBORS (KNN)

## 🎯 Problem Statement
Predict whether a tumor is **Benign or Malignant** based on medical measurements.

## 📂 Dataset
Breast Cancer Dataset (Kaggle – Public)

🔗 Dataset Link:  
https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset

### Target Variable:
Diagnosis (Benign / Malignant)

### Input Features:
- Radius  
- Texture  
- Perimeter  
- Area  
- Smoothness  

---

## ⚙️ Implementation Steps

1. Imported required Python libraries  
2. Loaded the Breast Cancer dataset  
3. Performed data inspection and preprocessing  
4. Encoded target labels  
5. Applied feature scaling using StandardScaler  
6. Split dataset into training and testing sets  
7. Trained KNN classifier  
8. Experimented with different K values  
9. Predicted diagnosis labels  
10. Evaluated model using:
   - Accuracy  
   - Precision  
   - Recall  
   - F1 Score  
11. Identified misclassified cases  
12. Analyzed model sensitivity to K  

---

## 📊 Visualizations

- Confusion Matrix  
- Accuracy vs K Plot  
- Decision Boundary (using two features)  

---

## 📈 Observations

- Smaller K values caused overfitting.  
- Moderate K values gave best performance.  
- Feature scaling significantly improved accuracy.  

---

# 🔷 SCENARIO 2 – DECISION TREE CLASSIFIER

## 🎯 Problem Statement
Predict whether a loan application should be **Approved or Rejected**.

## 📂 Dataset
Loan Prediction Dataset (Kaggle – Public)

🔗 Dataset Link:  
https://www.kaggle.com/datasets/ninzaami/loan-predication

### Target Variable:
Loan Status (Approved / Rejected)

### Input Features:
- Applicant Income  
- Loan Amount  
- Credit History  
- Education  
- Property Area  

---

## ⚙️ Implementation Steps

1. Imported required Python libraries  
2. Loaded Loan Prediction dataset  
3. Handled missing values  
4. Encoded categorical variables  
5. Split dataset into training and testing sets  
6. Trained Decision Tree classifier  
7. Tuned tree depth / pruning  
8. Predicted loan status  
9. Evaluated model using:
   - Accuracy  
   - Precision  
   - Recall  
   - F1 Score  
10. Analyzed feature importance  
11. Compared shallow vs deep trees  
12. Detected overfitting behavior  

---

## 📊 Visualizations

- Confusion Matrix  
- Tree Structure Plot  
- Feature Importance Plot  

---

# 🔍 Model Comparison

| KNN | Decision Tree |
|------|--------------|
| Distance-based | Rule-based |
| Requires feature scaling | No scaling required |
| Slower prediction | Faster prediction |
| Sensitive to noise | More stable |

---

# 🛠 Tools & Libraries Used

- Python  
- NumPy  
- Pandas  
- Matplotlib  
- Seaborn  
- Scikit-learn  

---
