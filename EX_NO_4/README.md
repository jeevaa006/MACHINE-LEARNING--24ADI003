# EXPT NO: 4  
## Probabilistic Classification using Naïve Bayes  
**Roll No:** 24BAD047  
**Name:** Jeevanantham K.  
**Course:** Machine Learning Laboratory  

---

## 📖 Overview

This experiment demonstrates probabilistic classification using:

- **Multinomial Naïve Bayes** for text classification (SMS Spam Detection)
- **Gaussian Naïve Bayes** for numerical feature classification (Iris Dataset)

The objective is to understand how Naïve Bayes applies Bayes’ Theorem for supervised classification problems.

---

# 🔹 Scenario 1 – Multinomial Naïve Bayes (SMS Spam Classification)

## 📌 Problem Statement
Classify SMS messages as:
- **Spam**
- **Ham (Not Spam)**

## 📂 Dataset

SMS Spam Collection Dataset (Public Dataset)

🔗 Dataset Link:  
https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

## 📊 Features

- **Input:** SMS Text Messages  
- **Target:** Spam / Ham  

## ⚙ Steps Performed

1. Loaded dataset using pandas  
2. Performed text preprocessing:
   - Lowercase conversion  
   - Punctuation removal  
3. Converted text into numerical features using:
   - TF-IDF Vectorization  
4. Encoded labels (Spam = 1, Ham = 0)  
5. Split dataset into training and testing sets  
6. Trained **Multinomial Naïve Bayes classifier**  
7. Evaluated using:
   - Accuracy  
   - Precision  
   - Recall  
   - F1 Score  
8. Generated:
   - Confusion Matrix  
   - Top spam influencing words  
   - Class probability outputs  

## 📈 Results

- High Accuracy (~98–100%)  
- Strong precision and recall for spam detection  
- Clear separation between spam and ham messages  

---

# 🔹 Scenario 2 – Gaussian Naïve Bayes (Iris Classification)

## 📌 Problem Statement
Classify flower species based on physical measurements.

## 📂 Dataset

Iris Dataset (Built-in sklearn dataset)

🔗 Dataset Information:  
https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html

## 🌸 Features

- Sepal Length  
- Sepal Width  
- Petal Length  
- Petal Width  

Target Classes:
- Setosa  
- Versicolor  
- Virginica  

## ⚙ Steps Performed

1. Loaded dataset using sklearn  
2. Applied feature scaling using StandardScaler  
3. Split dataset into training and testing sets  
4. Trained **Gaussian Naïve Bayes classifier**  
5. Evaluated using:
   - Accuracy  
   - Precision  
   - Recall  
   - F1 Score  
6. Generated:
   - Confusion Matrix  
   - Class probability analysis  

## 📈 Results

- High classification accuracy (~95–100%)  
- Effective probabilistic predictions  
- Good separation between flower species  

---

# 🧠 Key Concepts Used

- Bayes’ Theorem  
- Prior Probability  
- Likelihood  
- Conditional Independence Assumption  
- Laplace Smoothing  
- TF-IDF Vectorization  
- Gaussian Distribution  

---

# 🛠 Technologies Used

- Python  
- pandas  
- numpy  
- matplotlib  
- seaborn  
- scikit-learn  

---
