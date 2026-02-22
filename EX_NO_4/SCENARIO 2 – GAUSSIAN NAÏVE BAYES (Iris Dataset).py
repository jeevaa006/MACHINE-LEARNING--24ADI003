# EXPT NO:4
# Roll No: 24BAD047
# Name: Jeevanantham K.
# Probabilistic Classification using Gaussian Naïve Bayes

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load Dataset
iris = load_iris()
X = iris.data
y = iris.target

# 2. Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 4. Train Model
model = GaussianNB()
model.fit(X_train, y_train)

# 5. Prediction
y_pred = model.predict(X_test)

# 6. Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# 7. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap='Greens')
plt.title("Confusion Matrix - Iris")
plt.show()

# 8. Class Probabilities
probs = model.predict_proba(X_test[:5])
print("Class Probabilities (First 5 Samples):\n", probs)