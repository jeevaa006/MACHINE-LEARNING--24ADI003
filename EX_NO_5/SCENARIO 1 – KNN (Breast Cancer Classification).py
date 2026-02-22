# =========================================================
# EXPT NO:5 - KNN Classification
# Roll No: 24BAD047
# Name: Jeevanantham K.
# Date: 19-02-2026
# =========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ------------------------------
# 1. Load Dataset
# ------------------------------
data = pd.read_csv(r"C:\Users\Admin\Desktop\4_sem\ML\EX_NO_5\breast-cancer.csv")  # Change path if needed

print(data.head())
print(data.info())

# ------------------------------
# 2. Select Required Features
# ------------------------------
features = ['radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean']
X = data[features]
y = data['diagnosis']

# ------------------------------
# 3. Encode Target Labels
# ------------------------------
le = LabelEncoder()
y = le.fit_transform(y)

# ------------------------------
# 4. Feature Scaling
# ------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------------
# 5. Train Test Split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

# ------------------------------
# 6. Train KNN with different K
# ------------------------------
accuracy_list = []
k_values = range(1, 21)

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy_list.append(accuracy_score(y_test, y_pred))

# Best K
best_k = k_values[np.argmax(accuracy_list)]
print("Best K:", best_k)

# Train final model
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# ------------------------------
# 7. Evaluation
# ------------------------------
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix - KNN")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ------------------------------
# 8. Accuracy vs K Plot
# ------------------------------
plt.figure()
plt.plot(k_values, accuracy_list)
plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.title("Accuracy vs K")
plt.show()

# ------------------------------
# 9. Misclassified Cases
# ------------------------------
misclassified = np.where(y_test != y_pred)
print("Number of Misclassified:", len(misclassified[0]))