# =========================================================
# EXPT NO:5 - Decision Tree Classification
# Roll No: 24BAD047
# Name: Jeevanantham K.
# Date: 19-02-2026
# =========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ------------------------------
# 1. Load Dataset
# ------------------------------
data = pd.read_csv(r"C:\Users\Admin\Desktop\4_sem\ML\EX_NO_5\train_u6lujuX_CVtuZ9i (1).csv")  # Change path

print(data.head())
print(data.info())

# ------------------------------
# 2. Handle Missing Values
# ------------------------------
data.fillna(method='ffill', inplace=True)

# ------------------------------
# 3. Encode Categorical Variables
# ------------------------------
le = LabelEncoder()

for col in data.select_dtypes(include='object').columns:
    data[col] = le.fit_transform(data[col])

# ------------------------------
# 4. Select Features
# ------------------------------
features = ['ApplicantIncome','LoanAmount','Credit_History','Education','Property_Area']
X = data[features]
y = data['Loan_Status']

# ------------------------------
# 5. Train Test Split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# ------------------------------
# 6. Train Decision Tree
# ------------------------------
dt = DecisionTreeClassifier(max_depth=4, random_state=42)
dt.fit(X_train, y_train)

y_pred = dt.predict(X_test)

# ------------------------------
# 7. Evaluation
# ------------------------------
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix - Decision Tree")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ------------------------------
# 8. Tree Structure
# ------------------------------
plt.figure(figsize=(15,10))
plot_tree(dt, feature_names=features, class_names=True, filled=True)
plt.show()

# ------------------------------
# 9. Feature Importance
# ------------------------------
importance = pd.Series(dt.feature_importances_, index=features)

plt.figure()
importance.plot(kind='bar')
plt.title("Feature Importance")
plt.show()