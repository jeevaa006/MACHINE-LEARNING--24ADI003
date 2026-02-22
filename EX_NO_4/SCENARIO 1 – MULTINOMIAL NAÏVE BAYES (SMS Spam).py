# EXPT NO:4
# Roll No: 24BAD047
# Name: Jeevanantham K.
# Probabilistic Classification using Multinomial Naïve Bayes

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# 1. Load Dataset
data = pd.read_csv(r"C:\Users\Admin\Desktop\4_sem\ML\EX_NO_4\spam.csv", encoding='latin-1')[['v1','v2']]
data.columns = ['label','message']

# 2. Text Cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

data['message'] = data['message'].apply(clean_text)

# 3. Encode Labels
encoder = LabelEncoder()
data['label'] = encoder.fit_transform(data['label'])  # spam=1, ham=0

# 4. Vectorization (TF-IDF)
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(data['message'])
y = data['label']

# 5. Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Train Model with Laplace Smoothing (alpha=1)
model = MultinomialNB(alpha=1)
model.fit(X_train, y_train)

# 7. Prediction
y_pred = model.predict(X_test)

# 8. Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# 9. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# 10. Top Spam Words
feature_names = np.array(vectorizer.get_feature_names_out())
spam_indices = np.argsort(model.feature_log_prob_[1])[-10:]
print("Top Words Influencing Spam:")
print(feature_names[spam_indices])