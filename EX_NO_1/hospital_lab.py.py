import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\Jeevanantham\ML\temp2\diabetes.csv")

print(df.head())
print(df.tail())

df.info()
print(df.describe())
print(df.isnull().sum())

plt.figure(figsize=(8,5))
plt.hist(df["Glucose"], bins=20)
plt.xlabel("Glucose Level")
plt.ylabel("Number of Patients")
plt.title("Distribution of Glucose Levels")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))
plt.boxplot(df["Glucose"])
plt.ylabel("Glucose Level")
plt.title("Boxplot of Glucose Levels")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,5))
plt.hist(df["Age"], bins=15)
plt.xlabel("Age")
plt.ylabel("Number of Patients")
plt.title("Age Distribution of Patients")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))
plt.boxplot(df["Age"])
plt.ylabel("Age")
plt.title("Boxplot of Age Distribution")
plt.tight_layout()
plt.show()
