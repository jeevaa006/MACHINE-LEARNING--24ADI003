import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset (change path if required)
<<<<<<< HEAD
df = pd.read_csv(r"C:\Users\jeevanantham\Documents\ML\temp3\Housing.csv")
=======
df = pd.read_csv(r"C:\Users\Jeevanantham\Documents\ML\temp3\Housing.csv")
>>>>>>> dc9aca76e1392664093244a7807b0e7125390219

# Inspect dataset
print("First 5 rows:")
print(df.head())

print("\nLast 5 rows:")
print(df.tail())

print("\nDataset Info:")
df.info()

print("\nStatistical Summary:")
print(df.describe())
print("\nMissing values in each column:")
print(df.isnull().sum())

plt.figure(figsize=(8,5))
plt.scatter(df["area"], df["price"])
plt.xlabel("Area (sq.ft)")
plt.ylabel("Price")
plt.title("Area vs House Price")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,5))
plt.scatter(df["bedrooms"], df["price"])
plt.xlabel("Number of Bedrooms")
plt.ylabel("Price")
plt.title("Bedrooms vs House Price")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,6))
numeric_df = df.select_dtypes(include=["int64", "float64"])
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap of Housing Features")
plt.tight_layout()
plt.show()
