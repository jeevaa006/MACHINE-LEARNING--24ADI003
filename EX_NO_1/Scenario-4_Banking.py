import pandas as pd
import matplotlib.pyplot as plt

# Load dataset (tab separated)
df = pd.read_csv(r"C:\Users\Jeevanantham\Documents\ML\Exp 1\Temp4\marketing_campaign.csv", sep="\t")

# Dataset overview
df.info()
print(df.isnull().sum())

# Create Age column
df["Age"] = 2025 - df["Year_Birth"]

# Age distribution (Bar Plot)
plt.figure()
df["Age"].value_counts().sort_index().plot(kind="bar")
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Customers")
plt.show()

# Income distribution (Box Plot)
plt.figure()
plt.boxplot(df["Income"].dropna())
plt.title("Income Distribution")
plt.ylabel("Income")
plt.show()

# Spending pattern (Box Plot)
spend = ["MntWines","MntFruits","MntMeatProducts",
         "MntFishProducts","MntSweetProducts","MntGoldProds"]

plt.figure()
df[spend].boxplot()
plt.title("Customer Spending Pattern")
plt.xticks(rotation=45)
plt.show()
