import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\Jeevanantham\ML\temp\data.csv", encoding="latin1")
print(df.head())
print(df.tail())
df.info()
df.describe()

# Check missing values
print(df.isnull().sum())

# Create Sales column
df["Sales"] = df["Quantity"] * df["UnitPrice"]

# Top 10 products by total sales
sales_per_product = (
    df.groupby("Description")["Sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

# Bar Chart
plt.figure(figsize=(12,6))
sales_per_product.plot(kind="bar")
plt.xlabel("Product")
plt.ylabel("Total Sales")
plt.title("Top 10 Products by Sales")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# Line Chart
plt.figure(figsize=(12,6))
sales_per_product.plot(kind="line", marker='o')
plt.xlabel("Product")
plt.ylabel("Total Sales")
plt.title("Top 10 Products by Sales")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()
