import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/heart.csv")

# ----------------------------
# Target Distribution
# ----------------------------
plt.figure(figsize=(6,4))
df["target"].value_counts().plot(kind="bar")
plt.title("Heart Disease Distribution")
plt.xlabel("Target")
plt.ylabel("Count")
plt.xticks([0,1],["No Disease","Disease"], rotation=0)
plt.tight_layout()
plt.show()

# ----------------------------
# Age Distribution
# ----------------------------
plt.figure(figsize=(6,4))
plt.hist(df["age"], bins=15)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# ----------------------------
# Correlation Matrix
# ----------------------------
plt.figure(figsize=(10,8))
plt.imshow(df.corr(), interpolation="nearest")
plt.colorbar()
plt.xticks(range(len(df.columns)), df.columns, rotation=90)
plt.yticks(range(len(df.columns)), df.columns)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()