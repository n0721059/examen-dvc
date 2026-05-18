import pandas as pd
from sklearn.model_selection import train_test_split
import os

# --- Load raw data ---
df = pd.read_csv("data/raw_data/raw.csv")

# Drop the date column if present (non-numeric, not useful for regression)
if "date" in df.columns:
    df = df.drop(columns=["date"])

# Target is the last column
target_col = df.columns[-1]          # "% Silica Concentrate"
X = df.drop(columns=[target_col])
y = df[[target_col]]

# 80/20 split, fixed seed for reproducibility
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Save outputs
os.makedirs("data/processed", exist_ok=True)
X_train.to_csv("data/processed/X_train.csv", index=False)
X_test.to_csv("data/processed/X_test.csv", index=False)
y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)

print(f"Split complete — train: {len(X_train)}, test: {len(X_test)}")
