import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
import os

# --- Load split data ---
X_train = pd.read_csv("data/processed/X_train.csv")
X_test  = pd.read_csv("data/processed/X_test.csv")

# Fit scaler on TRAIN only — never fit on test data
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X_train.columns
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),       # transform only, not fit_transform
    columns=X_test.columns
)

# Save scaled datasets
os.makedirs("data/processed", exist_ok=True)
X_train_scaled.to_csv("data/processed/X_train_scaled.csv", index=False)
X_test_scaled.to_csv("data/processed/X_test_scaled.csv", index=False)

# Save fitted scaler for future inference
os.makedirs("models", exist_ok=True)
with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Normalization complete — scaler saved to models/scaler.pkl")