import pandas as pd
import pickle
from xgboost import XGBRegressor
import os

# --- Load data and best params ---
X_train = pd.read_csv("data/processed/X_train_scaled.csv")
y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()

with open("models/best_params.pkl", "rb") as f:
    best_params = pickle.load(f)

print(f"Training with params: {best_params}")

# Train final model
model = XGBRegressor(**best_params, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Save trained model
os.makedirs("models", exist_ok=True)
with open("models/gbr_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Training complete — gbr_model.pkl saved")