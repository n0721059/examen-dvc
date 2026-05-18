import pandas as pd
import pickle
import json
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import os

# --- Load test data and trained model ---
X_test = pd.read_csv("data/processed/X_test_scaled.csv")
y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

with open("models/gbr_model.pkl", "rb") as f:
    model = pickle.load(f)

# Generate predictions
predictions = model.predict(X_test)

# Compute metrics
mse  = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_test, predictions)
r2   = r2_score(y_test, predictions)

scores = {
    "MSE":  round(mse, 6),
    "RMSE": round(rmse, 6),
    "MAE":  round(mae, 6),
    "R2":   round(r2, 6),
}
print("Evaluation scores:", scores)

# Save predictions
os.makedirs("data/processed", exist_ok=True)
pred_df = pd.DataFrame({"y_true": y_test, "y_pred": predictions})
pred_df.to_csv("data/processed/prediction.csv", index=False)

# Save metrics as JSON (DVC will track this as a metric)
os.makedirs("metrics", exist_ok=True)
with open("metrics/scores.json", "w") as f:
    json.dump(scores, f, indent=2)

print("Evaluation complete — scores.json and prediction.csv saved")