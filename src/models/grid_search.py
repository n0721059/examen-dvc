import pandas as pd
import pickle
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
import os

# --- Load scaled training data ---
X_train = pd.read_csv("data/processed/X_train_scaled.csv")
y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()

# Define parameter grid
param_grid = {
    "n_estimators":    [100, 200],
    "max_depth":       [3, 5],
    "learning_rate":   [0.05, 0.1],
    "subsample":       [0.8, 1.0],
}

model = XGBRegressor(random_state=42, n_jobs=-1)

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=3,
    scoring="neg_mean_squared_error",
    verbose=1,
    n_jobs=-1,
)

grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print(f"Best params: {best_params}")
print(f"Best CV score (neg MSE): {grid_search.best_score_:.4f}")

# Save best params
os.makedirs("models", exist_ok=True)
with open("models/best_params.pkl", "wb") as f:
    pickle.dump(best_params, f)

# Save full grid search results for reference
pd.DataFrame(grid_search.cv_results_).to_csv(
    "models/grid_search_results.csv", index=False
)

print("Grid search complete — best_params.pkl saved")