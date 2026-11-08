
import pandas as pd
import joblib

from pathlib import Path
from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Project paths
ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT_DIR / "data" / "processed" / "dataset.csv"
MODEL_DIR = ROOT_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)

# Features
features = [
    "close",
    "high",
    "low",
    "return_1d",
    "return_5d",
    "ma_7",
    "ma_14",
    "ma_30",
    "volatility_7",
]

X = df[features]
y = df["target"]

# Chronological train/test split
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# Initialize XGBoost model
model = XGBRegressor(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)

# Train model
print("Training XGBoost model...")

model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, predictions)

rmse = mean_squared_error(
    y_test, predictions
) ** 0.5

r2 = r2_score(y_test, predictions)

# Results
print("\nXGBoost Model Evaluation")
print("------------------------")

print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2:   {r2:.4f}")

# Save model
model_path = MODEL_DIR / "xgboost_model.joblib"

joblib.dump(model, model_path)

print(f"\nModel saved: {model_path}")

# Feature importance
importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance")
print("------------------")
print(importance.to_string(index=False))
