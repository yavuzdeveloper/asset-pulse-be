
import pandas as pd
import joblib

from pathlib import Path
from sklearn.metrics import mean_absolute_error

ROOT_DIR = Path(__file__).resolve().parent.parent

df = pd.read_csv(ROOT_DIR / "data/processed/dataset.csv")

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

split_index = int(len(df) * 0.8)

X_test = df[features].iloc[split_index:]
y_test = df["target"].iloc[split_index:]

model = joblib.load(
    ROOT_DIR / "models/random_forest.joblib"
)

predictions = model.predict(X_test)

results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": predictions,
})

results["Difference"] = (
    results["Actual"] - results["Predicted"]
)

print("\nActual vs Predicted:")
print(results.tail(15).round(2))

print("\nMAE:", round(
    mean_absolute_error(y_test, predictions), 2
))
