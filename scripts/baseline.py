
import pandas as pd

from pathlib import Path
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

ROOT_DIR = Path(__file__).resolve().parent.parent

df = pd.read_csv(
    ROOT_DIR / "data" / "processed" / "dataset.csv"
)

# Same chronological split as other models
split_index = int(len(df) * 0.8)

test = df.iloc[split_index:]

# Baseline prediction:
# Tomorrow's close = Today's close
predictions = test["close"]

actual = test["target"]

mae = mean_absolute_error(actual, predictions)

rmse = mean_squared_error(
    actual, predictions
) ** 0.5

r2 = r2_score(actual, predictions)

print("\nBASELINE MODEL RESULTS")
print("----------------------")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2:   {r2:.4f}")
