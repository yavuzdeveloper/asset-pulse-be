
import pandas as pd

# Historical data
df = pd.read_csv("data/raw/gold_daily.csv")

# Convert date and sort
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# Daily returns
df["return_1d"] = df["close"].pct_change()
df["return_5d"] = df["close"].pct_change(5)

# Moving averages
df["ma_7"] = df["close"].rolling(7).mean()
df["ma_14"] = df["close"].rolling(14).mean()
df["ma_30"] = df["close"].rolling(30).mean()

# Volatility
df["volatility_7"] = df["return_1d"].rolling(7).std()

# Remove rows without enough historical data
df = df.dropna().reset_index(drop=True)

print("Prepared dataset:")
print(df.head())

print("\nDataset shape:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())

# Next trading day's closing price
df["target"] = df["close"].shift(-1)

# Remove the last row because its target is unknown
df = df.dropna().reset_index(drop=True)

print("\nFeatures and target:")
print(df[["date", "close", "ma_7", "ma_14", "ma_30", "target"]].tail(10))

print("\nFinal dataset shape:", df.shape)



from pathlib import Path

# Save processed dataset
output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "dataset.csv"

df.to_csv(output_path, index=False)

print(f"\nDataset saved successfully: {output_path}")
