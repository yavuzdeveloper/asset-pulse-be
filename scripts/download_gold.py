
import csv
from pathlib import Path

import httpx

API_URL = "https://xaus.com/api/v1/history"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_FILE = OUTPUT_DIR / "gold_daily.csv"


def main():
    print("Downloading gold historical data...")

    response = httpx.get(API_URL, timeout=30)
    response.raise_for_status()

    payload = response.json()
    points = payload.get("points", [])

    if not points:
        raise RuntimeError("No historical data received from XAUS API.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["date", "close", "high", "low"]
        )

        writer.writeheader()

        for point in points:
            writer.writerow({
                "date": point["d"],
                "close": point["c"],
                "high": point["h"],
                "low": point["l"],
            })

    print(f"Successfully downloaded {len(points)} records!")
    print(f"Saved to: {OUTPUT_FILE}")
    print(f"Date range: {points[0]['d']} - {points[-1]['d']}")


if __name__ == "__main__":
    main()
