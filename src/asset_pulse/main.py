import csv
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(
    title="AssetPulse API",
    description="Real-time Asset Intelligence Platform",
    version="1.0.0"
)


# Next.js frontend will run on http://localhost:3000, so we allow CORS for that origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {
        "message": "Welcome to AssetPulse API",
        "status": "running"
    }


# Metals endpoint
@app.get("/metals")
def get_metals():
    return {
        "metals": [
            {
                "symbol": "XAU",
                "name": "Gold",
                "price": 3985.50,
                "currency": "USD",
                "change_percent": 1.25
            },
            {
                "symbol": "XAG",
                "name": "Silver",
                "price": 48.75,
                "currency": "USD",
                "change_percent": -0.65
            }
        ]
    }


# XAUS API gold and silver spot prices
@app.get("/metals/spot")
def get_metals_spot():
    try:
        response = httpx.get(
            "https://xaus.com/api/v1/spot",
            timeout=10
        )
        response.raise_for_status()

        data = response.json()

        return {
            "gold": {
                "price": data["spot_usd_oz"],
                "currency": "USD",
                "unit": "troy_oz"
            },
            "silver": {
                "price": data["silver_usd_oz"],
                "currency": "USD",
                "unit": "troy_oz"
            },
            "updated_at": data["updated_at"]
        }

    except (httpx.HTTPError, KeyError, TypeError) as error:
        raise HTTPException(
            status_code=502,
            detail=f"Could not retrieve metal prices: {error}"
        )


# Gold 5 years data
@app.get("/metals/gold/history")
def get_gold_history():
    project_root = Path(__file__).resolve().parents[2]
    csv_file = project_root / "data" / "raw" / "gold_daily.csv"

    if not csv_file.exists():
        raise HTTPException(
            status_code=404,
            detail="Gold historical data not found"
        )

    with csv_file.open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        prices = [
            {
                "date": row["date"],
                "close": float(row["close"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
            }
            for row in reader
        ]

    if not prices:
        raise HTTPException(
            status_code=404,
            detail="Gold historical data is empty"
        )

    return {
        "symbol": "XAU",
        "currency": "USD",
        "unit": "troy_oz",
        "prices": prices
    }
