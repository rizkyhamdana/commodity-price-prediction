"""FastAPI app for serving generated commodity market data."""

import json
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from commodity_prediction.config import OUTPUT_DIR

JSON_PATH = os.path.join(OUTPUT_DIR, "mobile_backend.json")

app = FastAPI(title="Commodity Price API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {"message": "Commodity API is Running", "status": "active"}


@app.get("/api/market-summary")
def get_market_summary():
    """Endpoint utama untuk diambil oleh Flutter."""
    if not os.path.exists(JSON_PATH):
        raise HTTPException(status_code=404, detail="Data belum digenerate. Jalankan run_all.py terlebih dahulu.")

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def run_api_server():
    import uvicorn

    print("🚀 API Server starting on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
