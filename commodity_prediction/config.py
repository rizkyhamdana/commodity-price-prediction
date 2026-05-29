"""Shared application configuration."""

HISTORY_FILE = "data/commodity_history.json"
OUTPUT_DIR = "output"
FORECAST_DAYS = 7
USE_LIVE_API = True
BASE_URL = ""

COMMODITIES = [
    "Beras",
    "Daging Ayam",
    "Daging Sapi",
    "Telur Ayam",
    "Bawang Merah",
    "Bawang Putih",
    "Cabai Merah",
    "Cabai Rawit",
    "Minyak Goreng",
    "Gula Pasir",
]

COMMODITY_UNITS = {
    "Beras": "kg",
    "Daging Ayam": "kg",
    "Daging Sapi": "kg",
    "Telur Ayam": "kg",
    "Bawang Merah": "kg",
    "Bawang Putih": "kg",
    "Cabai Merah": "kg",
    "Cabai Rawit": "kg",
    "Minyak Goreng": "lt",
    "Gula Pasir": "kg",
}

MODEL_NAMES = ["arima", "ets", "prophet", "xgboost"]
