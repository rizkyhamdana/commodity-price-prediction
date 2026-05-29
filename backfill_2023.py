#!/usr/bin/env python3
"""
Temporary script to backfill historical data back to 2021
for Commodities, Exchange Rates, and Weather datasets.
"""

import os
import json
import logging
from datetime import datetime, timedelta
import requests
import pandas as pd
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = "data"
COMMODITY_FILE = os.path.join(DATA_DIR, "commodity_history.json")
EXCHANGE_FILE = os.path.join(DATA_DIR, "exchange_rate_database.json")
WEATHER_FILE = os.path.join(DATA_DIR, "exogenous_history_database.json")

# Start date for backfill (2021-01-01)
BACKFILL_START = "2021-01-01"
BACKFILL_START_DT = datetime(2021, 1, 1)

def backfill_commodity():
    logger.info("=== 🌾 BACKFILLING COMMODITY DATA (2021) ===")
    from commodity_prediction.infrastructure.data import load_json_data, update_history_with_api
    
    if not os.path.exists(COMMODITY_FILE):
        logger.error(f"Commodity file not found at: {COMMODITY_FILE}")
        return
        
    df, _ = load_json_data(COMMODITY_FILE)
    logger.info(f"Updating commodity dataset back to {BACKFILL_START}...")
    
    # Force start date to BACKFILL_START to pull historical data
    df_updated = update_history_with_api(df, COMMODITY_FILE, force_start_date=BACKFILL_START)
    
    # Save updated file
    with open(COMMODITY_FILE, "w", encoding="utf-8") as f:
        json.dump({"data": df_updated.to_dict(orient="records")}, f, indent=2, ensure_ascii=False)
    logger.info("✅ Commodity data backfilled successfully!")

def backfill_exchange():
    logger.info("=== 📈 BACKFILLING EXCHANGE RATE DATA (2021) ===")
    
    # Load existing database
    local_db = {}
    if os.path.exists(EXCHANGE_FILE):
        with open(EXCHANGE_FILE, "r", encoding="utf-8") as f:
            local_db = json.load(f)
            
    # Fetch from BACKFILL_START to today
    start_date = BACKFILL_START_DT
    end_date = datetime.now()
    
    period1 = int(start_date.timestamp())
    period2 = int(end_date.timestamp())
    
    ticker = "IDR=X"
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {
        "period1": period1,
        "period2": period2,
        "interval": "1d",
        "events": "history"
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    logger.info(f"Fetching USD/IDR exchange rates from Yahoo Finance since {start_date.strftime('%Y-%m-%d')}...")
    try:
        response = requests.get(url, headers=headers, params=params, timeout=20)
        if response.status_code == 200:
            data = response.json()
            chart = data.get("chart", {})
            result_node = chart.get("result", [])
            
            if not result_node:
                logger.error("No results returned from Yahoo Finance.")
                return
                
            indicators = result_node[0].get("indicators", {})
            quote = indicators.get("quote", [])
            timestamps = result_node[0].get("timestamp", [])
            
            if not quote or not timestamps:
                logger.error("Indicators data is empty.")
                return
                
            close_prices = quote[0].get("close", [])
            temp_db = {}
            for ts, close in zip(timestamps, close_prices):
                if ts is None or close is None:
                    continue
                dt_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
                temp_db[dt_str] = round(float(close), 2)
            
            # Interpolate and merge
            curr = start_date
            last_valid_rate = 14000.0  # Safe initial fallback for 2021
            added_count = 0
            
            while curr <= end_date:
                curr_str = curr.strftime("%Y-%m-%d")
                if curr_str in temp_db:
                    last_valid_rate = temp_db[curr_str]
                
                if curr_str not in local_db:
                    local_db[curr_str] = last_valid_rate
                    added_count += 1
                curr += timedelta(days=1)
                
            # Save back
            with open(EXCHANGE_FILE, "w", encoding="utf-8") as f:
                json.dump(local_db, f, indent=2, ensure_ascii=False)
            logger.info(f"✅ Exchange rates backfilled! Added {added_count} records. Total records: {len(local_db)}")
        else:
            logger.error(f"Yahoo Finance API failed with status code {response.status_code}")
    except Exception as e:
        logger.error(f"Error fetching exchange rate: {e}")

def backfill_weather():
    logger.info("=== 🌧️ BACKFILLING WEATHER DATA (2021) ===")
    
    local_db = {}
    if os.path.exists(WEATHER_FILE):
        with open(WEATHER_FILE, "r", encoding="utf-8") as f:
            local_db = json.load(f)
            
    start_str = BACKFILL_START
    # Open-Meteo archive API has a 2-5 day delay. We use 5 days ago to ensure the data exists in the archive database.
    end_str = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
    
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": -7.82,
        "longitude": 112.01,
        "start_date": start_str,
        "end_date": end_str,
        "daily": "rain_sum",
        "timezone": "Asia/Jakarta"
    }
    
    logger.info(f"Fetching weather historical records from Open-Meteo from {start_str} to {end_str}...")
    try:
        response = requests.get(url, params=params, timeout=20)
        if response.status_code == 200:
            data = response.json()
            daily_data = data.get("daily", {})
            time_list = daily_data.get("time", [])
            rain_list = daily_data.get("rain_sum", [])
            
            added_count = 0
            for t, r in zip(time_list, rain_list):
                if t not in local_db:
                    local_db[t] = float(r) if r is not None else 0.0
                    added_count += 1
                    
            with open(WEATHER_FILE, "w", encoding="utf-8") as f:
                json.dump(local_db, f, indent=2, ensure_ascii=False)
            logger.info(f"✅ Weather data backfilled! Added {added_count} records. Total records: {len(local_db)}")
        else:
            logger.error(f"Open-Meteo API failed with status code {response.status_code}. Response: {response.text}")
    except Exception as e:
        logger.error(f"Error fetching weather data: {e}")

def main():
    logger.info(f"🚀 STARTING COMPLETE {BACKFILL_START} DATA DATASET BACKFILL...")
    backfill_commodity()
    backfill_exchange()
    backfill_weather()
    logger.info("🎉 ALL BACKFILL TASKS COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
