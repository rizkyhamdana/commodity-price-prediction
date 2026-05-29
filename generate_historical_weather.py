#!/usr/bin/env python3
"""
Standalone Scraper: Penarik Data Cuaca Historis Riil (Open-Meteo Kediri).
Jalankan script ini sekali saja untuk menginisialisasi/melengkapi database historis.
"""

from datetime import datetime
import json
import os
import requests

DB_DIR = "data"
DB_PATH = os.path.join(DB_DIR, "exogenous_history_database.json")
LATITUDE = -7.82
LONGITUDE = 112.01


def load_local_db() -> dict:
    if not os.path.exists(DB_PATH):
        return {}
    try:
        with open(DB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_local_db(data: dict):
    os.makedirs(DB_DIR, exist_ok=True)
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"💾 Sukses menyimpan/memperbarui database lokal di: {DB_PATH}")


def download_history():
    print("🚀 MEMULAI SCRAPER DATA CUACA HISTORIS KEDIRI...")
    local_db = load_local_db()
    
    # Menetapkan rentang sejarah panjang (1 Januari 2024 s/d Hari Ini)
    start_str = "2024-01-01"
    end_str = datetime.now().strftime("%Y-%m-%d")
    
    print(f"🌐 Menembak API Open-Meteo untuk rentang total: {start_str} s/d {end_str}...")
    
    try:
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "start_date": start_str,
            "end_date": end_str,
            "daily": "rain_sum",
            "timezone": "Asia/Jakarta"
        }
        response = requests.get(url, params=params, timeout=15)
        
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
            
            save_local_db(local_db)
            print(f"✅ Inisialisasi Berhasil! Menambahkan {added_count} data hari cuaca baru ke database lokal.")
            print(f"📈 Total record tersimpan saat ini: {len(local_db)} hari.")
        else:
            print(f"❌ Gagal memanggil API. Status Code: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Terjadi kesalahan koneksi API: {e}")


if __name__ == "__main__":
    download_history()
