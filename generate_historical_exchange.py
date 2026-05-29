#!/usr/bin/env python3
"""
Standalone Scraper: Penarik Data Kurs USD/IDR Historis Riil (Yahoo Finance).
Jalankan script ini sekali saja untuk menginisialisasi/melengkapi database kurs lokal.
"""

from datetime import datetime, timedelta
import json
import os
import requests

DB_DIR = "data"
DB_PATH = os.path.join(DB_DIR, "exchange_rate_database.json")


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
    print(f"💾 Sukses menyimpan/memperbarui database kurs lokal di: {DB_PATH}")


def download_history():
    print("🚀 MEMULAI SCRAPER DATA KURS USD/IDR...")
    local_db = load_local_db()
    
    # 1 Januari 2024 s/d hari ini
    start_date = datetime(2024, 1, 1)
    end_date = datetime.now()
    
    # Konversi ke Unix timestamp untuk API Yahoo Finance
    period1 = int(start_date.timestamp())
    period2 = int(end_date.timestamp())
    
    # Yahoo Finance ticker untuk USD/IDR
    ticker = "IDR=X"
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {
        "period1": period1,
        "period2": period2,
        "interval": "1d",
        "events": "history"
    }
    
    # User-agent standard agar request tidak diblokir
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    print(f"🌐 Menembak API Yahoo Finance untuk kurs USD/IDR ({start_date.strftime('%Y-%m-%d')} s/d {end_date.strftime('%Y-%m-%d')})...")
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            chart = data.get("chart", {})
            result_node = chart.get("result", [])
            
            if not result_node:
                print("❌ Yahoo Finance tidak mengembalikan hasil data.")
                return
                
            indicators = result_node[0].get("indicators", {})
            quote = indicators.get("quote", [])
            timestamps = result_node[0].get("timestamp", [])
            
            if not quote or not timestamps:
                print("❌ Struktur data indikator kosong.")
                return
                
            close_prices = quote[0].get("close", [])
            
            added_count = 0
            
            # Buat mapping tanggal dari timestamp Unix
            temp_db = {}
            for ts, close in zip(timestamps, close_prices):
                if ts is None or close is None:
                    continue
                dt_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
                temp_db[dt_str] = round(float(close), 2)
            
            # Lakukan interpolation/forward-fill sederhana untuk hari Sabtu & Minggu (karena bursa kurs libur)
            curr = start_date
            last_valid_rate = 15400.0  # Default fallback awal Jan 2024
            
            while curr <= end_date:
                curr_str = curr.strftime("%Y-%m-%d")
                if curr_str in temp_db:
                    last_valid_rate = temp_db[curr_str]
                
                # Masukkan ke database jika belum ada
                if curr_str not in local_db:
                    local_db[curr_str] = last_valid_rate
                    added_count += 1
                curr += timedelta(days=1)
                
            save_local_db(local_db)
            print(f"✅ Inisialisasi Kurs Berhasil! Menambahkan {added_count} hari data kurs baru.")
            print(f"📈 Total record kurs tersimpan: {len(local_db)} hari.")
        else:
            print(f"❌ API Yahoo Finance gagal. Status Code: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Terjadi kesalahan saat menarik data kurs: {e}")


if __name__ == "__main__":
    download_history()
