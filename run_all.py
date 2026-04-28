import json
import os
import logging
from datetime import datetime, timedelta
from commodity_arima_pipeline import run_pipeline, load_json_data, extract_commodity_series

# ──────────────────────────────────────────────
#  Konfigurasi Global
# ──────────────────────────────────────────────
HISTORY_FILE   = "Data lengkap Komoditas 6 Bulan.json"
OUTPUT_DIR     = "output"
FORECAST_DAYS  = 7
USE_LIVE_API   = True  

# Ganti dengan domain server Anda nanti, misal: "https://api.anda.com/"
BASE_URL       = "" 

# Daftar Komoditas Utama
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
    "Gula Pasir"
]

def generate_commodity_insight(commodity, trend, pct_change):
    """Memberikan catatan otomatis berdasarkan tren."""
    disclaimer = "Catatan: Prediksi berbasis data historis, faktor eksternal (cuaca, regulasi, hari raya) tetap berpengaruh."
    
    if "NAIK" in trend:
        msg = f"Tren {commodity} cenderung meningkat ({pct_change}%). Disarankan untuk mengamankan stok jika kebutuhan mendesak."
    elif "TURUN" in trend:
        msg = f"Tren {commodity} sedang menurun ({pct_change}%). Peluang baik untuk pengadaan, namun pantau kualitas stok."
    else:
        msg = f"Tren {commodity} relatif stabil. Pantau pergerakan harga harian untuk perubahan mendadak."
        
    return f"{msg} {disclaimer}"

def generate_global_insight(summary_data):
    """Memberikan analisis pasar secara keseluruhan."""
    naik = [i['name'] for i in summary_data if "NAIK" in i['trend']]
    turun = [i['name'] for i in summary_data if "TURUN" in i['trend']]
    
    analysis = "Pasar hari ini secara umum stabil. "
    if naik:
        analysis += f"Waspadai kenaikan harga pada {', '.join(naik)}. "
    if turun:
        analysis += f"Terdapat penurunan harga yang signifikan pada {', '.join(turun)}, terutama komoditas hortikultura. "
        
    analysis += "Gunakan prediksi ini sebagai referensi pendukung, bukan satu-satunya dasar keputusan."
    return analysis

def run_all_predictions():
    print("\n" + "═" * 60)
    print(" 🚀 STARTING GLOBAL COMMODITY FORECAST PIPELINE")
    print(" 📅 Running all commodities in sequence...")
    print("═" * 60 + "\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
def run_all_predictions():
    print("\n" + "═" * 60)
    print(" 🚀 STARTING MOBILE BACKEND GENERATOR")
    print(" 📱 Preparing data for Flutter Integration...")
    print("═" * 60 + "\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_full, _ = load_json_data(HISTORY_FILE)
    
    success_count = 0
    fail_count = 0
    mobile_data = []

    for i, commodity in enumerate(COMMODITIES, 1):
        print(f"\n[{i}/{len(COMMODITIES)}] PROCESSING: {commodity.upper()}")
        
        try:
            api_status = USE_LIVE_API if i == 1 else False
            
            # 1. Run Forecast
            forecast_df = run_pipeline(
                json_path=HISTORY_FILE,
                commodity_name=commodity,
                n_days=FORECAST_DAYS,
                out_dir=OUTPUT_DIR,
                use_api=api_status
            )
            
            # 2. Get Historical Series for Chart
            series_hist = extract_commodity_series(df_full, commodity)
            last_actual = float(series_hist.iloc[-1])
            
            # 3. Find Sub-Commodities (Level 2)
            sub_items = []
            mask_sub = (df_full["level"] == 2)
            all_subs = df_full[mask_sub]
            
            for _, sub_row in all_subs.iterrows():
                if commodity.lower() in sub_row["name"].lower():
                    prices_series = sub_row.drop(["no", "name", "level"], errors="ignore").dropna()
                    
                    if len(prices_series) >= 2:
                        last_p = float(str(prices_series.iloc[-1]).replace(",", ""))
                        prev_p = float(str(prices_series.iloc[-2]).replace(",", ""))
                        change_val = last_p - prev_p
                        change_pct = (change_val / prev_p * 100) if prev_p != 0 else 0
                        sub_trend = "▲" if change_pct > 0 else ("▼" if change_pct < 0 else "—")
                    else:
                        last_p = float(str(prices_series.iloc[-1]).replace(",", "")) if not prices_series.empty else 0
                        change_pct = 0
                        sub_trend = "—"

                    # Nama file asset untuk sub-komoditas
                    sub_slug = sub_row["name"].lower().replace(" ", "_")

                    sub_items.append({
                        "name": sub_row["name"],
                        "price": last_p,
                        "change_pct": round(change_pct, 2),
                        "trend": sub_trend,
                        "image_asset": f"assets/images/{sub_slug}.png"
                    })

            # 4. Prepare Chart Data (Flutter Friendly)
            history_points = [{"date": d.strftime("%Y-%m-%d"), "price": float(v)} for d, v in series_hist.tail(30).items()]
            forecast_points = [{"date": r["date"], "price": r["predicted_price"]} for _, r in forecast_df.iterrows()]

            # 5. Build Object
            pred_end = float(forecast_df["predicted_price"].iloc[-1])
            pct_change = ((pred_end - last_actual) / last_actual) * 100
            trend = "NAIK 📈" if pct_change > 0.5 else ("TURUN 📉" if pct_change < -0.5 else "STABIL ➡️")
            
            slug = commodity.lower().replace(" ", "_")

            item_data = {
                "name": commodity,
                "current_price": last_actual,
                "pct_change": round(pct_change, 2),
                "trend": trend,
                "image_asset": f"assets/images/{slug}.png",
                "insight": generate_commodity_insight(commodity, trend, round(pct_change, 2)),
                "chart": {
                    "history": history_points,
                    "forecast": forecast_points
                },
                "sub_commodities": sub_items
            }
            mobile_data.append(item_data)
            success_count += 1
            
        except Exception as e:
            print(f"❌ Error on {commodity}: {e}")
            fail_count += 1

    # Final Mobile JSON
    mobile_backend = {
        "metadata": {
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "global_analysis": generate_global_insight(mobile_data),
            "about_us": {
                "app_name": "Komoditas-AI",
                "version": "1.0.0",
                "developer": "Tim Analis Data",
                "description": "Sistem prediksi harga bahan pokok menggunakan AI dan Machine Learning (ARIMA/ETS)."
            }
        },
        "commodities": mobile_data
    }

    mobile_path = os.path.join(OUTPUT_DIR, "mobile_backend.json")
    with open(mobile_path, "w", encoding="utf-8") as f:
        json.dump(mobile_backend, f, indent=2, ensure_ascii=False)

    print("\n" + "═" * 60)
    print(f" ✅ Success : {success_count} Commodities")
    print(f" 📱 Mobile Backend Ready at: {mobile_path}")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    run_all_predictions()
