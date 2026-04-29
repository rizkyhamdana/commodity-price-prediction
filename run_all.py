import json
import os
import logging
from datetime import datetime, timedelta
from commodity_forecaster import run_pipeline, load_json_data, extract_commodity_series, update_history_with_api
from openai import OpenAI
from dotenv import load_dotenv

# Muat .env file jika ada
load_dotenv()

# ──────────────────────────────────────────────
#  Setup LLM (OpenAI)
# ──────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
openai_client = None
if OPENAI_API_KEY:
    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        print(f"⚠️ Gagal setup LLM Client: {e}")

# ──────────────────────────────────────────────
#  Konfigurasi Global
# ──────────────────────────────────────────────
HISTORY_FILE   = "commodity_history.json"
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

def generate_commodity_insight(name, trend, forecast_pct, alert):
    """Menghasilkan kalimat analisis yang manusiawi dan tidak kontradiktif."""
    fallback_msg = ""
    if alert:
        if "Penurunan" in alert:
            if forecast_pct > 0.5:
                fallback_msg = f"Meskipun {alert.lower().replace('!', '')}, namun tetap waspada karena diprediksi harga {name} akan mulai merangkak naik kembali sekitar {forecast_pct}% minggu depan."
            else:
                fallback_msg = f"{alert} Ini saat yang tepat untuk stok barang, karena tren ke depan diprediksi masih akan melandai turun."
        elif "Lonjakan" in alert:
            if forecast_pct < -0.5:
                fallback_msg = f"{alert} Namun kabar baiknya, harga {name} diprediksi tidak akan lama tinggi dan segera melandai turun sekitar {abs(forecast_pct)}%."
            else:
                fallback_msg = f"{alert} Harap antisipasi pengeluaran lebih, karena tren menunjukkan harga {name} masih berpotensi naik."
        else:
            fallback_msg = f"{alert} Tetap pantau perkembangan harga {name} setiap hari."
    else:
        if "NAIK" in trend:
            fallback_msg = f"Tren {name} terpantau mulai merangkak naik. Prediksi menunjukkan kenaikan sekitar {abs(forecast_pct)}% minggu depan."
        elif "TURUN" in trend:
            fallback_msg = f"Kabar baik, tren {name} sedang menurun. Harga diprediksi turun sekitar {abs(forecast_pct)}% dalam waktu dekat."
        else:
            fallback_msg = f"Harga {name} terpantau stabil dan diperkirakan tidak banyak berubah dalam beberapa hari ke depan."

    fallback_msg += "\n\nSelalu pantau harga harian, karena faktor pasar sangat dinamis."

    if not openai_client:
        return fallback_msg

    try:
        prompt = f"""Kamu adalah analis pasar bahan pokok (berpengalaman namun gaya bahasa luwes/menarik).
Buatlah SATU ATAU DUA kalimat singkat yang menarik untuk disajikan di aplikasi mobile terkait komoditas {name}.
- Tren AI memprediksi: {trend}
- Prediksi perubahan harga: {forecast_pct}% minggu depan.
- Kondisi/Alert HARI INI: {alert if alert else 'Relatif stabil'}.

Syarat mutlak:
1. Jangan pakai poin-poin.
2. Gaya bahasa natural, asyik dibaca, tapi tetap profesional.
3. Jangan mengulang-ulang angka secara kaku, gabungkan ke dalam narasi.
4. Maksimal 40 kata."""
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100
        )
        if response.choices:
            return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"⚠️ Gagal menggunakan LLM untuk {name}: {e}")
        
    return fallback_msg

def generate_global_insight(summary_data):
    """Memberikan analisis pasar secara keseluruhan."""
    naik = [i['name'] for i in summary_data if "NAIK" in i['trend']]
    turun = [i['name'] for i in summary_data if "TURUN" in i['trend']]
    
    if len(naik) > 3:
        fallback_msg = "Pasar sedang mengalami tren kenaikan harga di beberapa sektor utama. "
    elif len(turun) > 3:
        fallback_msg = "Pasar menunjukkan tren penurunan harga yang cukup luas hari ini. "
    elif len(naik) > 0 or len(turun) > 0:
        fallback_msg = "Terdapat fluktuasi harga pada beberapa komoditas, namun pasar secara umum masih terkendali. "
    else:
        fallback_msg = "Pasar hari ini terpantau sangat stabil tanpa perubahan harga signifikan. "

    if naik:
        fallback_msg += f"Waspadai kenaikan pada {', '.join(naik)}. "
    if turun:
        fallback_msg += f"Potensi penghematan pada {', '.join(turun)}. "
    
    fallback_msg += "\nGunakan prediksi ini sebagai referensi pendukung, masih banyak faktor eksternal yang bisa mempengaruhi harga."

    if not openai_client:
        return fallback_msg

    try:
        prompt = f"""Kamu adalah pakar ekonomi. Buatlah rangkuman pasar bahan pokok hari ini dalam maksimal 3 kalimat padat dan menarik.
Komoditas yang diprediksi NAIK: {', '.join(naik) if naik else 'Tidak ada yang signifikan'}.
Komoditas yang diprediksi TURUN: {', '.join(turun) if turun else 'Tidak ada yang signifikan'}.

Tulis dengan gaya jurnalistik yang mengalir, seolah memberikan insight cepat untuk pemilik restoran atau ibu rumah tangga. 
Jangan gunakan sapaan halo. Jangan pakai poin-poin. Langsung ke intinya."""
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150
        )
        if response.choices:
            return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"⚠️ Gagal menggunakan LLM untuk global insight: {e}")
        
    return fallback_msg

def run_all_predictions():
    print("\n" + "═" * 60)
    print(" 🚀 STARTING MOBILE BACKEND GENERATOR")
    print(" 📱 Preparing data for Flutter Integration...")
    print("═" * 60 + "\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_full, _ = load_json_data(HISTORY_FILE)
    
    if USE_LIVE_API:
        print(" 🌐 Menghubungkan ke server BI (PIHPS) untuk update seluruh dataset...")
        df_full = update_history_with_api(df_full, HISTORY_FILE)
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump({"data": df_full.to_dict(orient="records")}, f, indent=2, ensure_ascii=False)
        print(" ✅ Dataset berhasil di-update ke tanggal terbaru.\n")
    
    success_count = 0
    fail_count = 0
    mobile_data = []
    audit_data = []
    
    # Mapping Satuan Komoditas
    commodity_units = {
        "Beras": "kg",
        "Daging Ayam": "kg",
        "Daging Sapi": "kg",
        "Telur Ayam": "kg",
        "Bawang Merah": "kg",
        "Bawang Putih": "kg",
        "Cabai Merah": "kg",
        "Cabai Rawit": "kg",
        "Minyak Goreng": "lt",
        "Gula Pasir": "kg"
    }

    for i, commodity in enumerate(COMMODITIES, 1):
        print(f"\n[{i}/{len(COMMODITIES)}] PROCESSING: {commodity.upper()}")
        
        try:
            # 1. Run Forecast (Sekarang ambil 5 return value: df, mape, winner, scores, all_fc)
            f_df, mape, winner, all_scores, all_fc = run_pipeline(
                json_path=HISTORY_FILE,
                commodity_name=commodity,
                n_days=FORECAST_DAYS,
                out_dir=OUTPUT_DIR,
                use_api=False # API sudah diupdate di awal sebelum loop
            )
            
            # 2. Get Historical Series for Chart & Current Price
            date_cols = [c for c in df_full.columns if "/" in c]
            
            # Buat rentang tanggal lengkap (termasuk Sabtu-Minggu) untuk 90 hari terakhir
            end_dt = datetime.strptime(max(date_cols, key=lambda x: datetime.strptime(x, "%d/%m/%Y")), "%d/%m/%Y")
            start_dt = end_dt - timedelta(days=90)
            
            # Buat list tanggal harian (Daily)
            full_date_range = []
            curr = start_dt
            while curr <= end_dt:
                full_date_range.append(curr.strftime("%d/%m/%Y"))
                curr += timedelta(days=1)
            
            series_hist = extract_commodity_series(df_full, commodity)
            # Reindex dengan kalender lengkap agar grafik linear di Flutter
            series_hist = series_hist.reindex(full_date_range)
            
            # ISI DATA KOSONG (NaN) dengan harga sebelumnya agar grafik menyambung
            series_hist = series_hist.ffill().bfill()
            
            last_actual = float(series_hist.iloc[-1])
            last_date = series_hist.index[-1]
            print(f"📊 Harga terakhir ({last_date}): Rp {last_actual:,.0f}")
            
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
            history_points = []
            # Kirim hanya 30 hari terakhir untuk efisiensi Mobile (1D, 7D, 30D)
            for d, v in series_hist.tail(30).items():
                d_obj = datetime.strptime(d, "%d/%m/%Y")
                history_points.append({
                    "date": d_obj.strftime("%Y-%m-%d"),
                    "price": float(v)
                })
            
            # Forecast diawali dengan harga terakhir hari ini agar "Tersambung" di grafik
            forecast_points = [{
                "date": datetime.strptime(last_date, "%d/%m/%Y").strftime("%Y-%m-%d"),
                "price": int(round(last_actual))
            }]
            # Tambahkan hasil prediksi besok dan seterusnya (Bulatkan ke Rupiah)
            for _, r in f_df.iterrows():
                forecast_points.append({
                    "date": r["date"],
                    "price": int(round(r["price"]))
                })

            # 5. Build Object
            # Hitung Perubahan Berbagai Rentang Waktu
            def get_pct_change(series, days_back):
                if len(series) > days_back:
                    current = float(series.iloc[-1])
                    past = float(series.iloc[-(days_back + 1)])
                    return round(((current - past) / past) * 100, 2)
                return 0.0

            changes = {
                "day_1": get_pct_change(series_hist, 1),
                "day_7": get_pct_change(series_hist, 7),
                "day_30": get_pct_change(series_hist, 30)
            }

            # Prediksi untuk penentuan Trend Masa Depan
            pred_end = float(f_df["price"].iloc[-1])
            forecast_change_pct = round(((pred_end - last_actual) / last_actual) * 100, 2)
            if forecast_change_pct == -0.0: forecast_change_pct = 0.0

            # Tingkat Keandalan (Reliability) berdasarkan MAPE
            if mape < 2.0:
                reliability = "SANGAT TINGGI"
            elif mape < 5.0:
                reliability = "TINGGI"
            elif mape < 15.0:
                reliability = "CUKUP"
            else:
                reliability = "RENDAH (PERINGATAN)"
            
            # Trend murni untuk Masa Depan (Ramalan)
            if forecast_change_pct > 0.5:
                trend = "NAIK 📈"
            elif forecast_change_pct < -0.5:
                trend = "TURUN 📉"
            else:
                trend = "STABIL ➡️"

            # Alert untuk Lonjakan/Penurunan Tajam HARI INI
            daily_pct = changes["day_1"]
            market_alert = None
            if daily_pct > 3.0:
                market_alert = "🚨 Lonjakan harga tajam hari ini!"
            elif daily_pct < -3.0:
                market_alert = "Penurunan harga signifikan hari ini!"
            
            slug = commodity.lower().replace(" ", "_")

            item_data = {
                "name": commodity,
                "current_price": last_actual,
                "unit": commodity_units.get(commodity, "kg"),
                "price_changes": changes,
                "forecast_pct": forecast_change_pct,
                "trend": trend,
                "reliability": reliability,
                "market_alert": market_alert,
                "image_asset": f"assets/images/{slug}.png",
                "insight": generate_commodity_insight(commodity, trend, round(forecast_change_pct, 2), market_alert),
                "chart": {
                    "history": history_points,
                    "forecast": forecast_points
                },
                "sub_commodities": sub_items
            }
            mobile_data.append(item_data)

            # Data untuk Audit Log (Teknis & Super Lengkap)
            audit_data.append({
                "commodity": commodity,
                "winner_today": winner,
                "winner_mape_score": round(float(mape), 2),
                "all_model_mape": {k: round(float(v), 2) for k, v in all_scores.items()},
                "forecast_comparison": {
                    "dates": [item["date"] for item in f_df.to_dict(orient="records")],
                    "arima": [float(p) for p in all_fc.get("arima", [])],
                    "ets": [float(p) for p in all_fc.get("ets", [])],
                    "prophet": [float(p) for p in all_fc.get("prophet", [])],
                    "xgboost": [float(p) for p in all_fc.get("xgboost", [])]
                }
            })

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
                "developer": "Rizky Hamdana",
                "description": "Sistem prediksi harga bahan pokok menggunakan AI dan Machine Learning (ARIMA/ETS/PROPHET/XGBOOST)."
            }
        },
        "commodities": mobile_data
    }

    mobile_path = os.path.join(OUTPUT_DIR, "mobile_backend.json")
    with open(mobile_path, "w", encoding="utf-8") as f:
        json.dump(mobile_backend, f, indent=2, ensure_ascii=False)

    # Technical Audit Log (Format khusus evaluasi)
    archive_dir = "archive"
    if not os.path.exists(archive_dir): os.makedirs(archive_dir)
    today_str = datetime.now().strftime("%Y-%m-%d")
    audit_path = os.path.join(archive_dir, f"audit_{today_str}.json")
    
    audit_log = {
        "execution_date": today_str,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "commodities_evaluated": success_count,
        "audit_results": audit_data
    }
    
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit_log, f, indent=2, ensure_ascii=False)

    print("\n" + "═" * 60)
    print(f" ✅ Success : {success_count} Commodities")
    print(f" 📱 Mobile Backend Ready at: {mobile_path}")
    print(f" 📂 Technical Audit Log at: {audit_path}")
    print(" 💡 Bandingkan isi Audit Log ini dengan harga asli minggu depan!")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    run_all_predictions()
