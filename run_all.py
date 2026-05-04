import json
import os
import logging
import pandas as pd
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
    """Menghasilkan analisis dua perspektif (Masyarakat & Pedagang) menggunakan LLM."""
    
    # Fallback default values
    fallback_masyarakat = ""
    fallback_pedagang = ""
    
    # Logic for fallback Masyarakat
    if alert:
        if "Penurunan" in alert:
            fallback_masyarakat = f"{alert} Saatnya belanja stok bulanan selagi harga melandai."
        elif "Lonjakan" in alert:
            fallback_masyarakat = f"{alert} Sebaiknya tunda pembelian besar atau cari alternatif komoditas lain."
        else:
            fallback_masyarakat = f"{alert} Pantau harga harian untuk mendapatkan waktu beli terbaik."
    else:
        if "NAIK" in trend:
            fallback_masyarakat = f"Tren {name} diprediksi naik {abs(forecast_pct)}%. Segera stok secukupnya sebelum harga makin mahal."
        elif "TURUN" in trend:
            fallback_masyarakat = f"Kabar baik, {name} sedang turun. Tunggu beberapa hari lagi untuk harga terendah."
        else:
            fallback_masyarakat = f"Harga {name} stabil. Beli sesuai kebutuhan harian Anda."

    # Logic for fallback Pedagang
    if "NAIK" in trend:
        fallback_pedagang = f"Potensi margin meningkat karena tren naik. Pastikan stok aman untuk memenuhi permintaan."
    elif "TURUN" in trend:
        fallback_pedagang = f"Hati-hati stok menumpuk saat harga turun. Percepat perputaran stok agar tidak rugi."
    else:
        fallback_pedagang = f"Pasar stabil. Fokus pada efisiensi operasional dan kualitas produk."

    disclaimer_text = "Analisis ini dihasilkan secara otomatis oleh AI dan bersifat saran referensi. Keputusan ekonomi tetap berada di tangan pengguna."

    if not openai_client:
        return {
            "masyarakat": fallback_masyarakat, 
            "pedagang": fallback_pedagang,
            "disclaimer": disclaimer_text
        }

    try:
        prompt = f"""Kamu adalah analis pasar bahan pokok profesional.
Buatlah analisis singkat untuk komoditas {name} dari DUA PERSPEKTIF berbeda:
1. MASYARAKAT (Pembeli/Konsumen): Berikan saran strategis apakah harus membeli sekarang, stok barang, atau tunda pembelian.
2. PEDAGANG (Penjual): Berikan saran bisnis terkait manajemen stok, strategi harga, atau potensi keuntungan.

Data pendukung (WAJIB DIIKUTI):
- Tren AI Masa Depan: {trend} (Ini adalah prediksi utama, jangan dilawan)
- Prediksi perubahan harga: {forecast_pct}% minggu depan.
- Kondisi HARI INI: {alert if alert else 'Relatif stabil'}.
- Pertimbangkan kondisi hari ini(wajib)

Instruksi Khusus:
- Jika tren {trend} adalah NAIK, jangan menyarankan tunda pembelian dengan alasan harga akan turun.
- Jika tren {trend} adalah TURUN, jangan menyarankan beli sekarang dengan alasan harga akan naik.
- Pastikan insight MASYARAKAT dan PEDAGANG konsisten dengan data Tren AI Masa Depan di atas.

Syarat:
1. Gunakan bahasa Indonesia yang santai tapi profesional.
2. Maksimal 30 kata per perspektif.
3. JANGAN gunakan poin-poin.
4. Output WAJIB dalam format JSON murni:
{{"masyarakat": "...", "pedagang": "..."}}"""

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_tokens=200
        )
        if response.choices:
            content = response.choices[0].message.content.strip()
            result = json.loads(content)
            result["disclaimer"] = disclaimer_text
            return result
    except Exception as e:
        print(f"⚠️ Gagal menggunakan LLM untuk {name}: {e}")
        
    return {
        "masyarakat": fallback_masyarakat, 
        "pedagang": fallback_pedagang,
        "disclaimer": disclaimer_text
    }

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
            # Konversi full_date_range ke DatetimeIndex agar match dengan series_hist.index
            series_hist = series_hist.reindex(pd.to_datetime(full_date_range, dayfirst=True))
            
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
                    # Ambil hanya kolom tanggal
                    prices_row = sub_row.drop(["no", "name", "level"], errors="ignore").dropna()
                    
                    # Buat series dengan index datetime agar bisa diurutkan
                    try:
                        p_idx = pd.to_datetime(prices_row.index, format="%d/%m/%Y")
                        prices_series = pd.Series(prices_row.values, index=p_idx).sort_index()
                        
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
                    except:
                        # Fallback jika ada format tanggal aneh
                        last_p = 0
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
                # d sekarang adalah Timestamp karena index sudah DatetimeIndex
                history_points.append({
                    "date": d.strftime("%Y-%m-%d"),
                    "price": float(v)
                })
            
            # Forecast diawali dengan harga terakhir hari ini agar "Tersambung" di grafik
            forecast_points = [{
                "date": last_date.strftime("%Y-%m-%d"),
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
            "disclaimer": "Data ini adalah hasil prediksi model AI (Machine Learning). Gunakan hanya sebagai referensi tambahan. Harga riil di pasar dapat dipengaruhi oleh faktor eksternal mendadak yang tidak terekam dalam data historis.",
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
