"""Generate Flutter/mobile backend JSON from commodity forecasts."""

import json
import os
from datetime import datetime, timedelta

import pandas as pd

from commodity_prediction.config import COMMODITIES, COMMODITY_UNITS, FORECAST_DAYS, HISTORY_FILE, OUTPUT_DIR, USE_LIVE_API
from commodity_prediction.infrastructure.data import extract_commodity_series, load_json_data, update_history_with_api
from commodity_prediction.infrastructure.llm import generate_commodity_insight, generate_global_insight
from commodity_prediction.application.use_cases.forecast_commodity import run_pipeline


def _date_columns(df):
    return [c for c in df.columns if "/" in c]


def _pct_change(series, days_back):
    if len(series) > days_back:
        current = float(series.iloc[-1])
        past = float(series.iloc[-(days_back + 1)])
        return round(((current - past) / past) * 100, 2)
    return 0.0


def _reliability_label(mape):
    if mape < 2.0:
        return "SANGAT TINGGI"
    if mape < 5.0:
        return "TINGGI"
    if mape < 15.0:
        return "CUKUP"
    return "RENDAH (PERINGATAN)"


def _trend_label(forecast_change_pct):
    if forecast_change_pct > 0.5:
        return "NAIK 📈"
    if forecast_change_pct < -0.5:
        return "TURUN 📉"
    return "STABIL ➡️"


def _market_alert(daily_pct):
    if daily_pct > 3.0:
        return "🚨 Lonjakan harga tajam hari ini!"
    if daily_pct < -3.0:
        return "Penurunan harga signifikan hari ini!"
    return None


def _history_series(df_full, commodity):
    date_cols = _date_columns(df_full)
    end_dt = datetime.strptime(max(date_cols, key=lambda x: datetime.strptime(x, "%d/%m/%Y")), "%d/%m/%Y")
    start_dt = end_dt - timedelta(days=90)
    full_date_range = pd.date_range(start=start_dt, end=end_dt, freq="D")
    series_hist = extract_commodity_series(df_full, commodity)
    return series_hist.reindex(full_date_range).ffill().bfill()


def _sub_commodities(df_full, commodity):
    sub_items = []
    all_subs = df_full[df_full["level"] == 2]

    for _, sub_row in all_subs.iterrows():
        if commodity.lower() not in sub_row["name"].lower():
            continue

        prices_row = sub_row.drop(["no", "name", "level"], errors="ignore").dropna()
        try:
            p_idx = pd.to_datetime(prices_row.index, format="%d/%m/%Y")
            prices_series = pd.Series(prices_row.values, index=p_idx).sort_index()

            if len(prices_series) >= 2:
                last_p = float(str(prices_series.iloc[-1]).replace(",", ""))
                prev_p = float(str(prices_series.iloc[-2]).replace(",", ""))
                change_pct = ((last_p - prev_p) / prev_p * 100) if prev_p != 0 else 0
                sub_trend = "▲" if change_pct > 0 else ("▼" if change_pct < 0 else "—")
            else:
                last_p = float(str(prices_series.iloc[-1]).replace(",", "")) if not prices_series.empty else 0
                change_pct = 0
                sub_trend = "—"
        except Exception:
            last_p = 0
            change_pct = 0
            sub_trend = "—"

        sub_slug = sub_row["name"].lower().replace(" ", "_")
        sub_items.append(
            {
                "name": sub_row["name"],
                "price": last_p,
                "change_pct": round(change_pct, 2),
                "trend": sub_trend,
                "image_asset": f"assets/images/{sub_slug}.png",
            }
        )

    return sub_items


def _build_commodity_payload(df_full, commodity, forecast_df, mape):
    series_hist = _history_series(df_full, commodity)
    last_actual = float(series_hist.iloc[-1])
    last_date = series_hist.index[-1]
    print(f"📊 Harga terakhir ({last_date}): Rp {last_actual:,.0f}")

    history_points = [{"date": d.strftime("%Y-%m-%d"), "price": float(v)} for d, v in series_hist.tail(30).items()]
    forecast_points = [{"date": last_date.strftime("%Y-%m-%d"), "price": int(round(last_actual))}]
    forecast_points.extend(
        {"date": r["date"], "price": int(round(r["price"]))}
        for _, r in forecast_df.iterrows()
    )

    changes = {
        "day_1": _pct_change(series_hist, 1),
        "day_7": _pct_change(series_hist, 7),
        "day_30": _pct_change(series_hist, 30),
    }
    pred_end = float(forecast_df["price"].iloc[-1])
    forecast_change_pct = round(((pred_end - last_actual) / last_actual) * 100, 2)
    if forecast_change_pct == -0.0:
        forecast_change_pct = 0.0

    trend = _trend_label(forecast_change_pct)
    market_alert = _market_alert(changes["day_1"])
    slug = commodity.lower().replace(" ", "_")

    return {
        "name": commodity,
        "current_price": last_actual,
        "unit": COMMODITY_UNITS.get(commodity, "kg"),
        "price_changes": changes,
        "forecast_pct": forecast_change_pct,
        "trend": trend,
        "reliability": _reliability_label(mape),
        "market_alert": market_alert,
        "image_asset": f"assets/images/{slug}.png",
        "insight": generate_commodity_insight(commodity, trend, round(forecast_change_pct, 2), market_alert),
        "sub_commodities": _sub_commodities(df_full, commodity),
    }


def _process_single_commodity(commodity, df_full, weather_forecast=None):
    """Memproses satu komoditas secara terisolasi untuk mendukung parallel processing."""
    from commodity_prediction.config import FORECAST_DAYS, HISTORY_FILE, OUTPUT_DIR
    from commodity_prediction.application.use_cases.forecast_commodity import run_pipeline

    forecast_df, mape, winner, all_scores, all_forecasts = run_pipeline(
        json_path=HISTORY_FILE,
        commodity_name=commodity,
        n_days=FORECAST_DAYS,
        out_dir=OUTPUT_DIR,
        use_api=False,
        df=df_full,
        weather_forecast=weather_forecast,
    )

    mobile_payload = _build_commodity_payload(df_full, commodity, forecast_df, mape)
    audit_payload = {
        "commodity": commodity,
        "winner_today": winner,
        "winner_mape_score": round(float(mape), 2),
        "winner_combined_score": round(float(all_scores[winner]), 2) if winner in all_scores else None,
        "all_model_combined_scores": {k: round(float(v), 2) for k, v in all_scores.items()},
        "forecast_comparison": {
            "dates": [item["date"] for item in forecast_df.to_dict(orient="records")],
            "arima": [float(p) for p in all_forecasts.get("arima", [])],
            "ets": [float(p) for p in all_forecasts.get("ets", [])],
            "prophet": [float(p) for p in all_forecasts.get("prophet", [])],
            "xgboost": [float(p) for p in all_forecasts.get("xgboost", [])],
        },
    }
    return mobile_payload, audit_payload


def _write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def run_all_predictions():
    print("\n" + "═" * 60)
    print(" 🚀 STARTING PARALLEL MOBILE BACKEND GENERATOR (ALL CPU CORES)")
    print(" 📱 Preparing data for Flutter Integration...")
    print("═" * 60 + "\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_full, _ = load_json_data(HISTORY_FILE)

    if USE_LIVE_API:
        print(" 🌐 Menghubungkan ke server BI (PIHPS) untuk update seluruh dataset...")
        df_full = update_history_with_api(df_full, HISTORY_FILE)
        _write_json(HISTORY_FILE, {"data": df_full.to_dict(orient="records")})
        print(" ✅ Dataset berhasil di-update ke tanggal terbaru.\n")

    success_count = 0
    fail_count = 0
    
    # Gunakan dictionary untuk menyimpan hasil sementara agar bisa di-sorting kembali
    mobile_dict = {}
    audit_dict = {}

    # Ambil prakiraan cuaca live SEBELUM memulai paralel untuk menghindari bentrokan rate-limit paralel
    from commodity_prediction.infrastructure.ml.exogenous_fetcher import get_live_rainfall_forecast
    try:
        print("🌐 Menarik prakiraan cuaca Kediri (Open-Meteo) di awal...")
        # Tarik forecast cuaca untuk hari ke depan (biasanya FORECAST_DAYS)
        weather_forecast = get_live_rainfall_forecast(FORECAST_DAYS)
        print("✅ Berhasil mendapatkan prakiraan cuaca untuk proses forecasting.")
    except Exception as e:
        print(f"⚠️ Gagal menarik prakiraan cuaca: {e}. Menggunakan fallback.")
        weather_forecast = {}

    from concurrent.futures import ProcessPoolExecutor, as_completed

    # Menjalankan proses training secara paralel menggunakan seluruh core CPU yang tersedia
    max_cores = os.cpu_count() or 4
    print(f"🔥 Memulai pelatihan paralel menggunakan {max_cores} Core CPU sekaligus...")
    
    with ProcessPoolExecutor(max_workers=max_cores) as executor:
        futures = {
            executor.submit(_process_single_commodity, commodity, df_full, weather_forecast): commodity 
            for commodity in COMMODITIES
        }
        
        for future in as_completed(futures):
            commodity = futures[future]
            try:
                mobile_payload, audit_payload = future.result()
                mobile_dict[commodity] = mobile_payload
                audit_dict[commodity] = audit_payload
                print(f"✅ Selesai memproses: {commodity.upper()}")
                success_count += 1
            except Exception as e:
                print(f"❌ Gagal memproses {commodity.upper()}: {e}")
                fail_count += 1

    # SORTING KEMBALI HASH MAP SESUAI URUTAN ASLI DI VARIABEL COMMODITIES
    mobile_data = [mobile_dict[c] for c in COMMODITIES if c in mobile_dict]
    audit_data = [audit_dict[c] for c in COMMODITIES if c in audit_dict]

    mobile_backend = {
        "metadata": {
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "global_analysis": generate_global_insight(mobile_data),
            "disclaimer": (
                "Data ini adalah hasil prediksi model AI (Machine Learning). Gunakan hanya sebagai referensi tambahan. "
                "Harga riil di pasar dapat dipengaruhi oleh faktor eksternal mendadak yang tidak terekam dalam data historis."
            ),
            "about_us": {
                "app_name": "Komoditas-AI",
                "version": "1.0.0",
                "developer": "Rizky Hamdana",
                "description": "Sistem prediksi harga bahan pokok menggunakan AI dan Machine Learning (ARIMA/ETS/PROPHET/XGBOOST).",
            },
        },
        "commodities": mobile_data,
    }

    mobile_path = os.path.join(OUTPUT_DIR, "mobile_backend.json")
    _write_json(mobile_path, mobile_backend)

    archive_dir = "archive"
    os.makedirs(archive_dir, exist_ok=True)
    today_str = datetime.now().strftime("%Y-%m-%d")
    audit_path = os.path.join(archive_dir, f"audit_{today_str}.json")
    _write_json(
        audit_path,
        {
            "execution_date": today_str,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "commodities_evaluated": success_count,
            "audit_results": audit_data,
        },
    )

    print("\n" + "═" * 60)
    print(f" ✅ Success : {success_count} Commodities")
    print(f" 📱 Mobile Backend Ready at: {mobile_path}")
    print(f" 📂 Technical Audit Log at: {audit_path}")
    print(" 💡 Bandingkan isi Audit Log ini dengan harga asli minggu depan!")
    print("═" * 60 + "\n")
