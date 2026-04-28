import json
import os
import sys
import warnings
import logging
import requests
from datetime import datetime

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pmdarima import auto_arima

# Suppress warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
#  Logging Configuration
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════
#  SECTION 1: DATA LOADING
# ══════════════════════════════════════════════

def load_json_data(filepath: str) -> tuple[pd.DataFrame, str]:
    """Load data dari file lokal."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File tidak ditemukan: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        raw = json.load(f)
    
    if isinstance(raw, dict) and "data" in raw:
        df = pd.DataFrame(raw["data"])
        return df, "dataframe"
    return pd.DataFrame(raw), "dataframe"


def update_history_with_api(df: pd.DataFrame, history_path: str, sim_date: str = None) -> pd.DataFrame:
    """
    Ambil semua data yang hilang dari API BI (PIHPS) secara otomatis.
    Menarik dari tanggal terakhir di lokal sampai hari ini (atau sim_date).
    """
    now_obj = datetime.strptime(sim_date, "%Y-%m-%d") if sim_date else datetime.now()
    
    # Cari tanggal terakhir di history (kolom yang formatnya DD/MM/YYYY)
    date_cols = []
    for col in df.columns:
        try:
            date_cols.append(datetime.strptime(col, "%d/%m/%Y"))
        except: continue
    
    if not date_cols:
        logger.error("❌ Tidak dapat menemukan kolom tanggal di history.")
        return df
        
    last_date = max(date_cols)
    start_date_obj = last_date + pd.Timedelta(days=1)
    
    if start_date_obj > now_obj:
        logger.info("✅ Data history sudah up-to-date.")
        return df

    start_str = start_date_obj.strftime("%Y-%m-%d")
    end_str   = now_obj.strftime("%Y-%m-%d")

    # Generate URL BI PIHPS dengan Date Range
    api_url = (
        f"https://www.bi.go.id/hargapangan/WebSite/TabelHarga/GetGridDataDaerah?"
        f"price_type_id=1&comcat_id=&province_id=&regency_id=&market_id=&tipe_laporan=1"
        f"&start_date={start_str}&end_date={end_str}&_={int(now_obj.timestamp()*1000)}"
    )

    logger.info(f"🌐 Menarik data range: {start_str} s/d {end_str}")
    try:
        response = requests.get(api_url, timeout=20, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        raw_api = response.json()
        
        if "data" not in raw_api or not raw_api["data"]:
            logger.warning("⚠️ API BI tidak mengembalikan data untuk rentang ini.")
            return df

        new_entries = raw_api["data"]
        # Ambil semua kolom tanggal yang ada di hasil API (format DD/MM/YYYY)
        api_cols = [c for c in new_entries[0].keys() if "/" in c]
        
        updated = False
        for entry in new_entries:
            name = entry.get("name")
            if not name: continue
            
            mask = df["name"].astype(str).str.strip().str.lower() == name.strip().lower()
            if mask.any():
                for col in api_cols:
                    price = entry.get(col)
                    if price:
                        df.loc[mask, col] = str(price)
                        updated = True

        if updated:
            raw_to_save = {"data": df.to_dict(orient="records")}
            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(raw_to_save, f, indent=2, ensure_ascii=False)
            logger.info(f"✅ History diperbarui dengan {len(api_cols)} kolom tanggal baru.")
            
    except Exception as e:
        logger.warning(f"⚠️ Gagal update range dari API BI: {e}")
    
    return df


def extract_commodity_series(df: pd.DataFrame, commodity_name: str, level_filter: int = 1) -> pd.Series:
    NON_DATE_COLS = {"no", "name", "level"}
    if level_filter is not None and "level" in df.columns:
        df = df[df["level"] == level_filter]
    
    mask = df["name"].astype(str).str.strip().str.lower() == commodity_name.strip().lower()
    if mask.sum() == 0:
        raise ValueError(f"Komoditas '{commodity_name}' tidak ditemukan.")
    
    row = df[mask].iloc[0]
    price_dict = {}
    for col in df.columns:
        if col.lower() not in NON_DATE_COLS:
            try:
                date = pd.to_datetime(col, dayfirst=True)
                val = str(row[col]).replace(",", "").strip()
                if val not in ["", "nan", "-", "None"]:
                    price_dict[date] = float(val)
            except: pass
    
    series = pd.Series(price_dict).sort_index()
    return series


def handle_missing_dates(series: pd.Series, freq: str = "B") -> pd.Series:
    full_index = pd.date_range(start=series.index.min(), end=series.index.max(), freq=freq)
    series_full = series.reindex(full_index).ffill()
    return series_full


# ══════════════════════════════════════════════
#  SECTION 2: MODELING (ARIMA & ETS)
# ══════════════════════════════════════════════

def find_best_arima_params(series: pd.Series):
    """Mencari parameter ARIMA terbaik secara otomatis."""
    series_log = np.log1p(series)
    # Jangan paksa trend='t' agar tidak bentrok dengan nilai d (differencing)
    model = auto_arima(
        series_log, 
        seasonal=False, 
        stepwise=True, 
        suppress_warnings=True, 
        error_action="ignore",
        n_jobs=-1
    )
    return {
        "p": model.order[0], 
        "d": model.order[1], 
        "q": model.order[2], 
        "trend": model.trend
    }


def backtest_model(series: pd.Series, arima_order: dict, test_days: int, model_type: str):
    n = len(series)
    train = series.iloc[: n - test_days]
    test  = series.iloc[n - test_days :]
    train_log = np.log1p(train)

    if model_type == "arima":
        model = ARIMA(train_log, order=(arima_order["p"], arima_order["d"], arima_order["q"]), trend=arima_order.get("trend"))
        fitted = model.fit()
        pred = np.expm1(fitted.get_forecast(steps=test_days).predicted_mean)
    else:
        model = ExponentialSmoothing(train_log, trend='add', damped_trend=True)
        fitted = model.fit()
        pred = np.expm1(fitted.forecast(steps=test_days))
    
    pred.index = test.index
    mape = float((np.abs((test - pred) / test) * 100).mean())
    return {"mape": mape, "pred": pred, "test": test}


def train_final_model(series: pd.Series, arima_order: dict, model_type: str):
    """
    Melatih model akhir. Jika ARIMA gagal karena masalah parameter, 
    otomatis fallback ke ETS.
    """
    series_log = np.log1p(series)
    
    if model_type == "arima":
        try:
            model = ARIMA(
                series_log, 
                order=(arima_order["p"], arima_order["d"], arima_order["q"]), 
                trend=arima_order.get("trend")
            )
            return model.fit(), "arima"
        except Exception as e:
            logger.warning(f"⚠️ ARIMA Gagal ({e}), beralih ke ETS...")
            model = ExponentialSmoothing(series_log, trend='add', damped_trend=True)
            return model.fit(), "ets"
    else:
        model = ExponentialSmoothing(series_log, trend='add', damped_trend=True)
        return model.fit(), "ets"


def forecast_prices(fitted_model, series: pd.Series, n_days: int, model_type: str):
    """Melakukan prediksi berdasarkan model yang sudah dilatih."""
    if model_type == "arima":
        forecast_res = fitted_model.get_forecast(steps=n_days)
        mean = np.expm1(forecast_res.predicted_mean)
        conf = np.expm1(forecast_res.conf_int())
    else:
        # Untuk model ETS atau fallback
        mean = np.expm1(fitted_model.forecast(steps=n_days))
        # Estimasi confidence interval sederhana untuk ETS
        conf = pd.DataFrame({
            "lower": mean * 0.98, 
            "upper": mean * 1.02
        }, index=mean.index)
    
    dates = pd.bdate_range(start=series.index[-1] + pd.Timedelta(days=1), periods=n_days)
    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "predicted_price": mean.values.round(2),
        "lower_ci": conf.iloc[:, 0].values.round(2),
        "upper_ci": conf.iloc[:, 1].values.round(2)
    })
    return df


def save_forecast_to_json(forecast_df: pd.DataFrame, output_path: str, commodity_name: str):
    """
    Simpan hasil prediksi ke file JSON.
    """
    predictions = forecast_df[["date", "predicted_price"]].to_dict(orient="records")
    output = {
        "commodity": commodity_name,
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "forecast_days": len(forecast_df),
        "predictions": predictions
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info(f"💾 Prediksi disimpan ke: {output_path}")


# ══════════════════════════════════════════════
#  SECTION 3: VISUALIZATION
# ══════════════════════════════════════════════

def plot_final(series: pd.Series, forecast_df: pd.DataFrame, commodity: str, path: str, bt_res: dict):
    series_plot = series.tail(60)
    f_dates = pd.to_datetime(forecast_df["date"])
    
    fig, ax = plt.subplots(figsize=(14, 7), facecolor="#0f172a")
    ax.set_facecolor("#1e293b")
    ax.grid(color="#334155", linestyle="--", alpha=0.5)
    
    ax.plot(series_plot.index, series_plot.values, color="#38bdf8", lw=2, label="Historis")
    ax.plot(f_dates, forecast_df["predicted_price"], color="#f59e0b", lw=2.5, ls="--", marker="o", label="Prediksi")
    ax.fill_between(f_dates, forecast_df["lower_ci"], forecast_df["upper_ci"], color="#f59e0b", alpha=0.1)
    
    if bt_res:
        ax.plot(bt_res["pred"].index, bt_res["pred"].values, color="#a78bfa", ls=":", label="Backtest")

    ax.set_title(f"Prediksi Harga {commodity}", color="white", fontsize=14, pad=20)
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ══════════════════════════════════════════════
#  MAIN RUNNER
# ══════════════════════════════════════════════

def run_pipeline(json_path, commodity_name, n_days, out_dir, use_api=True, sim_date=None):
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Load History Lokal
    df, _ = load_json_data(json_path)
    
    # 2. Update dengan API BI PIHPS
    if use_api:
        df = update_history_with_api(df, json_path, sim_date=sim_date)
    
    series = extract_commodity_series(df, commodity_name)
    series = handle_missing_dates(series)
    
    # WINDOWING: Ambil 90 hari terakhir
    if len(series) > 90:
        series = series.tail(90)
    
    # Model Selection
    order = find_best_arima_params(series)
    test_days = 10
    bt_arima = backtest_model(series, order, test_days, "arima")
    bt_ets   = backtest_model(series, order, test_days, "ets")
    
    best_type = "ets" if bt_ets["mape"] < bt_arima["mape"] else "arima"
    best_bt = bt_ets if best_type == "ets" else bt_arima
    
    print(f"\n🏆 Model Terpilih: {best_type.upper()}")
    print(f"📊 Backtest MAPE: {best_bt['mape']:.2f}%")
    
    # Final Training
    final_model, actual_model_type = train_final_model(series, order, best_type)
    forecast_df = forecast_prices(final_model, series, n_days, actual_model_type)
    
    # Output
    slug = commodity_name.lower().replace(" ", "_")
    json_path = os.path.join(out_dir, f"forecast_{slug}.json")
    chart_path = os.path.join(out_dir, f"forecast_{slug}.png")
    
    # Simpan JSON dan Gambar
    save_forecast_to_json(forecast_df, json_path, commodity_name)
    plot_final(series, forecast_df, commodity_name, chart_path, best_bt)
    
    print("\n🔮 Hasil Prediksi:")
    for _, r in forecast_df.iterrows():
        print(f"   {r['date']} → Rp {r['predicted_price']:,.0f}")
        
    return forecast_df

if __name__ == "__main__":
    # ─── Konfigurasi ───────────────────────────────────────
    HISTORY_FILE   = "Data lengkap Komoditas 6 Bulan.json"
    USE_LIVE_API   = True  
    
    # Masukkan tanggal (YYYY-MM-DD) untuk simulasi update
    # Set ke None untuk mode normal (tanggal hari ini)
    SIM_DATE       = None 
    
    COMMODITY      = "Cabai Rawit" 
    FORECAST_DAYS  = 7
    OUTPUT_DIR     = "output"
    # ───────────────────────────────────────────────────────
    
    run_pipeline(HISTORY_FILE, COMMODITY, FORECAST_DAYS, OUTPUT_DIR, 
                 use_api=USE_LIVE_API, sim_date=SIM_DATE)
