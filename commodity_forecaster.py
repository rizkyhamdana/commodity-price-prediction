import os
import json
import requests
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import warnings

# Bungkam semua peringatan teknis agar log bersih
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Impor Model dengan penanganan error yang benar
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    HAS_STATSMODELS = True
except Exception as e:
    logger.warning(f"⚠️ Statsmodels error: {e}")
    HAS_STATSMODELS = False

try:
    from pmdarima import auto_arima
    HAS_PMDARIMA = True
except Exception:
    HAS_PMDARIMA = False

try:
    from prophet import Prophet
    HAS_PROPHET = True
except Exception:
    HAS_PROPHET = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except Exception as e:
    logger.warning(f"⚠️ XGBoost error: {e}")
    HAS_XGBOOST = False

def load_json_data(filepath: str) -> tuple[pd.DataFrame, str]:
    """Membaca data riwayat harga dari JSON lokal."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File tidak ditemukan: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict) and "data" in raw:
        df = pd.DataFrame(raw["data"])
    else:
        df = pd.DataFrame(raw)
    return df, "dataframe"

def update_history_with_api(df: pd.DataFrame, history_path: str, sim_date: str = None, force_start_date: str = None) -> pd.DataFrame:
    """Mengupdate data sejarah menggunakan API Bank Indonesia (PIHPS) dengan chunking 180 hari."""
    now_obj = datetime.strptime(sim_date, "%Y-%m-%d") if sim_date else datetime.now()
    
    if force_start_date:
        start_date_obj = datetime.strptime(force_start_date, "%Y-%m-%d")
        date_cols = []
        for col in df.columns:
            try: date_cols.append(datetime.strptime(col, "%d/%m/%Y"))
            except: continue
        end_date_obj = min(date_cols) - timedelta(days=1) if date_cols else now_obj
    else:
        date_cols = []
        for col in df.columns:
            try: date_cols.append(datetime.strptime(col, "%d/%m/%Y"))
            except: continue
        if not date_cols: return df
        # Mundur 7 hari dari tanggal terakhir untuk menangkap revisi/update data lama dari API
        start_date_obj = max(date_cols) - timedelta(days=7)
        end_date_obj = now_obj

    if start_date_obj > end_date_obj:
        logger.info("✅ Data history sudah up-to-date.")
        return df

    # Ambil kolom tanggal terakhir yang sudah ada sebagai referensi awal
    date_cols = [c for c in df.columns if "/" in c]
    if date_cols:
        last_date_col = sorted(date_cols, key=lambda x: datetime.strptime(x, "%d/%m/%Y"))[-1]
    else:
        last_date_col = None

    # Tambahkan kolom baru untuk semua tanggal dalam rentang agar dataset lengkap (Daily)
    all_dates_range = pd.date_range(start=start_date_obj, end=end_date_obj)
    for d_obj in all_dates_range:
        d_col = d_obj.strftime("%d/%m/%Y")
        if d_col not in df.columns:
            # Inisialisasi langsung dengan data terakhir agar tidak pernah kosong
            if last_date_col:
                df[d_col] = df[last_date_col]
            else:
                df[d_col] = np.nan

    current_start = start_date_obj
    actual_last_date_api = None
    
    while current_start <= end_date_obj:
        current_end = min(current_start + timedelta(days=180), end_date_obj)
        start_str = current_start.strftime("%Y-%m-%d")
        end_str   = current_end.strftime("%Y-%m-%d")
        
        api_url = (
            f"https://www.bi.go.id/hargapangan/WebSite/TabelHarga/GetGridDataDaerah?"
            f"price_type_id=1&comcat_id=&province_id=&regency_id=&market_id=&tipe_laporan=1"
            f"&start_date={start_str}&end_date={end_str}&_={int(datetime.now().timestamp()*1000)}"
        )
        logger.info(f"🌐 Menarik chunk data: {start_str} s/d {end_str}")
        try:
            response = requests.get(api_url, timeout=60, headers={'User-Agent': 'Mozilla/5.0'})
            response.raise_for_status()
            raw_api = response.json()
            if "data" in raw_api and raw_api["data"]:
                new_entries = raw_api["data"]
                api_date_cols = [c for c in new_entries[0].keys() if "/" in c]
                if api_date_cols:
                    # Track latest date actually returned by API
                    latest_api_col = sorted(api_date_cols, key=lambda x: datetime.strptime(x, "%d/%m/%Y"))[-1]
                    if actual_last_date_api is None or datetime.strptime(latest_api_col, "%d/%m/%Y") > datetime.strptime(actual_last_date_api, "%d/%m/%Y"):
                        actual_last_date_api = latest_api_col

                for entry in new_entries:
                    name_api = str(entry.get("name", "")).strip().lower()
                    mask = df["name"].astype(str).str.strip().str.lower() == name_api
                    if mask.any():
                        for d_col in api_date_cols:
                            val = entry[d_col]
                            # Jika data API kosong atau '-', anggap sebagai NaN agar bisa diisi harga sebelumnya
                            if val == "-" or val == "" or val is None:
                                df.loc[mask, d_col] = np.nan
                            else:
                                df.loc[mask, d_col] = val
        except Exception as e:
            logger.warning(f"⚠️ Gagal menarik data chunk {start_str}: {e}")
        current_start = current_end + timedelta(days=1)

    # --- LOGIKA GAP FILLING PERMANEN (Double Check) ---
    # Mengambil semua kolom tanggal dan mengurutkannya
    date_cols = [c for c in df.columns if "/" in c]
    date_cols_sorted = sorted(date_cols, key=lambda x: datetime.strptime(x, "%d/%m/%Y"))
    
    # Lakukan ffill (Forward Fill) untuk mengisi harga kosong dengan harga hari sebelumnya
    df[date_cols_sorted] = df[date_cols_sorted].ffill(axis=1)
    
    if actual_last_date_api:
        logger.info(f"✅ Sinkronisasi selesai: Data API asli ditemukan sampai {actual_last_date_api}. (Lookback 7 hari).")
    else:
        logger.info(f"✅ Sinkronisasi selesai: Tidak ada data baru di API, menggunakan harga terakhir (ffill).")
    
    return df

def extract_commodity_series(df: pd.DataFrame, commodity_name: str, level_filter: int = 1) -> pd.Series:
    """Mengambil deret waktu harga untuk komoditas spesifik dan membersihkannya."""
    NON_DATE_COLS = {"no", "name", "level"}
    if level_filter is not None and "level" in df.columns:
        df = df[df["level"] == level_filter]
    
    mask = df["name"].astype(str).str.strip().str.lower() == commodity_name.strip().lower()
    if mask.sum() == 0:
        raise ValueError(f"Komoditas '{commodity_name}' tidak ditemukan di dataset.")
    
    row = df[mask].iloc[0]
    price_dict = {}
    for col in df.columns:
        if col.lower() not in NON_DATE_COLS:
            try:
                val = str(row[col]).replace(",", "")
                if val and val != "nan" and val != "-":
                    price_dict[datetime.strptime(col, "%d/%m/%Y")] = float(val)
            except: continue
    
    if not price_dict:
        raise ValueError(f"Tidak ada data harga valid untuk {commodity_name}")
    
    s = pd.Series(price_dict).sort_index()
    # Cleaning: Isi bolong-bolong data dengan interpolasi linear agar grafik mulus
    full_idx = pd.date_range(start=s.index.min(), end=s.index.max(), freq='D')
    s = s.reindex(full_idx).interpolate(method='linear').ffill().bfill()
    return s

def prepare_ml_features(series):
    """Mempersiapkan fitur teknis untuk XGBoost."""
    df = series.to_frame(name='y')
    # Feature Engineering yang lebih kaya
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['day_of_month'] = df.index.day
    df['is_weekend'] = df.index.dayofweek.isin([5, 6]).astype(int)
    
    # Fitur Hari Libur Indonesia (Menggunakan library holidays)
    try:
        import holidays
        id_holidays = holidays.Indonesia()
        df['is_holiday'] = df.index.map(lambda x: 1 if x in id_holidays else 0)
        
        # Fitur Kedekatan Hari Raya (H-7 sangat krusial untuk Daging/Ayam)
        # Kita cek apakah ada hari libur dalam 7 hari ke depan
        df['holiday_proximity'] = 0
        for i in range(1, 8):
            df['holiday_proximity'] += df.index.map(lambda x: 1 if (x + timedelta(days=i)) in id_holidays else 0)
    except:
        # Fallback jika library gagal
        df['is_holiday'] = 0
        df['holiday_proximity'] = 0
    
    # Fitur Lag dan Window
    df['lag_1'] = df['y'].shift(1)
    df['lag_7'] = df['y'].shift(7)
    df['lag_14'] = df['y'].shift(14)
    df['rolling_mean_7'] = df['y'].shift(1).rolling(window=7).mean()
    df['rolling_std_7'] = df['y'].shift(1).rolling(window=7).std()
    return df.dropna()

def predict_xgboost(series, n_days):
    """Prediksi menggunakan XGBoost dengan strategi rekursif."""
    if not HAS_XGBOOST: 
        raise ValueError("XGBoost tidak terinstall")
    
    df_feat = prepare_ml_features(series)
    if len(df_feat) < 20: 
        raise ValueError("Data terlalu sedikit untuk dilatih dengan XGBoost (butuh min 20)")
    
    feat_cols = ['day_of_week', 'month', 'day_of_month', 'lag_1', 'lag_7', 'lag_14', 'rolling_mean_7', 'rolling_std_7']
    X = df_feat[feat_cols]
    y = df_feat['y']
    
    model = xgb.XGBRegressor(n_estimators=150, learning_rate=0.07, max_depth=5, objective='reg:squarederror')
    model.fit(X, y)
    
    history = series.copy()
    forecasts = []
    curr_date = history.index[-1]
    
    for _ in range(n_days):
        curr_date += timedelta(days=1)
        feat = pd.DataFrame([{
            'day_of_week': curr_date.weekday(),
            'month': curr_date.month,
            'day_of_month': curr_date.day,
            'lag_1': history.iloc[-1],
            'lag_7': history.iloc[-7] if len(history)>=7 else history.iloc[-1],
            'lag_14': history.iloc[-14] if len(history)>=14 else history.iloc[-1],
            'rolling_mean_7': history.iloc[-7:].mean(),
            'rolling_std_7': history.iloc[-7:].std() if len(history)>=7 else 0
        }])
        p = model.predict(feat[feat_cols])[0]
        forecasts.append(p)
        history = pd.concat([history, pd.Series([p], index=[curr_date])])
    return np.array(forecasts)

def backtest_model(series, test_days=30, model_type="arima"):
    """Menguji performa model pada 30 hari terakhir untuk menghitung MAPE."""
    if len(series) < test_days + 20: return {"mape": 99.9}
    train, test = series.iloc[:-test_days], series.iloc[-test_days:]
    try:
        if model_type == "arima" and HAS_STATSMODELS:
            from pmdarima import auto_arima
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                order = auto_arima(np.log1p(train), seasonal=False, m=1).order
                model = SARIMAX(np.log1p(train), order=order).fit(disp=False)
            fc = np.expm1(model.forecast(steps=test_days))
        elif model_type == "ets" and HAS_STATSMODELS:
            # ETS seringkali butuh parameter seasonal jika data harian
            model = ExponentialSmoothing(train, trend='add', seasonal=None).fit()
            fc = model.forecast(steps=test_days)
        elif model_type == "prophet" and HAS_PROPHET:
            df_p = train.reset_index(); df_p.columns = ['ds', 'y']; df_p['ds'] = df_p['ds'].dt.tz_localize(None)
            m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
            m.add_country_holidays(country_name='ID'); m.fit(df_p)
            fc = m.predict(m.make_future_dataframe(periods=test_days, freq='D'))['yhat'].iloc[-test_days:].values
        elif model_type == "xgboost" and HAS_XGBOOST:
            fc = predict_xgboost(train, test_days)
        else: return {"mape": 99.9}
        
        fc = fc[:len(test)]
        mape = np.mean(np.abs((test.values - fc) / test.values)) * 100
        return {"mape": mape, "forecast": fc}
    except Exception as e:
        logger.error(f"❌ Error pada model {model_type}: {e}")
        return {"mape": 99.9}

def plot_forecast(series, forecast_dates, forecast_values, commodity_name, out_path):
    """Menyimpan grafik prediksi statis (Opsional)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib diperlukan untuk membuat grafik prediksi. "
            "Install dependency dari requirements.txt sebelum menjalankan pipeline prediksi."
        ) from exc

    plt.figure(figsize=(12, 6))
    plt.plot(series.index[-60:], series.values[-60:], label="Riwayat (60 Hari)", color="blue", marker='o')
    plt.plot(forecast_dates, forecast_values, label="Prediksi AI", color="red", linestyle="--", marker='s')
    plt.title(f"Prediksi Harga: {commodity_name}")
    plt.xlabel("Tanggal")
    plt.ylabel("Harga (Rp)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(out_path)
    plt.close()

def run_pipeline(json_path, commodity_name, n_days=7, out_dir="output", use_api=True, df=None):
    """Orkestrasi utama: Ambil Data -> Kompetisi Model -> Ramalan -> Simpan JSON."""
    if df is None:
        df, _ = load_json_data(json_path)
        
    if use_api:
        df = update_history_with_api(df, json_path)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"data": df.to_dict(orient="records")}, f, indent=2, ensure_ascii=False)
    
    series = extract_commodity_series(df, commodity_name)
    
    # Kompetisi 4 Model (Quad-Hybrid)
    scores = {}
    for m in ["arima", "ets", "prophet", "xgboost"]:
        scores[m] = backtest_model(series, test_days=30, model_type=m)["mape"]
    
    valid_scores = {k: v for k, v in scores.items() if v < 99.0}
    if not valid_scores:
        logger.warning(f"⚠️ Semua model gagal untuk {commodity_name}. Menggunakan fallback.")
        best_type = "ets"
        best_mape = 99.9
    else:
        best_type = min(valid_scores, key=valid_scores.get)
        best_mape = valid_scores[best_type]
    
    print(f"   - ARIMA: {scores['arima']:.2f}% | ETS: {scores['ets']:.2f}% | PROPHET: {scores['prophet']:.2f}% | XGB: {scores['xgboost']:.2f}%")
    print(f"🏆 Pemenang: {best_type.upper()} ({best_mape:.2f}%)")
    
    # Final Forecast dengan SEMUA Model untuk Audit
    all_forecasts = {}
    forecast_dates = pd.date_range(start=series.index[-1], periods=n_days + 1, freq='D')[1:]
    forecast_dates_str = [d.strftime("%Y-%m-%d") for d in forecast_dates]

    # 1. ARIMA
    try:
        from pmdarima import auto_arima
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            order = auto_arima(np.log1p(series), seasonal=False, m=1).order
            model_arima = SARIMAX(np.log1p(series), order=order).fit(disp=False)
        fc_arima = np.expm1(model_arima.forecast(steps=n_days))
        all_forecasts["arima"] = [round(p, 0) for p in fc_arima]
    except: all_forecasts["arima"] = []

    # 2. ETS
    try:
        model_ets = ExponentialSmoothing(series, trend='add').fit()
        fc_ets = model_ets.forecast(steps=n_days)
        all_forecasts["ets"] = [round(p, 0) for p in fc_ets]
    except: all_forecasts["ets"] = []

    # 3. Prophet
    try:
        df_p = series.reset_index(); df_p.columns = ['ds', 'y']; df_p['ds'] = df_p['ds'].dt.tz_localize(None)
        m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
        m.add_country_holidays(country_name='ID'); m.fit(df_p)
        fc_prophet = m.predict(m.make_future_dataframe(periods=n_days, freq='D'))['yhat'].iloc[-n_days:].values
        all_forecasts["prophet"] = [round(p, 0) for p in fc_prophet]
    except: all_forecasts["prophet"] = []

    # 4. XGBoost
    try:
        fc_xgb = predict_xgboost(series, n_days)
        all_forecasts["xgboost"] = [round(p, 0) for p in fc_xgb]
    except: all_forecasts["xgboost"] = []

    # Hasil untuk Mobile (Tetap pakai yang terbaik berdasarkan MAPE)
    best_fc = all_forecasts[best_type]
    forecast_df = pd.DataFrame({
        "date": forecast_dates_str,
        "price": best_fc
    })
    
    # Simpan JSON Individual
    res = {
        "commodity": commodity_name,
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_used": best_type,
        "mape": round(best_mape, 2),
        "last_price": float(series.iloc[-1]),
        "forecast": forecast_df.to_dict(orient="records")
    }
    
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"forecast_{commodity_name.lower().replace(' ', '_')}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    
    # Simpan Grafik
    plot_path = os.path.join(out_dir, f"chart_{commodity_name.lower().replace(' ', '_')}.png")
    plot_forecast(series, forecast_dates, best_fc, commodity_name, plot_path)
    
    print(f"🔮 Hasil Prediksi ({best_type.upper()}):")
    for date, price in zip(forecast_dates_str[:3], best_fc[:3]):
        print(f"   {date} → Rp {price:,}")
    
    return forecast_df, best_mape, best_type, scores, all_forecasts
