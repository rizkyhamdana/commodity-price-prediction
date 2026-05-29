from datetime import timedelta

import pandas as pd

from commodity_prediction.infrastructure.ml.exogenous_fetcher import get_live_rainfall_history, get_live_inflation_rate, get_live_exchange_rate


def prepare_ml_features(series: pd.Series) -> pd.DataFrame:
    """Mempersiapkan fitur teknis dengan data iklim/inflasi/kurs riil untuk XGBoost."""
    df = series.to_frame(name="y")
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df["day_of_month"] = df.index.day
    df["is_weekend"] = df.index.dayofweek.isin([5, 6]).astype(int)

    try:
        import holidays

        id_holidays = holidays.Indonesia()
        df["is_holiday"] = df.index.map(lambda x: 1 if x in id_holidays else 0)
        df["holiday_proximity"] = 0
        for i in range(1, 8):
            df["holiday_proximity"] += df.index.map(lambda x: 1 if (x + timedelta(days=i)) in id_holidays else 0)
    except Exception:
        df["is_holiday"] = 0
        df["holiday_proximity"] = 0

    # PURE PYTHON DYNAMIC HIJRI CALENDAR CONVERTER FOR DETECTING ISLAMIC SHOCKS (RAMADHAN & LEBARAN)
    # Ini sangat penting di Indonesia karena harga Daging Sapi, Ayam, Cabai melambung ekstrim pada siklus ini.
    def get_hijri_features(date_obj):
        # Algoritma konversi sederhana & handal Kuveyt Turk / Astronomis untuk kalender Hijriah
        jd = date_obj.toordinal() + 1721426
        l = jd - 1948440 + 10632
        n = int((l - 1) / 10631)
        l = l - 10631 * n + 354
        j = int((10985 - l) / 5316) * int((50 + l) / 135) + int(l / 30) * int((700 - l) / 909)
        l = l - int((700 - j) / 909) * int((30 * j + 59) / 30) + 30
        h_month = int((24 * l + 30) / 709)
        h_day = l - int((709 * h_month + 24) / 24)
        h_year = 30 * n + j - 30
        
        # Ramadhan adalah bulan ke-9 Hijriah, Syawal (Lebaran) adalah bulan ke-10 Hijriah
        is_ram = 1 if h_month == 9 else 0
        
        # Jarak ke Lebaran (1 Syawal / Idul Fitri)
        # Jika berada di bulan Ramadhan (bulan 9), hari Syawal tinggal (30 - h_day) hari lagi
        dist_to_eid = 99
        if h_month == 9:
            dist_to_eid = max(0, 30 - h_day)
        elif h_month == 10 and h_day <= 5: # Momen lebaran s/d H+5
            dist_to_eid = 0
            
        return is_ram, dist_to_eid

    hijri_data = df.index.map(get_hijri_features)
    df["is_ramadhan"] = [x[0] for x in hijri_data]
    df["days_to_eid"] = [x[1] for x in hijri_data]

    # INTEGRASI LIVE API FAKTOR EKSTERNAL (EKSOGEN)
    # 1. Curah Hujan Riil Kediri (milimeter harian dari Open-Meteo API)
    try:
        start_dt = df.index[0].to_pydatetime()
        end_dt = df.index[-1].to_pydatetime()
        rain_map = get_live_rainfall_history(start_dt, end_dt)
        df["curah_hujan"] = df.index.map(lambda x: rain_map.get(x.strftime("%Y-%m-%d"), 0.0))
    except Exception:
        # Fallback musiman
        df["curah_hujan"] = df["month"].map(lambda m: 250.0 if m in [11, 12, 1, 2, 3, 4] else 80.0)

    # 2. Nilai Tukar Rupiah Riil (USD/IDR harian dari Yahoo Finance API)
    try:
        start_dt = df.index[0].to_pydatetime()
        end_dt = df.index[-1].to_pydatetime()
        exc_map = get_live_exchange_rate(start_dt, end_dt)
        df["kurs_usd"] = df.index.map(lambda x: exc_map.get(x.strftime("%Y-%m-%d"), 16200.0))
    except Exception:
        df["kurs_usd"] = 16200.0

    # 3. Tingkat Inflasi Pangan Bulanan Nasional Riil (dari BPS)
    df["inflasi"] = get_live_inflation_rate()

    df["lag_1"] = df["y"].shift(1)
    df["lag_2"] = df["y"].shift(2)
    df["price_diff_1"] = df["lag_1"] - df["lag_2"]
    df["lag_7"] = df["y"].shift(7)
    df["lag_14"] = df["y"].shift(14)
    df["rolling_mean_7"] = df["y"].shift(1).rolling(window=7).mean()
    df["rolling_std_7"] = df["y"].shift(1).rolling(window=7).std()
    return df.dropna()
