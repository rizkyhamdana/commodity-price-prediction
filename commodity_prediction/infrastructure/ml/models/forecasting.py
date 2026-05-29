"""Forecast generation for each supported algorithm."""

from datetime import timedelta
import warnings

import numpy as np
import pandas as pd

from commodity_prediction.infrastructure.ml.features import prepare_ml_features
from commodity_prediction.infrastructure.ml.exogenous_fetcher import get_live_rainfall_forecast, get_live_inflation_rate, get_live_exchange_rate
from commodity_prediction.infrastructure.ml.dependencies import (
    HAS_PROPHET,
    HAS_STATSMODELS,
    HAS_XGBOOST,
    ExponentialSmoothing,
    Prophet,
    SARIMAX,
    auto_arima,
    xgb,
)


def predict_arima(series: pd.Series, n_days: int) -> np.ndarray:
    if not HAS_STATSMODELS or auto_arima is None:
        raise ValueError("ARIMA dependencies tidak terinstall")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Mengaktifkan seasonality mingguan (m=7) untuk menangkap hari pasar komoditas
        order = auto_arima(np.log1p(series), seasonal=True, m=7).order
        model = SARIMAX(np.log1p(series), order=order).fit(disp=False)
    return np.expm1(model.forecast(steps=n_days))


def predict_ets(series: pd.Series, n_days: int) -> np.ndarray:
    if not HAS_STATSMODELS:
        raise ValueError("Statsmodels tidak terinstall")

    # Mengaktifkan seasonal aditif mingguan (seasonal_periods=7)
    model = ExponentialSmoothing(series, trend="add", seasonal="add", seasonal_periods=7).fit()
    return model.forecast(steps=n_days)


def predict_prophet(series: pd.Series, n_days: int) -> np.ndarray:
    if not HAS_PROPHET:
        raise ValueError("Prophet tidak terinstall")

    df_p = series.reset_index()
    df_p.columns = ["ds", "y"]
    df_p["ds"] = df_p["ds"].dt.tz_localize(None)
    
    # Tambahkan regressor eksternal (eksogen) ke data training Prophet
    exc_map = get_live_exchange_rate(df_p["ds"].min(), df_p["ds"].max())
    
    # Tambahkan fitur Hijriah ke data training Prophet
    def get_prophet_hijri(date_obj):
        jd = date_obj.toordinal() + 1721426
        l = jd - 1948440 + 10632
        n = int((l - 1) / 10631)
        l = l - 10631 * n + 354
        j = int((10985 - l) / 5316) * int((50 + l) / 135) + int(l / 30) * int((700 - l) / 909)
        l = l - int((700 - j) / 909) * int((30 * j + 59) / 30) + 30
        h_month = int((24 * l + 30) / 709)
        h_day = l - int((709 * h_month + 24) / 24)
        is_ram = 1 if h_month == 9 else 0
        dist_to_eid = 99
        if h_month == 9:
            dist_to_eid = max(0, 30 - h_day)
        elif h_month == 10 and h_day <= 5:
            dist_to_eid = 0
        return is_ram, dist_to_eid

    df_p["is_ramadhan"] = df_p["ds"].map(lambda d: get_prophet_hijri(d)[0])
    df_p["days_to_eid"] = df_p["ds"].map(lambda d: get_prophet_hijri(d)[1])

    df_p["curah_hujan"] = df_p["ds"].dt.month.map(lambda m: 250.0 if m in [11, 12, 1, 2, 3, 4] else 80.0)
    df_p["kurs_usd"] = df_p["ds"].map(lambda d: exc_map.get(d.strftime("%Y-%m-%d"), 16200.0))
    df_p["inflasi"] = get_live_inflation_rate()

    model = Prophet(
        daily_seasonality=False, 
        weekly_seasonality=True, 
        yearly_seasonality=True,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=1.0
    )
    model.add_regressor("is_ramadhan")
    model.add_regressor("days_to_eid")
    model.add_regressor("curah_hujan")
    model.add_regressor("kurs_usd")
    model.add_regressor("inflasi")
    model.add_country_holidays(country_name="ID")
    model.fit(df_p)

    # Siapkan DataFrame masa depan dengan regressor eksogen dari live forecast API
    future = model.make_future_dataframe(periods=n_days, freq="D")
    
    # Gunakan prakiraan cuaca yang sudah tersimpan di cache memori (ditarik dari main thread)
    forecast_weather = get_live_rainfall_forecast(n_days)
    
    # Menarik estimasi kurs masa depan secara forward-fill (konstan dari hari terakhir)
    last_valid_rate = 16200.0
    sorted_keys = sorted(exc_map.keys())
    if sorted_keys:
        last_valid_rate = exc_map[sorted_keys[-1]]
        
    future["is_ramadhan"] = future["ds"].map(lambda d: get_prophet_hijri(d)[0])
    future["days_to_eid"] = future["ds"].map(lambda d: get_prophet_hijri(d)[1])
    future["curah_hujan"] = future["ds"].map(
        lambda d: forecast_weather.get(d.strftime("%Y-%m-%d"), 8.0 if d.month in [11, 12, 1, 2, 3, 4] else 1.5)
    )
    future["kurs_usd"] = future["ds"].map(lambda d: exc_map.get(d.strftime("%Y-%m-%d"), last_valid_rate))
    future["inflasi"] = get_live_inflation_rate()
    
    return model.predict(future)["yhat"].iloc[-n_days:].values


def predict_xgboost(series: pd.Series, n_days: int) -> np.ndarray:
    """Prediksi menggunakan XGBoost dengan strategi rekursif yang kaya fitur (Holidays, Live Exogenous, Lags, Diff)."""
    if not HAS_XGBOOST:
        raise ValueError("XGBoost tidak terinstall")

    df_feat = prepare_ml_features(series)
    if len(df_feat) < 20:
        raise ValueError("Data terlalu sedikit untuk dilatih dengan XGBoost (butuh min 20)")

    # Memasukkan regressor eksogen (curah_hujan, kurs_usd, inflasi, ramadhan, lebaran) ke dalam list fitur XGBoost
    feat_cols = [
        "day_of_week", "month", "day_of_month", "is_weekend", "is_holiday", "holiday_proximity",
        "is_ramadhan", "days_to_eid",
        "curah_hujan", "kurs_usd", "inflasi",
        "lag_1", "lag_2", "price_diff_1", "lag_7", "lag_14", "rolling_mean_7", "rolling_std_7"
    ]
    X = df_feat[feat_cols]
    y = df_feat["y"]

    # Tuning hyperparameter yang sangat robust untuk mencegah overfitting dan error accumulation pada deret waktu
    model = xgb.XGBRegressor(
        n_estimators=150, 
        learning_rate=0.03, 
        max_depth=3, 
        subsample=0.7,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="reg:squarederror"
    )
    model.fit(X, y)

    # Inisialisasi daftar hari libur Indonesia untuk loop prediksi rekursif
    try:
        import holidays
        id_holidays = holidays.Indonesia()
    except Exception:
        id_holidays = {}

    # Gunakan prakiraan cuaca yang sudah tersimpan di cache memori (ditarik dari main thread)
    forecast_weather = get_live_rainfall_forecast(n_days)
    
    # Dapatkan mapping data kurs harian historis untuk membantu pencarian forward-fill
    start_dt = series.index[0].to_pydatetime()
    end_dt = series.index[-1].to_pydatetime()
    exc_map = get_live_exchange_rate(start_dt, end_dt)
    
    last_valid_rate = 16200.0
    sorted_keys = sorted(exc_map.keys())
    if sorted_keys:
        last_valid_rate = exc_map[sorted_keys[-1]]
        
    live_inflation = get_live_inflation_rate()

    history = series.copy()
    forecasts = []
    curr_date = history.index[-1]

    for _ in range(n_days):
        curr_date += timedelta(days=1)
        curr_date_str = curr_date.strftime("%Y-%m-%d")
        
        # Hitung holiday proximity secara dinamis untuk tanggal prediksi baru
        is_hol = 1 if curr_date in id_holidays else 0
        hol_prox = 0
        for i in range(1, 8):
            if (curr_date + timedelta(days=i)) in id_holidays:
                hol_prox += 1
                
        # Konversi Hijriah dinamis untuk tanggal ramalan baru
        jd = curr_date.toordinal() + 1721426
        l = jd - 1948440 + 10632
        n = int((l - 1) / 10631)
        l = l - 10631 * n + 354
        j = int((10985 - l) / 5316) * int((50 + l) / 135) + int(l / 30) * int((700 - l) / 909)
        l = l - int((700 - j) / 909) * int((30 * j + 59) / 30) + 30
        h_month = int((24 * l + 30) / 709)
        h_day = l - int((709 * h_month + 24) / 24)
        
        is_ram = 1 if h_month == 9 else 0
        dist_to_eid = 99
        if h_month == 9:
            dist_to_eid = max(0, 30 - h_day)
        elif h_month == 10 and h_day <= 5:
            dist_to_eid = 0
            
        feat = pd.DataFrame(
            [
                {
                    "day_of_week": curr_date.weekday(),
                    "month": curr_date.month,
                    "day_of_month": curr_date.day,
                    "is_weekend": 1 if curr_date.weekday() in [5, 6] else 0,
                    "is_holiday": is_hol,
                    "holiday_proximity": hol_prox,
                    "is_ramadhan": is_ram,
                    "days_to_eid": dist_to_eid,
                    "curah_hujan": forecast_weather.get(curr_date_str, 8.0 if curr_date.month in [11, 12, 1, 2, 3, 4] else 1.5),
                    "kurs_usd": exc_map.get(curr_date_str, last_valid_rate),
                    "inflasi": live_inflation,
                    "lag_1": history.iloc[-1],
                    "lag_2": history.iloc[-2] if len(history) >= 2 else history.iloc[-1],
                    "price_diff_1": history.iloc[-1] - history.iloc[-2] if len(history) >= 2 else 0,
                    "lag_7": history.iloc[-7] if len(history) >= 7 else history.iloc[-1],
                    "lag_14": history.iloc[-14] if len(history) >= 14 else history.iloc[-1],
                    "rolling_mean_7": history.iloc[-7:].mean(),
                    "rolling_std_7": history.iloc[-7:].std() if len(history) >= 7 else 0,
                }
            ]
        )
        prediction = model.predict(feat[feat_cols])[0]
        forecasts.append(prediction)
        history = pd.concat([history, pd.Series([prediction], index=[curr_date])])

    return np.array(forecasts)


def forecast_all_models(series: pd.Series, n_days: int) -> dict[str, list[float]]:
    forecasters = {
        "arima": predict_arima,
        "ets": predict_ets,
        "prophet": predict_prophet,
        "xgboost": predict_xgboost,
    }
    all_forecasts = {}

    for model_name, forecaster in forecasters.items():
        try:
            forecast = forecaster(series, n_days)
            all_forecasts[model_name] = [round(p, 0) for p in forecast]
        except Exception:
            all_forecasts[model_name] = []

    return all_forecasts
