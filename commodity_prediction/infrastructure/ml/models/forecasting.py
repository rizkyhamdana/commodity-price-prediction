"""Forecast generation for each supported algorithm."""

from datetime import timedelta
import warnings

import numpy as np
import pandas as pd

from commodity_prediction.infrastructure.ml.features import prepare_ml_features
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
        order = auto_arima(np.log1p(series), seasonal=False, m=1).order
        model = SARIMAX(np.log1p(series), order=order).fit(disp=False)
    return np.expm1(model.forecast(steps=n_days))


def predict_ets(series: pd.Series, n_days: int) -> np.ndarray:
    if not HAS_STATSMODELS:
        raise ValueError("Statsmodels tidak terinstall")

    model = ExponentialSmoothing(series, trend="add").fit()
    return model.forecast(steps=n_days)


def predict_prophet(series: pd.Series, n_days: int) -> np.ndarray:
    if not HAS_PROPHET:
        raise ValueError("Prophet tidak terinstall")

    df_p = series.reset_index()
    df_p.columns = ["ds", "y"]
    df_p["ds"] = df_p["ds"].dt.tz_localize(None)
    model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
    model.add_country_holidays(country_name="ID")
    model.fit(df_p)
    return model.predict(model.make_future_dataframe(periods=n_days, freq="D"))["yhat"].iloc[-n_days:].values


def predict_xgboost(series: pd.Series, n_days: int) -> np.ndarray:
    """Prediksi menggunakan XGBoost dengan strategi rekursif."""
    if not HAS_XGBOOST:
        raise ValueError("XGBoost tidak terinstall")

    df_feat = prepare_ml_features(series)
    if len(df_feat) < 20:
        raise ValueError("Data terlalu sedikit untuk dilatih dengan XGBoost (butuh min 20)")

    feat_cols = ["day_of_week", "month", "day_of_month", "lag_1", "lag_7", "lag_14", "rolling_mean_7", "rolling_std_7"]
    X = df_feat[feat_cols]
    y = df_feat["y"]

    model = xgb.XGBRegressor(n_estimators=150, learning_rate=0.07, max_depth=5, objective="reg:squarederror")
    model.fit(X, y)

    history = series.copy()
    forecasts = []
    curr_date = history.index[-1]

    for _ in range(n_days):
        curr_date += timedelta(days=1)
        feat = pd.DataFrame(
            [
                {
                    "day_of_week": curr_date.weekday(),
                    "month": curr_date.month,
                    "day_of_month": curr_date.day,
                    "lag_1": history.iloc[-1],
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
