"""Model backtesting and scoring."""

import numpy as np
import pandas as pd

from commodity_prediction.logging_config import logger
from commodity_prediction.infrastructure.ml.models.forecasting import predict_arima, predict_ets, predict_prophet, predict_xgboost


def backtest_model(series: pd.Series, test_days: int = 30, model_type: str = "arima") -> dict:
    """Menguji performa model pada 30 hari terakhir untuk menghitung MAPE, MAE, RMSE, dan Combined Score."""
    fallback_metrics = {
        "mape": 99.9,
        "mae": 9999.0,
        "rmse": 9999.0,
        "relative_mae": 99.9,
        "relative_rmse": 99.9,
        "combined_score": 99.9,
        "forecast": []
    }
    
    if len(series) < test_days + 20:
        return fallback_metrics

    train, test = series.iloc[:-test_days], series.iloc[-test_days:]
    forecasters = {
        "arima": predict_arima,
        "ets": predict_ets,
        "prophet": predict_prophet,
        "xgboost": predict_xgboost,
    }

    try:
        forecaster = forecasters.get(model_type)
        if forecaster is None:
            return fallback_metrics

        forecast = forecaster(train, test_days)[: len(test)]
        
        # Hitung metrik evaluasi
        actual = test.values
        mape = np.mean(np.abs((actual - forecast) / actual)) * 100
        mae = np.mean(np.abs(actual - forecast))
        rmse = np.sqrt(np.mean((actual - forecast) ** 2))
        
        # Normalisasi MAE dan RMSE terhadap rata-rata harga aktual
        mean_actual = np.mean(actual)
        relative_mae = (mae / mean_actual) * 100 if mean_actual > 0 else 99.9
        relative_rmse = (rmse / mean_actual) * 100 if mean_actual > 0 else 99.9
        
        # Skor Gabungan (30% MAPE, 30% Rel MAE, 40% Rel RMSE)
        combined_score = (0.3 * mape) + (0.3 * relative_mae) + (0.4 * relative_rmse)
        
        return {
            "mape": float(mape),
            "mae": float(mae),
            "rmse": float(rmse),
            "relative_mae": float(relative_mae),
            "relative_rmse": float(relative_rmse),
            "combined_score": float(combined_score),
            "forecast": forecast
        }
    except Exception as e:
        logger.error(f"❌ Error pada model {model_type}: {e}")
        return fallback_metrics
