"""Model backtesting and scoring."""

import numpy as np
import pandas as pd

from commodity_prediction.logging_config import logger
from commodity_prediction.infrastructure.ml.models.forecasting import predict_arima, predict_ets, predict_prophet, predict_xgboost


def backtest_model(series: pd.Series, test_days: int = 30, model_type: str = "arima") -> dict:
    """Menguji performa model pada 30 hari terakhir untuk menghitung MAPE."""
    if len(series) < test_days + 20:
        return {"mape": 99.9}

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
            return {"mape": 99.9}

        forecast = forecaster(train, test_days)[: len(test)]
        mape = np.mean(np.abs((test.values - forecast) / test.values)) * 100
        return {"mape": mape, "forecast": forecast}
    except Exception as e:
        logger.error(f"❌ Error pada model {model_type}: {e}")
        return {"mape": 99.9}
