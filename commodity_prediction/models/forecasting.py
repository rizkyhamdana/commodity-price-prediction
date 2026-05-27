"""Compatibility facade for forecasting models."""

from commodity_prediction.infrastructure.ml.models.forecasting import (
    forecast_all_models,
    predict_arima,
    predict_ets,
    predict_prophet,
    predict_xgboost,
)

__all__ = ["forecast_all_models", "predict_arima", "predict_ets", "predict_prophet", "predict_xgboost"]
