"""Compatibility facade for forecast model APIs."""

from commodity_prediction.application.services import backtest_model
from commodity_prediction.infrastructure.ml.models import forecast_all_models, predict_xgboost

__all__ = ["backtest_model", "forecast_all_models", "predict_xgboost"]
