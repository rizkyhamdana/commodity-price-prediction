"""Backward-compatible imports for the refactored forecasting package."""

from commodity_prediction.infrastructure.data import extract_commodity_series, load_json_data, update_history_with_api
from commodity_prediction.infrastructure.ml import prepare_ml_features
from commodity_prediction.models import backtest_model, predict_xgboost
from commodity_prediction.application.use_cases.forecast_commodity import run_pipeline
from commodity_prediction.infrastructure.output import plot_forecast

__all__ = [
    "backtest_model",
    "extract_commodity_series",
    "load_json_data",
    "plot_forecast",
    "prepare_ml_features",
    "predict_xgboost",
    "run_pipeline",
    "update_history_with_api",
]
