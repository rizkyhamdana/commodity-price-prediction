"""Optional forecasting dependency probes."""

from commodity_prediction.logging_config import logger

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    HAS_STATSMODELS = True
except Exception as e:
    logger.warning(f"⚠️ Statsmodels error: {e}")
    ExponentialSmoothing = None
    SARIMAX = None
    HAS_STATSMODELS = False

try:
    from pmdarima import auto_arima

    HAS_PMDARIMA = True
except Exception:
    auto_arima = None
    HAS_PMDARIMA = False

try:
    from prophet import Prophet

    HAS_PROPHET = True
except Exception:
    Prophet = None
    HAS_PROPHET = False

try:
    import xgboost as xgb

    HAS_XGBOOST = True
except Exception as e:
    logger.warning(f"⚠️ XGBoost error: {e}")
    xgb = None
    HAS_XGBOOST = False
