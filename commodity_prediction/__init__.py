"""Commodity price forecasting package."""

__all__ = [
    "extract_commodity_series",
    "load_json_data",
    "run_pipeline",
    "update_history_with_api",
]


def __getattr__(name):
    if name in {"extract_commodity_series", "load_json_data", "update_history_with_api"}:
        from .infrastructure import data

        return getattr(data, name)
    if name == "run_pipeline":
        from .application.use_cases.forecast_commodity import run_pipeline

        return run_pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
