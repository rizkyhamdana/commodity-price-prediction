"""Compatibility facade for infrastructure data adapters."""

from commodity_prediction.infrastructure.data import extract_commodity_series, load_json_data, update_history_with_api

__all__ = ["extract_commodity_series", "load_json_data", "update_history_with_api"]
