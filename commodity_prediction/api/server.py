"""Compatibility facade for the API server."""

from commodity_prediction.interfaces.api.server import app, run_api_server

__all__ = ["app", "run_api_server"]
