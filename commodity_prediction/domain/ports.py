"""Application ports for infrastructure adapters."""

from typing import Protocol

import pandas as pd


class HistoryRepository(Protocol):
    def load(self, path: str) -> pd.DataFrame:
        ...

    def save(self, path: str, df: pd.DataFrame) -> None:
        ...


class HistoryUpdater(Protocol):
    def update(self, df: pd.DataFrame, history_path: str) -> pd.DataFrame:
        ...


class CommoditySeriesExtractor(Protocol):
    def extract(self, df: pd.DataFrame, commodity_name: str) -> pd.Series:
        ...


class ChartRenderer(Protocol):
    def render(self, series: pd.Series, forecast_dates, forecast_values, commodity_name: str, out_path: str) -> None:
        ...
