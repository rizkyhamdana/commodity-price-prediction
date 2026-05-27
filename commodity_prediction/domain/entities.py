"""Domain entities that are independent from frameworks and storage."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ForecastPoint:
    date: str
    price: float


@dataclass(frozen=True)
class ForecastRun:
    commodity: str
    model_used: str
    mape: float
    last_price: float
    forecast: tuple[ForecastPoint, ...]
    model_scores: dict[str, float]
    all_model_forecasts: dict[str, list[float]]
