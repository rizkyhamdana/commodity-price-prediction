"""Compatibility facade for LLM insight generation."""

from commodity_prediction.infrastructure.llm import generate_commodity_insight, generate_global_insight

__all__ = ["generate_commodity_insight", "generate_global_insight"]
