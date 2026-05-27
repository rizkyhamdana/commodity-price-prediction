"""Use case entry points."""

__all__ = ["run_all_predictions", "run_pipeline"]


def __getattr__(name):
    if name == "run_pipeline":
        from .forecast_commodity import run_pipeline

        return run_pipeline
    if name == "run_all_predictions":
        from .generate_mobile_backend import run_all_predictions

        return run_all_predictions
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
