"""Feature engineering for machine-learning forecasting models."""

from datetime import timedelta

import pandas as pd


def prepare_ml_features(series: pd.Series) -> pd.DataFrame:
    """Mempersiapkan fitur teknis untuk XGBoost."""
    df = series.to_frame(name="y")
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df["day_of_month"] = df.index.day
    df["is_weekend"] = df.index.dayofweek.isin([5, 6]).astype(int)

    try:
        import holidays

        id_holidays = holidays.Indonesia()
        df["is_holiday"] = df.index.map(lambda x: 1 if x in id_holidays else 0)
        df["holiday_proximity"] = 0
        for i in range(1, 8):
            df["holiday_proximity"] += df.index.map(lambda x: 1 if (x + timedelta(days=i)) in id_holidays else 0)
    except Exception:
        df["is_holiday"] = 0
        df["holiday_proximity"] = 0

    df["lag_1"] = df["y"].shift(1)
    df["lag_7"] = df["y"].shift(7)
    df["lag_14"] = df["y"].shift(14)
    df["rolling_mean_7"] = df["y"].shift(1).rolling(window=7).mean()
    df["rolling_std_7"] = df["y"].shift(1).rolling(window=7).std()
    return df.dropna()
