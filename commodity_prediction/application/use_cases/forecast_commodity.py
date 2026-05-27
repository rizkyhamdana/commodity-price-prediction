"""Main forecast orchestration."""

import json
import os
from datetime import datetime

import pandas as pd

from commodity_prediction.config import MODEL_NAMES
from commodity_prediction.domain.entities import ForecastPoint, ForecastRun
from commodity_prediction.infrastructure.data import extract_commodity_series, load_json_data, update_history_with_api
from commodity_prediction.infrastructure.ml.models import forecast_all_models
from commodity_prediction.infrastructure.output import plot_forecast
from commodity_prediction.logging_config import logger
from commodity_prediction.application.services import backtest_model


def run_pipeline(json_path, commodity_name, n_days=7, out_dir="output", use_api=True, df=None):
    """Orkestrasi utama: Ambil Data -> Kompetisi Model -> Ramalan -> Simpan JSON."""
    if df is None:
        df, _ = load_json_data(json_path)

    if use_api:
        df = update_history_with_api(df, json_path)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"data": df.to_dict(orient="records")}, f, indent=2, ensure_ascii=False)

    series = extract_commodity_series(df, commodity_name)

    scores = {}
    for model_name in MODEL_NAMES:
        scores[model_name] = backtest_model(series, test_days=30, model_type=model_name)["mape"]

    valid_scores = {k: v for k, v in scores.items() if v < 99.0}
    if not valid_scores:
        logger.warning(f"⚠️ Semua model gagal untuk {commodity_name}. Menggunakan fallback.")
        best_type = "ets"
        best_mape = 99.9
    else:
        best_type = min(valid_scores, key=valid_scores.get)
        best_mape = valid_scores[best_type]

    print(
        f"   - ARIMA: {scores['arima']:.2f}% | ETS: {scores['ets']:.2f}% | "
        f"PROPHET: {scores['prophet']:.2f}% | XGB: {scores['xgboost']:.2f}%"
    )
    print(f"🏆 Pemenang: {best_type.upper()} ({best_mape:.2f}%)")

    forecast_dates = pd.date_range(start=series.index[-1], periods=n_days + 1, freq="D")[1:]
    forecast_dates_str = [d.strftime("%Y-%m-%d") for d in forecast_dates]
    all_forecasts = forecast_all_models(series, n_days)

    best_fc = all_forecasts.get(best_type) or all_forecasts.get("ets") or []
    forecast_df = pd.DataFrame({"date": forecast_dates_str, "price": best_fc})
    forecast_run = ForecastRun(
        commodity=commodity_name,
        model_used=best_type,
        mape=float(best_mape),
        last_price=float(series.iloc[-1]),
        forecast=tuple(ForecastPoint(date=row["date"], price=float(row["price"])) for _, row in forecast_df.iterrows()),
        model_scores={k: float(v) for k, v in scores.items()},
        all_model_forecasts=all_forecasts,
    )

    res = {
        "commodity": forecast_run.commodity,
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_used": forecast_run.model_used,
        "mape": round(forecast_run.mape, 2),
        "last_price": forecast_run.last_price,
        "forecast": [point.__dict__ for point in forecast_run.forecast],
    }

    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"forecast_{commodity_name.lower().replace(' ', '_')}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)

    plot_path = os.path.join(out_dir, f"chart_{commodity_name.lower().replace(' ', '_')}.png")
    plot_forecast(series, forecast_dates, best_fc, commodity_name, plot_path)

    print(f"🔮 Hasil Prediksi ({best_type.upper()}):")
    for date, price in zip(forecast_dates_str[:3], best_fc[:3]):
        print(f"   {date} → Rp {price:,}")

    return forecast_df, best_mape, best_type, scores, all_forecasts
