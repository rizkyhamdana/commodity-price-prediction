"""Historical commodity data loading, updating, and extraction."""

import json
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

from commodity_prediction.logging_config import logger


def load_json_data(filepath: str) -> tuple[pd.DataFrame, str]:
    """Membaca data riwayat harga dari JSON lokal."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File tidak ditemukan: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict) and "data" in raw:
        df = pd.DataFrame(raw["data"])
    else:
        df = pd.DataFrame(raw)

    return df, "dataframe"


def _get_sorted_date_columns(df: pd.DataFrame) -> list[str]:
    date_cols = []
    for col in df.columns:
        try:
            datetime.strptime(col, "%d/%m/%Y")
            date_cols.append(col)
        except Exception:
            continue
    return sorted(date_cols, key=lambda x: datetime.strptime(x, "%d/%m/%Y"))


def update_history_with_api(
    df: pd.DataFrame,
    history_path: str,
    sim_date: str | None = None,
    force_start_date: str | None = None,
) -> pd.DataFrame:
    """Mengupdate data sejarah menggunakan API Bank Indonesia (PIHPS) dengan chunking 180 hari."""
    now_obj = datetime.strptime(sim_date, "%Y-%m-%d") if sim_date else datetime.now()
    date_cols = _get_sorted_date_columns(df)

    if force_start_date:
        start_date_obj = datetime.strptime(force_start_date, "%Y-%m-%d")
        end_date_obj = datetime.strptime(date_cols[0], "%d/%m/%Y") - timedelta(days=1) if date_cols else now_obj
    else:
        if not date_cols:
            return df
        start_date_obj = datetime.strptime(date_cols[-1], "%d/%m/%Y") - timedelta(days=7)
        end_date_obj = now_obj

    if start_date_obj > end_date_obj:
        logger.info("✅ Data history sudah up-to-date.")
        return df

    last_date_col = date_cols[-1] if date_cols else None
    for d_obj in pd.date_range(start=start_date_obj, end=end_date_obj):
        d_col = d_obj.strftime("%d/%m/%Y")
        if d_col not in df.columns:
            df[d_col] = df[last_date_col] if last_date_col else np.nan

    current_start = start_date_obj
    actual_last_date_api = None

    while current_start <= end_date_obj:
        current_end = min(current_start + timedelta(days=180), end_date_obj)
        start_str = current_start.strftime("%Y-%m-%d")
        end_str = current_end.strftime("%Y-%m-%d")
        api_url = (
            "https://www.bi.go.id/hargapangan/WebSite/TabelHarga/GetGridDataDaerah?"
            "price_type_id=1&comcat_id=&province_id=&regency_id=&market_id=&tipe_laporan=1"
            f"&start_date={start_str}&end_date={end_str}&_={int(datetime.now().timestamp() * 1000)}"
        )

        logger.info(f"🌐 Menarik chunk data: {start_str} s/d {end_str}")
        try:
            response = requests.get(api_url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()
            raw_api = response.json()
            if "data" in raw_api and raw_api["data"]:
                new_entries = raw_api["data"]
                api_date_cols = _get_sorted_date_columns(pd.DataFrame(new_entries))
                if api_date_cols:
                    latest_api_col = api_date_cols[-1]
                    if (
                        actual_last_date_api is None
                        or datetime.strptime(latest_api_col, "%d/%m/%Y")
                        > datetime.strptime(actual_last_date_api, "%d/%m/%Y")
                    ):
                        actual_last_date_api = latest_api_col

                for entry in new_entries:
                    name_api = str(entry.get("name", "")).strip().lower()
                    mask = df["name"].astype(str).str.strip().str.lower() == name_api
                    if mask.any():
                        for d_col in api_date_cols:
                            val = entry[d_col]
                            df.loc[mask, d_col] = np.nan if val in ("-", "", None) else val
        except Exception as e:
            logger.warning(f"⚠️ Gagal menarik data chunk {start_str}: {e}")

        current_start = current_end + timedelta(days=1)

    date_cols_sorted = _get_sorted_date_columns(df)
    df[date_cols_sorted] = df[date_cols_sorted].ffill(axis=1)

    if actual_last_date_api:
        logger.info(f"✅ Sinkronisasi selesai: Data API asli ditemukan sampai {actual_last_date_api}. (Lookback 7 hari).")
    else:
        logger.info("✅ Sinkronisasi selesai: Tidak ada data baru di API, menggunakan harga terakhir (ffill).")

    return df


def extract_commodity_series(df: pd.DataFrame, commodity_name: str, level_filter: int | None = 1) -> pd.Series:
    """Mengambil deret waktu harga untuk komoditas spesifik dan membersihkannya."""
    non_date_cols = {"no", "name", "level"}
    if level_filter is not None and "level" in df.columns:
        df = df[df["level"] == level_filter]

    mask = df["name"].astype(str).str.strip().str.lower() == commodity_name.strip().lower()
    if mask.sum() == 0:
        raise ValueError(f"Komoditas '{commodity_name}' tidak ditemukan di dataset.")

    row = df[mask].iloc[0]
    price_dict = {}
    for col in df.columns:
        if col.lower() not in non_date_cols:
            try:
                val = str(row[col]).replace(",", "")
                if val and val != "nan" and val != "-":
                    price_dict[datetime.strptime(col, "%d/%m/%Y")] = float(val)
            except Exception:
                continue

    if not price_dict:
        raise ValueError(f"Tidak ada data harga valid untuk {commodity_name}")

    series = pd.Series(price_dict).sort_index()
    full_idx = pd.date_range(start=series.index.min(), end=series.index.max(), freq="D")
    return series.reindex(full_idx).interpolate(method="linear").ffill().bfill()
