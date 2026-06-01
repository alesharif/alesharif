#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""High-resolution kline loader (5m and 1s) from the Binance daily archive.

Used by the hybrid exit engine: scan exits on 5-minute candles, and when a
candle's range approaches the stop, zoom into 1-second candles for tick-like
precision (matching the live bot's 30s monitoring, even finer).
"""

from __future__ import annotations

import io
import os
import zipfile
from typing import Optional

import pandas as pd
import requests

ARCHIVE = "https://data.binance.vision/data/spot/daily/klines/{s}/{iv}/{s}-{iv}-{d}.zip"
CACHE_DIR = os.environ.get("BINANCE_SIM_CACHE", "data/cache")
HIRES_DIR = os.path.join(CACHE_DIR, "hires")

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "binance-sim/0.1"})


def _norm(v: int) -> int:
    return int(v // 1000) if v > 1e15 else int(v)


def load_day(symbol: str, interval: str, day: str) -> Optional[pd.DataFrame]:
    """Load one day of klines at the given interval ('5m' or '1s'). Cached."""
    cache = os.path.join(HIRES_DIR, f"{symbol}-{interval}-{day}.csv")
    if os.path.exists(cache):
        try:
            return pd.read_csv(cache)
        except Exception:  # noqa: BLE001
            pass
    url = ARCHIVE.format(s=symbol, iv=interval, d=day)
    try:
        r = _SESSION.get(url, timeout=60)
        if r.status_code != 200 or r.content[:2] != b"PK":
            return None
        z = zipfile.ZipFile(io.BytesIO(r.content))
        raw = z.read(z.namelist()[0]).decode()
    except Exception:  # noqa: BLE001
        return None
    first = raw.splitlines()[0].split(",")[0]
    header = None if first.replace(".", "").replace("-", "").isdigit() else 0
    df = pd.read_csv(io.StringIO(raw), header=header).iloc[:, :7]
    df.columns = ["time", "open", "high", "low", "close", "volume", "close_time"]
    df["time"] = df["time"].map(_norm).astype("int64")
    df["close_time"] = df["close_time"].map(_norm).astype("int64")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    os.makedirs(HIRES_DIR, exist_ok=True)
    df.to_csv(cache, index=False)
    return df


def load_range(symbol: str, interval: str, start_ms: int, end_ms: int) -> Optional[pd.DataFrame]:
    """Load [start_ms, end_ms] at the given interval across day archives."""
    days = pd.date_range(pd.Timestamp(start_ms, unit="ms").normalize(),
                         pd.Timestamp(end_ms, unit="ms").normalize(), freq="D")
    frames = []
    for d in days:
        df = load_day(symbol, interval, d.strftime("%Y-%m-%d"))
        if df is not None and len(df):
            frames.append(df)
    if not frames:
        return None
    out = pd.concat(frames, ignore_index=True).drop_duplicates("time").sort_values("time")
    out = out[(out["time"] >= start_ms) & (out["time"] <= end_ms)].reset_index(drop=True)
    return out if len(out) else None
