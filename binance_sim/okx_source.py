#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OKX spot market-data adapter (cross-exchange validation).

Provides the same DataFrame schema the backtester expects, sourced from OKX
instead of Binance. Running the identical strategy on a *different* exchange
(different coins, listings and microstructure) is a strong generalisation
test: a real edge should survive the venue change; an artefact of Binance
data should not.

Only public endpoints are used (no keys). Candles come from
``/api/v5/market/history-candles`` (paginated backwards).
"""

from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from typing import List, Optional

import pandas as pd

from . import paper_bot_strategy as S

OKX_BASE = "https://www.okx.com"
FOUR_H_MS = 4 * 60 * 60 * 1000
CACHE_DIR = os.environ.get("BINANCE_SIM_CACHE", "data/cache")
OKX_DIR = os.path.join(CACHE_DIR, "okx")


def _get(path: str, params: dict, retries: int = 4):
    url = OKX_BASE + path + "?" + urllib.parse.urlencode(params)
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "binance-sim/0.1"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode())
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(0.4 * (attempt + 1))
    raise RuntimeError(f"OKX GET {url} failed: {last}")


def list_usdt_symbols() -> List[str]:
    """Live OKX spot USDT instruments, minus the strategy exclusions."""
    data = _get("/api/v5/public/instruments", {"instType": "SPOT"})
    out = []
    for i in data.get("data", []):
        if i.get("quoteCcy") != "USDT" or i.get("state") != "live":
            continue
        base = i.get("baseCcy", "")
        if S.should_exclude(base):
            continue
        out.append(i["instId"])
    return sorted(out)


def _cache_path(inst: str, start_ms: int, end_ms: int) -> str:
    os.makedirs(OKX_DIR, exist_ok=True)
    safe = inst.replace("-", "_")
    return os.path.join(OKX_DIR, f"{safe}_4H_{start_ms}_{end_ms}.csv")


def fetch_range(inst: str, start_ms: int, end_ms: int,
                use_cache: bool = True) -> Optional[pd.DataFrame]:
    """Fetch 4h candles for [start_ms, end_ms] from OKX, paginating backwards."""
    path = _cache_path(inst, start_ms, end_ms)
    if use_cache and os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:  # noqa: BLE001
            pass

    rows = []
    after = end_ms + 1
    while after > start_ms:
        data = _get("/api/v5/market/history-candles",
                    {"instId": inst, "bar": "4H", "after": str(after), "limit": "100"})
        batch = data.get("data", [])
        if not batch:
            break
        rows.extend(batch)
        oldest = int(batch[-1][0])
        if oldest <= start_ms or oldest >= after:
            break
        after = oldest
        time.sleep(0.12)  # respect rate limits

    if not rows:
        return None
    # OKX row: [ts, o, h, l, c, vol(base), volCcy, volCcyQuote, confirm]
    df = pd.DataFrame(rows, columns=[
        "time", "open", "high", "low", "close", "vol", "volccy", "quote_av", "confirm"])
    for c in ["open", "high", "low", "close", "vol", "quote_av"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["time"] = df["time"].astype("int64")
    df["volume"] = df["vol"]
    df["close_time"] = df["time"] + FOUR_H_MS - 1
    df = df[["time", "open", "high", "low", "close", "volume", "close_time", "quote_av"]]
    df = df.drop_duplicates("time").sort_values("time").reset_index(drop=True)
    df = df[(df["close_time"] >= start_ms) & (df["close_time"] <= end_ms + 1)]
    if use_cache and len(df):
        df.to_csv(path, index=False)
    return df if len(df) else None


def prefetch_universe(symbols, start_ms, end_ms, workers: int = 8, log=print):
    """Download (parallel) and return featurisable frames for OKX symbols."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    log(f"OKX: fetching {len(symbols)} symbols, {workers} workers")
    result = {}
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_range, s, start_ms, end_ms): s for s in symbols}
        for fut in as_completed(futs):
            sym = futs[fut]
            done += 1
            try:
                df = fut.result()
            except Exception:  # noqa: BLE001
                df = None
            if df is not None and len(df) >= S.MIN_HISTORY:
                result[sym] = df
            if done % 50 == 0:
                log(f"  fetched {done}/{len(symbols)}  (usable: {len(result)})")
    log(f"OKX usable symbols: {len(result)}")
    return result
