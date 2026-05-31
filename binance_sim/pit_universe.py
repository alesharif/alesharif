#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Point-in-time universe + historical data from the Binance public archive.

Unlike the live REST API (which only knows *currently listed* symbols), the
``data.binance.vision`` archive keeps candles for **delisted** coins too. By
sourcing the universe and the candles from the archive we eliminate the
survivorship bias that makes the API-based backtest wildly optimistic.

  * symbol list  : S3 listing of every spot/USDT folder that ever existed
  * availability : the set of months a symbol has data == its trading life
  * candles      : monthly 4h kline ZIPs, concatenated per symbol

Downloads are parallelised and cached to disk, so the first full build is the
only slow part; reruns are instant.
"""

from __future__ import annotations

import io
import os
import re
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import pandas as pd
import requests

from . import paper_bot_strategy as S

# S3 REST endpoints that serve the XML bucket listing. We try them in order;
# some AWS edge nodes intermittently present a "certificate not yet valid"
# cert (a clock/rotation quirk), so we retry and, as a last resort for this
# PUBLIC archive listing only, fall back to an unverified TLS handshake. Actual
# candle ZIPs are always downloaded from the verified CloudFront host below.
S3_HOSTS = [
    "https://s3.ap-northeast-1.amazonaws.com/data.binance.vision/",
    "https://s3.amazonaws.com/data.binance.vision/",
    "https://data.binance.vision.s3.amazonaws.com/",
]
ARCHIVE = "https://data.binance.vision/"
CACHE_DIR = os.environ.get("BINANCE_SIM_CACHE", "data/cache")
ARCHIVE_DIR = os.path.join(CACHE_DIR, "archive")

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "binance-sim/0.1"})

# silence the single warning emitted when we fall back to verify=False
try:
    requests.packages.urllib3.disable_warnings()  # type: ignore[attr-defined]
except Exception:  # noqa: BLE001
    pass


def _get(url: str, params: dict, timeout: int = 30, tries: int = 4):
    """GET with verified TLS first; fall back to unverified on cert errors."""
    last = None
    for attempt in range(tries):
        for verify in (True, False):
            try:
                return _SESSION.get(url, params=params, timeout=timeout, verify=verify)
            except requests.exceptions.SSLError as exc:
                last = exc  # try unverified next, then retry
            except requests.RequestException as exc:
                last = exc
                break  # non-TLS error: retry the request fresh
    raise RuntimeError(f"GET {url} failed after {tries} tries: {last}")


def _get_listing(params: dict):
    """Fetch an S3 XML listing, trying each mirror host until one responds."""
    last = None
    for host in S3_HOSTS:
        try:
            return _get(host, params).text
        except Exception as exc:  # noqa: BLE001
            last = exc
    raise RuntimeError(f"All S3 listing hosts failed: {last}")


def _list_prefix(prefix: str) -> List[str]:
    """Return the CommonPrefixes (sub-folders) under an S3 prefix, paginated."""
    out: List[str] = []
    marker = ""
    while True:
        params = {"delimiter": "/", "prefix": prefix}
        if marker:
            params["marker"] = marker
        text = _get_listing(params)
        names = re.findall(rf"<Prefix>{re.escape(prefix)}([^/]+)/</Prefix>", text)
        out.extend(names)
        if "<IsTruncated>true</IsTruncated>" not in text:
            break
        nm = re.search(r"<NextMarker>([^<]+)</NextMarker>", text)
        if not nm:
            break
        marker = nm.group(1)
    return out


def list_all_usdt_symbols() -> List[str]:
    """Every spot USDT symbol that ever existed, minus the strategy exclusions."""
    all_syms = _list_prefix("data/spot/monthly/klines/")
    keep = []
    for s in all_syms:
        if not s.endswith("USDT"):
            continue
        base = s[:-4]
        if S.should_exclude(base):
            continue
        keep.append(s)
    return sorted(keep)


def list_symbol_months(symbol: str, interval: str = "4h") -> List[str]:
    """Available 'YYYY-MM' months for a symbol's interval (its trading life)."""
    prefix = f"data/spot/monthly/klines/{symbol}/{interval}/"
    marker = ""
    months: List[str] = []
    while True:
        params = {"prefix": prefix}
        if marker:
            params["marker"] = marker
        text = _get_listing(params)
        keys = re.findall(
            rf"<Key>{re.escape(prefix)}{re.escape(symbol)}-{interval}-(\d{{4}}-\d{{2}})\.zip</Key>",
            text)
        months.extend(keys)
        if "<IsTruncated>true</IsTruncated>" not in text:
            break
        nm = re.search(r"<NextMarker>([^<]+)</NextMarker>", text)
        if not nm:
            break
        marker = nm.group(1)
    return sorted(set(months))


def _normalise_time(v: int) -> int:
    """Archive switched to microseconds in 2025; fold everything back to ms."""
    return int(v // 1000) if v > 1e15 else int(v)


def _download_month(symbol: str, ym: str, interval: str = "4h") -> Optional[pd.DataFrame]:
    cache = os.path.join(ARCHIVE_DIR, f"{symbol}-{interval}-{ym}.csv")
    if os.path.exists(cache):
        try:
            return pd.read_csv(cache)
        except Exception:  # noqa: BLE001
            pass
    url = f"{ARCHIVE}data/spot/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{ym}.zip"
    try:
        r = _get(url, {}, timeout=60)
        if r.status_code != 200 or not r.content[:2] == b"PK":
            return None
        z = zipfile.ZipFile(io.BytesIO(r.content))
        raw = z.read(z.namelist()[0]).decode()
    except Exception:  # noqa: BLE001
        return None
    first_cell = raw.splitlines()[0].split(",")[0]
    header = None if first_cell.replace(".", "").replace("-", "").isdigit() else 0
    df = pd.read_csv(io.StringIO(raw), header=header)
    df = df.iloc[:, :8]
    df.columns = ["time", "open", "high", "low", "close", "volume", "close_time", "quote_av"]
    df["time"] = df["time"].map(_normalise_time).astype("int64")
    df["close_time"] = df["close_time"].map(_normalise_time).astype("int64")
    for c in ["open", "high", "low", "close", "volume", "quote_av"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    df.to_csv(cache, index=False)
    return df


def _months_in_range(start_ms: int, end_ms: int) -> List[str]:
    # Timestamps from epoch-ms are naive UTC; bucket to calendar months.
    start = pd.Timestamp(start_ms, unit="ms").to_period("M")
    end = pd.Timestamp(end_ms, unit="ms").to_period("M")
    return [str(p) for p in pd.period_range(start, end, freq="M")]


def fetch_range_archive(symbol: str, start_ms: int, end_ms: int,
                        interval: str = "4h") -> Optional[pd.DataFrame]:
    """Concatenate monthly archive candles covering [start_ms, end_ms]."""
    months = _months_in_range(start_ms, end_ms)
    frames = []
    for ym in months:
        df = _download_month(symbol, ym, interval)
        if df is not None and len(df):
            frames.append(df)
    if not frames:
        return None
    out = pd.concat(frames, ignore_index=True).drop_duplicates("time")
    out = out.sort_values("time").reset_index(drop=True)
    out = out[(out["close_time"] >= start_ms) & (out["close_time"] <= end_ms + 1)]
    return out if len(out) else None


def prefetch_universe(symbols: List[str], start_ms: int, end_ms: int,
                      workers: int = 24, log=print) -> Dict[str, pd.DataFrame]:
    """Download (in parallel) and return candle frames for all symbols.

    Symbols with no data in the window are dropped. This is the survivorship-
    free dataset: delisted coins are present for exactly their trading life.
    """
    months = _months_in_range(start_ms, end_ms)
    log(f"Archive: {len(symbols)} symbols x up to {len(months)} months, {workers} workers")
    result: Dict[str, pd.DataFrame] = {}
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_range_archive, s, start_ms, end_ms): s for s in symbols}
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
    log(f"Archive usable symbols: {len(result)}")
    return result
