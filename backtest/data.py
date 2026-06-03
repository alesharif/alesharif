#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data.py — تنزيل وتخزين بيانات Binance التاريخية من المرايا العامة.

نتجنّب api.binance.com (محظور جغرافياً 451) ونستخدم:
  - data-api.binance.vision/api/v3/exchangeInfo  → قائمة الأزواج
  - data.binance.vision/data/spot/monthly/klines → شموع شهرية (ZIP/CSV)

ملاحظات مهمة:
  - منذ 2025-01-01 طوابع الوقت في dumps أصبحت بالميكروثانية (16 رقم)
    → نطبّعها إلى ميلي ثانية.
  - بعض ملفات 2025 تحتوي صف عناوين (header) → نكتشفه ونتجاهله.
"""

import io
import time
import zipfile
import calendar
from pathlib import Path
from datetime import datetime, timezone

import requests
import pandas as pd

CACHE_DIR = Path(__file__).resolve().parent / 'cache'
EXINFO_URL = 'https://data-api.binance.vision/api/v3/exchangeInfo'
VISION_BASE = 'https://data.binance.vision/data/spot/monthly/klines'

_KLINE_COLS = [
    'open_time', 'open', 'high', 'low', 'close', 'volume',
    'close_time', 'quote_volume', 'count',
    'taker_base', 'taker_quote', 'ignore',
]

_SESSION = requests.Session()
_SESSION.headers.update({'User-Agent': 'Mozilla/5.0 (backtest)'})


def _norm_ts(series: pd.Series) -> pd.Series:
    """طبّع طابع الوقت إلى ميلي ثانية (يكتشف µs > 1e14)."""
    s = pd.to_numeric(series, errors='coerce')
    # µs ~ 1.7e15 ، ms ~ 1.7e12
    return s.where(s < 1e14, s // 1000)


def list_usdt_pairs(timeout=30):
    """أزواج USDT الفورية المتداولة حالياً (current universe)."""
    r = _SESSION.get(EXINFO_URL, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    pairs = []
    for s in data.get('symbols', []):
        if (s.get('quoteAsset') == 'USDT'
                and s.get('status') == 'TRADING'
                and s.get('isSpotTradingAllowed', False)):
            pairs.append(s['symbol'])
    return sorted(pairs)


def _month_iter(start_dt, end_dt):
    """يولّد (year, month) من start إلى end شاملاً."""
    y, m = start_dt.year, start_dt.month
    while (y, m) <= (end_dt.year, end_dt.month):
        yield y, m
        if m == 12:
            y, m = y + 1, 1
        else:
            m += 1


def _download_month(symbol, interval, year, month, retries=3):
    """ينزّل شهراً واحداً → DataFrame أو None لو غير موجود (404)."""
    fname = f"{symbol}-{interval}-{year}-{month:02d}.zip"
    url = f"{VISION_BASE}/{symbol}/{interval}/{fname}"
    last_err = None
    for attempt in range(retries):
        try:
            r = _SESSION.get(url, timeout=60)
            if r.status_code == 404:
                return None  # الشهر غير متاح (عملة جديدة أو شطب)
            r.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
                csv_name = zf.namelist()[0]
                with zf.open(csv_name) as f:
                    raw = f.read().decode('utf-8')
            # كشف header: لو أول حقل ليس رقماً
            first_field = raw.lstrip().split(',', 1)[0].strip().strip('"')
            has_header = not first_field.replace('.', '').isdigit()
            df = pd.read_csv(
                io.StringIO(raw),
                header=0 if has_header else None,
                names=None if has_header else _KLINE_COLS,
            )
            if has_header:
                # نوحّد أسماء الأعمدة لأول 12 عمود
                df = df.iloc[:, :12]
                df.columns = _KLINE_COLS
            df['open_time'] = _norm_ts(df['open_time'])
            df['close_time'] = _norm_ts(df['close_time'])
            for col in ('open', 'high', 'low', 'close', 'volume', 'quote_volume'):
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df[['open_time', 'open', 'high', 'low', 'close',
                     'volume', 'close_time', 'quote_volume']].dropna()
            return df
        except Exception as e:
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"download failed {url}: {last_err}")


def get_klines_df(symbol, interval, start_dt, end_dt):
    """
    يُرجع DataFrame شموع [start_dt, end_dt) مرتّبة بالوقت، مع تخزين مؤقت
    شهري على القرص (parquet). open_time/close_time بالميلي ثانية.
    """
    cache_sub = CACHE_DIR / interval
    cache_sub.mkdir(parents=True, exist_ok=True)
    frames = []
    for y, m in _month_iter(start_dt, end_dt):
        pq = cache_sub / f"{symbol}-{interval}-{y}-{m:02d}.parquet"
        empty_marker = cache_sub / f"{symbol}-{interval}-{y}-{m:02d}.empty"
        if pq.exists():
            frames.append(pd.read_parquet(pq))
            continue
        if empty_marker.exists():
            continue
        df = _download_month(symbol, interval, y, m)
        if df is None or df.empty:
            empty_marker.touch()
            continue
        df.to_parquet(pq, index=False)
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=['open_time', 'open', 'high', 'low',
                                     'close', 'volume', 'close_time', 'quote_volume'])
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset='open_time').sort_values('open_time')
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)
    out = out[(out['open_time'] >= start_ms) & (out['open_time'] < end_ms)]
    return out.reset_index(drop=True)


def df_to_candles(df):
    """يحوّل DataFrame إلى list[dict] بصيغة البوت (لـ calc_signals/zigzag)."""
    return [
        {
            'open_time': int(r.open_time), 'open': float(r.open),
            'high': float(r.high), 'low': float(r.low), 'close': float(r.close),
            'volume': float(r.volume), 'close_time': int(r.close_time),
            'quote_volume': float(r.quote_volume),
        }
        for r in df.itertuples(index=False)
    ]


if __name__ == '__main__':
    # اختبار سريع
    pairs = list_usdt_pairs()
    print(f"USDT pairs: {len(pairs)} (sample: {pairs[:5]})")
    df = get_klines_df('BTCUSDT', '4h',
                       datetime(2025, 1, 1, tzinfo=timezone.utc),
                       datetime(2025, 2, 1, tzinfo=timezone.utc))
    print(f"BTCUSDT 4h Jan-2025: {len(df)} candles")
    print(df.head(2).to_string())
    print(df.tail(2).to_string())
