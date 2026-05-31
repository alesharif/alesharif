#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Faithful port of the "paper_bot_v2" strategy logic (signal + indicators).

This module contains ONLY the deterministic, side-effect-free parts of the
original live bot: the technical indicators, the feature computation and the
entry-signal test. The network code, Telegram credentials and live loop have
been deliberately removed so this can be imported safely and used for
historical backtesting.

Strategy summary (4h timeframe, Binance Spot, long-only, multi-symbol):
  Signal  : ema_order >= 0.75  AND  rsi < 75  AND  calm_long > rolling-80th-pctl
  Filters : price > EMA200, MACD(12,26) > 0, ADX >= 25,
            point-in-time 30d quote-volume in [2M, 200M], ATR > 0
  Exit    : hard stop = entry - 1.5*ATR
            trailing  = peak - 0.2*ATR, activated after price moves +0.2*ATR
  Universe: all active USDT spot pairs except stablecoins / fiat / metals /
            wrapped / leveraged tokens.

All features are *causal* (each value at index T depends only on data <= T),
so computing them once over a full history is equivalent to a point-in-time
walk-forward — i.e. zero look-ahead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

np.seterr(divide="ignore", invalid="ignore")

# ════════════════════════════ STRATEGY CONFIG ═══════════════════════
# Portfolio
INITIAL_CAPITAL = 2000.0
POSITION_PCT = 0.125          # 12.5% per trade (8 trades = 100%)
MAX_CONCURRENT = 8
FEE_RT = 0.2                  # round-trip fee, percent

# Entry signal
EMA_ORDER_MIN = 0.75
RSI_MAX = 75.0
CALM_LOOKBACK = 200
CALM_PCTL = 80.0
CALM_MIN_HISTORY = 80
EMA_SPANS = [5, 10, 20, 50, 99]
LONG_WINDOW = 20
ADX_MIN = 25.0
STABLE_RATIO_MAX = 1.3
ATR_PERIOD = 14
MIN_HISTORY = 210            # need 200 for EMA200

# Exit
SL_ATR = 1.5
TRAIL_ATR = 0.2
ACTIVATE_ATR = 0.2
MAX_HOLD = 60               # defined in the original bot but NOT enforced there

# Point-in-time volume filter
VOL_WINDOW_DAYS = 30
VOL_BARS = VOL_WINDOW_DAYS * 6
VOLUME_MIN_USD = 2_000_000
VOLUME_MAX_USD = 200_000_000

ENTRY_TF = "4h"

# ════════════════════════════ EXCLUSION LISTS ═══════════════════════
EXCLUDED_STABLECOINS = {
    'USDT', 'USDC', 'BUSD', 'DAI', 'TUSD', 'USDP', 'USDD', 'FDUSD', 'GUSD', 'LUSD',
    'FRAX', 'MIM', 'SUSD', 'EURS', 'EURT', 'PYUSD', 'USDE', 'USDK', 'USDN', 'HUSD',
    'UST', 'MUSD', 'USDJ', 'AUSD', 'CUSD', 'USDX', 'USDB', 'FEI', 'TRIBE', 'USTC',
    'VAI', 'OUSD', 'ZUSD', 'BIDR', 'BVND', 'IDRT', 'NGN', 'USDSB', 'USDS', 'XUSD',
    'EURC', 'CEUR', 'JEUR', 'AGEUR', 'USDTB', 'USD1', 'BFUSD',
}
EXCLUDED_FIAT = {
    'EUR', 'GBP', 'JPY', 'CNY', 'AUD', 'CAD', 'CHF', 'KRW', 'RUB', 'TRY', 'BRL', 'INR',
    'MXN', 'ZAR', 'ARS', 'PLN', 'SEK', 'NOK', 'DKK', 'HKD', 'SGD', 'NZD', 'THB', 'IDR',
    'PHP', 'MYR', 'VND', 'TWD', 'AED', 'SAR', 'EGP', 'ILS', 'CZK', 'HUF', 'RON', 'BGN',
    'UAH', 'BYN', 'KZT', 'AZN', 'GEL', 'AMD', 'MDL', 'RSD',
}
EXCLUDED_METALS = {
    'PAXG', 'XAUT', 'KAU', 'KAG', 'DGX', 'CACHE', 'AABB', 'GLD', 'XAGT', 'DGLD', 'AUX',
    'TXAU', 'TXAG', 'GOLD', 'SILVER', 'PLT', 'PALL',
}
EXCLUDED_WRAPPED = {
    'WBTC', 'WETH', 'STETH', 'CBETH', 'RETH', 'FRXETH', 'WSTETH', 'ANKRETH', 'SETH',
    'WBNB', 'WMATIC', 'WAVAX', 'RENBTC', 'HBTC', 'BTCB', 'WBETH',
}
LEVERAGE_PATTERNS = ['3L', '3S', '5L', '5S', '2L', '2S', '4L', '4S', 'BULL', 'BEAR', 'HALF', 'HEDGE']
LEVERAGE_SUFFIXES = ['UP', 'DOWN']


def should_exclude(base: str):
    """Return the exclusion reason for a base asset, or None if tradeable."""
    b = base.upper()
    if b in EXCLUDED_STABLECOINS:
        return "stablecoin"
    if b in EXCLUDED_FIAT:
        return "fiat"
    if b in EXCLUDED_METALS:
        return "metal"
    if b in EXCLUDED_WRAPPED:
        return "wrapped"
    for p in LEVERAGE_PATTERNS:
        if b.endswith(p) and b[:-len(p)] in {'BTC', 'ETH', 'BNB', 'ADA', 'XRP',
                'DOT', 'LINK', 'LTC', 'SOL', 'DOGE', 'MATIC', 'AVAX', 'UNI', 'SUSHI'}:
            return "leverage"
    for s in LEVERAGE_SUFFIXES:
        if b.endswith(s) and len(b) > len(s):
            return "leverage"
    return None


# ════════════════════════════ INDICATORS ════════════════════════════
def ema_series(a, span):
    return pd.Series(a).ewm(span=span, adjust=False).mean().to_numpy()


def rsi_series(c, period=14):
    d = np.diff(c, prepend=c[0])
    up = np.where(d > 0, d, 0.0)
    dn = np.where(d < 0, -d, 0.0)
    ru = pd.Series(up).ewm(alpha=1 / period, adjust=False).mean().to_numpy()
    rd = pd.Series(dn).ewm(alpha=1 / period, adjust=False).mean().to_numpy()
    rs = ru / np.where(rd > 1e-12, rd, np.nan)
    return 100 - 100 / (1 + rs)


def atr_series(h, l, c, period=14):
    pc = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])
    return pd.Series(tr).ewm(alpha=1 / period, adjust=False).mean().to_numpy()


def compute_adx(h, l, c, period=14):
    n = len(c)
    if n < period * 2:
        return np.full(n, np.nan)
    up = np.zeros(n)
    dn = np.zeros(n)
    up[1:] = h[1:] - h[:-1]
    dn[1:] = l[:-1] - l[1:]
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    pc = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])
    atr = pd.Series(tr).ewm(alpha=1 / period, adjust=False).mean().to_numpy()
    pdi = 100 * pd.Series(plus_dm).ewm(alpha=1 / period, adjust=False).mean().to_numpy() / np.where(atr > 1e-12, atr, np.nan)
    mdi = 100 * pd.Series(minus_dm).ewm(alpha=1 / period, adjust=False).mean().to_numpy() / np.where(atr > 1e-12, atr, np.nan)
    dx = 100 * np.abs(pdi - mdi) / np.where((pdi + mdi) > 1e-12, pdi + mdi, np.nan)
    return pd.Series(np.nan_to_num(dx)).ewm(alpha=1 / period, adjust=False).mean().to_numpy()


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all indicator columns on 4h candles. Zero look-ahead."""
    c = df["close"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    eps = 1e-12
    n = len(df)
    emas = {s: ema_series(c, s) for s in EMA_SPANS}
    order = np.zeros(n)
    for a, b_ in [(5, 10), (10, 20), (20, 50), (50, 99)]:
        order += (emas[a] > emas[b_]).astype(float)
    df["ema_order"] = order / 4.0
    df["rsi"] = rsi_series(c, 14)
    df["atr"] = atr_series(h, l, c, ATR_PERIOD)
    ret = np.concatenate([[np.nan], np.diff(c) / np.where(c[:-1] > eps, c[:-1], np.nan)])
    df["calm_long"] = pd.Series(np.abs(ret)).rolling(LONG_WINDOW).mean().shift(1).to_numpy() * 100
    ema200 = ema_series(c, 200)
    df["above_ema200"] = (c > ema200).astype(float)
    macd = ema_series(c, 12) - ema_series(c, 26)
    df["macd_pos"] = (macd > 0).astype(float)
    df["adx"] = compute_adx(h, l, c, 14)
    df["vol_pit"] = pd.Series(df["quote_av"].to_numpy(float)).rolling(VOL_BARS).mean().shift(1).to_numpy() * 6
    df["cs_spread"] = corwin_schultz_spread(h, l, window=20)
    return df


def corwin_schultz_spread(high, low, window: int = 20):
    """Corwin & Schultz (2012) bid-ask spread estimate from high/low prices.

    Returns the estimated spread as a *fraction* of price, smoothed with a
    causal rolling mean. Lets us approximate per-coin spread historically
    (real spread is absent from kline data). Zero look-ahead.
    """
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    n = len(h)
    out = np.full(n, np.nan)
    if n < 2:
        return out
    with np.errstate(divide="ignore", invalid="ignore"):
        hl = np.log(h / l) ** 2                       # single-bar squared log range
        h2 = np.maximum(h[1:], h[:-1])               # 2-bar high
        l2 = np.minimum(l[1:], l[:-1])               # 2-bar low
        beta = hl[1:] + hl[:-1]
        gamma = np.log(h2 / l2) ** 2
        k = 3.0 - 2.0 * np.sqrt(2.0)
        alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
        s = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    s = np.where(np.isfinite(s) & (s > 0), s, 0.0)   # CS convention: clamp negatives
    out[1:] = s
    return pd.Series(out).rolling(window, min_periods=max(2, window // 4)).mean().to_numpy()


def entry_ok_at(df: pd.DataFrame, T: int):
    """Return (signal_bool, atr_value) for the candle at integer index T.

    Mirrors the original ``check_entry_signal`` exactly, but lets the caller
    choose which completed candle to evaluate (needed for a historical walk).
    """
    if T < MIN_HISTORY - 1:
        return False, None
    eo = df["ema_order"].to_numpy()
    rv = df["rsi"].to_numpy()
    cl = df["calm_long"].to_numpy()
    ab = df["above_ema200"].to_numpy()
    mp = df["macd_pos"].to_numpy()
    adx = df["adx"].to_numpy()
    vol = df["vol_pit"].to_numpy()
    atr = df["atr"].to_numpy()
    thr = pd.Series(cl).shift(1).rolling(
        CALM_LOOKBACK, min_periods=CALM_MIN_HISTORY
    ).quantile(CALM_PCTL / 100.0).to_numpy()
    ok = (eo[T] >= EMA_ORDER_MIN and rv[T] < RSI_MAX and np.isfinite(cl[T]) and
          np.isfinite(thr[T]) and cl[T] > thr[T] and ab[T] > 0.5 and mp[T] > 0.5 and
          np.isfinite(adx[T]) and adx[T] >= ADX_MIN and
          np.isfinite(vol[T]) and VOLUME_MIN_USD <= vol[T] <= VOLUME_MAX_USD and
          np.isfinite(atr[T]) and atr[T] > 0)
    return bool(ok), (float(atr[T]) if ok else None)


def attach_entry_signal(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorised entry-signal column for the whole history (zero look-ahead)."""
    cl = df["calm_long"].to_numpy()
    thr = pd.Series(cl).shift(1).rolling(
        CALM_LOOKBACK, min_periods=CALM_MIN_HISTORY
    ).quantile(CALM_PCTL / 100.0).to_numpy()
    df["calm_thr"] = thr
    eo = df["ema_order"].to_numpy()
    rv = df["rsi"].to_numpy()
    ab = df["above_ema200"].to_numpy()
    mp = df["macd_pos"].to_numpy()
    adx = df["adx"].to_numpy()
    vol = df["vol_pit"].to_numpy()
    atr = df["atr"].to_numpy()
    sig = (
        (eo >= EMA_ORDER_MIN) & (rv < RSI_MAX) & np.isfinite(cl) &
        np.isfinite(thr) & (cl > thr) & (ab > 0.5) & (mp > 0.5) &
        np.isfinite(adx) & (adx >= ADX_MIN) &
        np.isfinite(vol) & (vol >= VOLUME_MIN_USD) & (vol <= VOLUME_MAX_USD) &
        np.isfinite(atr) & (atr > 0)
    )
    # before MIN_HISTORY candles the signal is never valid
    if len(sig) >= MIN_HISTORY:
        sig[:MIN_HISTORY - 1] = False
    else:
        sig[:] = False
    df["entry_signal"] = sig
    return df
