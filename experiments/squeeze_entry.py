#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Closest mechanical PROXY of the user's FULL system, with his 10% stop.

User's real entry (discretionary, Wobbler code locked -> can't replicate exactly):
  volatility SQUEEZE then EXPANSION + MACD curling up + EMAs converging +
  small bullish candles = 'buying has started', enter, SL 10%, take +10..50%.

Proxy on DAILY candles:
  * squeeze->expansion: Bollinger bandwidth was in its bottom 25% (last 100d)
    and is now rising (just started expanding)
  * MACD histogram curling up (rising 2 bars) and MACD line rising
  * EMA20/EMA50 converged: |ema20-ema50|/close < 3%
  * bullish candle (close>open)
Then SL = -10%, TP first-touch grid (+10/20/30/50) on 4h data, 90-day window.
Reports win% and full expectancy. Run:  python experiments/squeeze_entry.py
"""

from __future__ import annotations

import gc, sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from binance_sim import pit_universe as PIT               # noqa: E402

START = "2022-06-01"; END = "2026-06-01"
DS = int(pd.Timestamp(START, tz="UTC").timestamp()*1000)
DE = int(pd.Timestamp(END, tz="UTC").timestamp()*1000)
SL = 0.10; WINDOW_MS = 90*24*3600*1000; FEE = 0.2
TPS = [0.10, 0.20, 0.30, 0.50]


def resample_d(df):
    g = df.set_index("dt")
    return pd.DataFrame({"o": g["open"].resample("D").first(),
                         "h": g["high"].resample("D").max(),
                         "l": g["low"].resample("D").min(),
                         "c": g["close"].resample("D").last(),
                         "t": g["time"].resample("D").last()}).dropna()


def ema(a, n): return pd.Series(a).ewm(span=n, adjust=False).mean().to_numpy()


def entries(d):
    o = d["o"].to_numpy(); c = d["c"].to_numpy(); t = d["t"].to_numpy()
    n = len(c)
    if n < 120:
        return []
    sma = pd.Series(c).rolling(20).mean(); std = pd.Series(c).rolling(20).std()
    bbw = (4 * std / sma).to_numpy()
    q25 = pd.Series(bbw).rolling(100).quantile(0.25).to_numpy()
    e20 = ema(c, 20); e50 = ema(c, 50)
    macd = ema(c, 12) - ema(c, 26); sig = ema(macd, 9); hist = macd - sig
    out = []
    for i in range(100, n):
        squeeze_prev = bbw[i-1] <= q25[i-1] if np.isfinite(q25[i-1]) else False
        expanding = bbw[i] > bbw[i-1]
        macd_up = (hist[i] > hist[i-1] > hist[i-2]) and (macd[i] > macd[i-1])
        conv = abs(e20[i] - e50[i]) / c[i] < 0.03 if c[i] > 0 else False
        bull = c[i] > o[i]
        if squeeze_prev and expanding and macd_up and conv and bull:
            out.append((int(t[i]), float(c[i])))
    return out


def main():
    print("SQUEEZE+confirmation entry (proxy of user's full system) + SL 10%\n", flush=True)
    raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), DS, DE, log=lambda *a: None)
    sigs = {}
    for sym, df in raw.items():
        df = df.sort_values("time").reset_index(drop=True)
        df["dt"] = pd.to_datetime(df["time"], unit="ms")
        d = resample_d(df)
        e = entries(d)
        if e:
            sigs[sym] = e
        df.drop(columns=["dt"], inplace=True)
    nsig = sum(len(v) for v in sigs.values())
    print(f"  {nsig} entry signals (squeeze->expansion + MACD up + EMA conv + bull)\n", flush=True)
    if nsig == 0:
        print("no signals."); return

    print(f"{'TP':<7}{'n':>6}{'TP%':>7}{'stop%':>7}{'timeo%':>8}{'WIN%':>7}{'mean(exp)':>11}{'median':>9}")
    print("-" * 62)
    for tp in TPS:
        rets = []
        for sym, lst in sigs.items():
            df = raw[sym]; t = df["time"].to_numpy()
            H = df["high"].to_numpy(float); L = df["low"].to_numpy(float); C = df["close"].to_numpy(float)
            for ent_t, ent_px in lst:
                j0 = np.searchsorted(t, ent_t)
                if j0 >= len(t):
                    continue
                tp_px = ent_px*(1+tp); sl_px = ent_px*(1-SL); end_t = ent_t+WINDOW_MS
                r = None; j = j0
                while j < len(t) and t[j] <= end_t:
                    if L[j] <= sl_px:
                        r = -SL*100 - FEE; break
                    if H[j] >= tp_px:
                        r = tp*100 - FEE; break
                    j += 1
                if r is None:
                    r = (C[min(j, len(t)-1)]/ent_px - 1)*100 - FEE
                rets.append(r)
        a = np.array(rets)
        tp_hit = (np.isclose(a, tp*100-FEE)).mean()*100
        st_hit = (np.isclose(a, -SL*100-FEE)).mean()*100
        print(f"+{int(tp*100)}%{'':<3}{len(a):>6}{tp_hit:>6.0f}%{st_hit:>6.0f}%"
              f"{100-tp_hit-st_hit:>7.0f}%{(a>0).mean()*100:>6.0f}%{a.mean():>+10.1f}%{np.median(a):>+8.1f}%")
    del raw; gc.collect()
    print("\nmean>0 => النظام (الوكيل) رابح حتى برسوم 0.2%. ثم نختبر الانزلاق الواقعي.")
    print("ملاحظة: هذا وكيل ميكانيكي؛ تقديرك البشري قد يكون أفضل (لا يُبرمَج).")
    print("\nDONE_SQUEEZE.", flush=True)


if __name__ == "__main__":
    main()
