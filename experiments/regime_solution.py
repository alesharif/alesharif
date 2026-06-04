#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""THE solution test: a SLOW (90-day) regime gate to sit out the 2025 drought.

Root cause (proven): runner supply collapsed in 2025. Solution: only trade when
the market is in a runner-rich era, detected on a SLOW gauge (user's larger-
timeframe insight). Two gates, lagged (no lookahead):
  BTC90    : BTC close > BTC close 90 days ago (quarterly uptrend)
  SUPPLY   : trailing-60d market runner-rate (% liquid coins +50% in last 60d) >= thr
Goal: does the gate turn 2025's losses into 'sit out' while keeping 2023-2024 wins?
Developed system, TP+100, SL10, liquid. Run:  python experiments/regime_solution.py
"""

from __future__ import annotations

import gc, sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from binance_sim import pit_universe as PIT               # noqa: E402

SL = 0.10; TP = 1.0; WINDOW_MS = 90*24*3600*1000; LIQ_MIN = 300_000; DAY = 86400000
S, E, SF = "2022-06-01", "2026-06-01", "2022-09-01"
POS_W = 0.125


def ms(s): return int(pd.Timestamp(s, tz="UTC").timestamp()*1000)
def ema(a, n): return pd.Series(a).ewm(span=n, adjust=False).mean().to_numpy()


def main():
    print("Regime SOLUTION: 90-day gate to sit out the 2025 drought\n", flush=True)
    raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), ms(S), ms(E), log=lambda *a: None)

    # BTC daily closes
    btc = raw["BTCUSDT"]
    bd = pd.Series(btc["close"].to_numpy(float), index=pd.to_datetime(btc["time"], unit="ms")).resample("D").last().dropna()
    btc_t = np.asarray(bd.index.astype("int64")) // 10**6; btc_c = bd.to_numpy()

    def btc90(t):
        i = np.searchsorted(btc_t, t, side="left") - 1
        if i < 90:
            return False
        return btc_c[i] > btc_c[i-90]

    # trailing 60d runner-rate series (market-wide)
    runs = {}; tots = {}
    trades = []
    for sym, df in raw.items():
        df = df.sort_values("time").reset_index(drop=True)
        df["dt"] = pd.to_datetime(df["time"], unit="ms")
        g = df.set_index("dt")
        d = pd.DataFrame({"o": g["open"].resample("D").first(), "h": g["high"].resample("D").max(),
                          "l": g["low"].resample("D").min(), "c": g["close"].resample("D").last(),
                          "v": g["volume"].resample("D").sum(), "t": g["time"].resample("D").last()}).dropna()
        c = d["c"].to_numpy(); n = len(c)
        if n < 220:
            df.drop(columns=["dt"], inplace=True, errors="ignore"); continue
        o = d["o"].to_numpy(); v = d["v"].to_numpy(); t = d["t"].to_numpy(); dv = c*v
        for i in range(60, n):                       # trailing 60d runner contribution
            if np.nanmean(dv[max(0, i-30):i]) <= LIQ_MIN:
                continue
            day = int(t[i]) // DAY
            tots[day] = tots.get(day, 0) + 1
            if c[i] / c[i-60] - 1 >= 0.50:
                runs[day] = runs.get(day, 0) + 1
        sma = pd.Series(c).rolling(20).mean(); std = pd.Series(c).rolling(20).std()
        bbw = (4*std/sma).to_numpy(); q25 = pd.Series(bbw).rolling(100).quantile(0.25).to_numpy()
        e20 = ema(c, 20); e50 = ema(c, 50); e200 = ema(c, 200)
        macd = ema(c, 12)-ema(c, 26); sg = ema(macd, 9); hist = macd-sg
        T = df["time"].to_numpy(); H = df["high"].to_numpy(float); L = df["low"].to_numpy(float); C = df["close"].to_numpy(float)
        for i in range(200, n):
            if not (np.isfinite(q25[i-1]) and bbw[i-1] <= q25[i-1] and bbw[i] > bbw[i-1]
                    and hist[i] > hist[i-1] > hist[i-2] and macd[i] > macd[i-1]
                    and abs(e20[i]-e50[i])/c[i] < 0.03 and c[i] > o[i]
                    and int(t[i]) >= ms(SF) and c[i] > e200[i]):
                continue
            if np.nanmean(dv[max(0, i-30):i]) <= LIQ_MIN:
                continue
            ent_t, ent_px = int(t[i]), float(c[i]); j0 = np.searchsorted(T, ent_t)
            if j0 >= len(T):
                continue
            sl = ent_px*(1-SL); tp = ent_px*(1+TP); end_t = ent_t+WINDOW_MS
            r = None; j = j0
            while j < len(T) and T[j] <= end_t:
                if L[j] <= sl: r = (-SL*100, True); break
                if H[j] >= tp: r = (TP*100, False); break
                j += 1
            if r is None:
                r = ((C[min(j, len(T)-1)]/ent_px-1)*100, False)
            nr = r[0] - 1.0 - (1.0 if r[1] else 0.0)
            trades.append((ent_t, nr, int(pd.Timestamp(ent_t, unit="ms").strftime("%Y"))))
        df.drop(columns=["dt"], inplace=True, errors="ignore")
    del raw; gc.collect()

    days = sorted(tots); sup_t = np.array([dd*DAY for dd in days])
    sup_v = np.array([runs.get(dd, 0)/tots[dd]*100 for dd in days])
    def supply(t):
        i = np.searchsorted(sup_t, t, side="left") - 1
        return sup_v[i] if i >= 0 else np.nan
    thr = np.nanmedian([supply(et) for et, _, _ in trades])

    def peryear(name, sel):
        yrs = {}
        for _, nr, y in sel:
            yrs.setdefault(y, []).append(nr)
        line = f"{name:<16}"
        for y in [2023, 2024, 2025, 2026]:
            a = yrs.get(y, [])
            line += f"{(np.sum(a)*POS_W if a else 0):>+7.0f}%({len(a)})"
        allr = np.array([x[1] for x in sel])
        line += f"  | exp {allr.mean() if len(allr) else 0:+.1f}% n{len(sel)}"
        print(line)

    print(f"\n{'config':<16}{'2023':>11}{'2024':>11}{'2025':>11}{'2026':>11}")
    print("-" * 76)
    peryear("BASE (no gate)", trades)
    peryear("BTC90 gate", [x for x in trades if btc90(x[0])])
    peryear(f"SUPPLY>{thr:.0f}%", [x for x in trades if supply(x[0]) >= thr])
    peryear("BTC90 & SUPPLY", [x for x in trades if btc90(x[0]) and supply(x[0]) >= thr])
    print("\nالهدف: تبقى أرباح 2023-2024 وتختفي/تتقلّص خسائر 2025 (الجلوس خارج السوق).")
    print("الأرقام = ربح السنة (وزن 0.125) و(عدد الصفقات).")
    print("\nDONE_SOLUTION.", flush=True)


if __name__ == "__main__":
    main()
