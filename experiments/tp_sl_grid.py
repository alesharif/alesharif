#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""User's redesign: make the strategy feed on SMALLER, abundant moves so it stays
profitable IN THE DROUGHT (2025) instead of needing rare +100% runners.

Squeeze breakout entry (+ EMA200 + liquid). Grid of TP x SL (first-touch, 90d).
NET expectancy (1% round-trip + 1% stop slippage) for: FULL | 2024 (bull) |
2025 (DROUGHT). Goal: find a (TP,SL) POSITIVE in 2025 -> strategy can grind
smaller moves and survive droughts. Run:  python experiments/tp_sl_grid.py
"""

from __future__ import annotations

import gc, sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from binance_sim import pit_universe as PIT               # noqa: E402

WINDOW_MS = 90*24*3600*1000; LIQ_MIN = 300_000
S, E, SF = "2022-06-01", "2026-06-01", "2022-09-01"
TPS = [0.20, 0.30, 0.40, 0.50, 1.00]
SLS = [0.05, 0.07, 0.10]
COST = 1.0; STOP_SLIP = 1.0


def ms(s): return int(pd.Timestamp(s, tz="UTC").timestamp()*1000)
def ema(a, n): return pd.Series(a).ewm(span=n, adjust=False).mean().to_numpy()


def main():
    print("TP x SL grid - can smaller targets stay profitable in the 2025 drought?\n", flush=True)
    raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), ms(S), ms(E), log=lambda *a: None)
    res = {(tp, sl): {"full": [], 2024: [], 2025: []} for tp in TPS for sl in SLS}

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
        o = d["o"].to_numpy(); v = d["v"].to_numpy(); t = d["t"].to_numpy()
        sma = pd.Series(c).rolling(20).mean(); std = pd.Series(c).rolling(20).std()
        bbw = (4*std/sma).to_numpy(); q25 = pd.Series(bbw).rolling(100).quantile(0.25).to_numpy()
        e20 = ema(c, 20); e50 = ema(c, 50); e200 = ema(c, 200)
        macd = ema(c, 12)-ema(c, 26); sg = ema(macd, 9); hist = macd-sg
        dv = c*v
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
            end = min(j0 + int(np.searchsorted(T[j0:], ent_t+WINDOW_MS)) + 1, len(T))
            Hs = H[j0:end]; Ls = L[j0:end]; lastc = C[end-1] if end > j0 else ent_px
            yr = int(pd.Timestamp(ent_t, unit="ms").strftime("%Y"))
            for tp in TPS:
                hb = Hs >= ent_px*(1+tp)
                tp_hit = int(np.argmax(hb)) if hb.any() else 10**9
                for sl in SLS:
                    lb = Ls <= ent_px*(1-sl)
                    sl_hit = int(np.argmax(lb)) if lb.any() else 10**9
                    if sl_hit <= tp_hit and sl_hit < 10**9:
                        nr = -sl*100 - COST - STOP_SLIP
                    elif tp_hit < 10**9:
                        nr = tp*100 - COST
                    else:
                        nr = (lastc/ent_px - 1)*100 - COST
                    res[(tp, sl)]["full"].append(nr)
                    if yr in (2024, 2025):
                        res[(tp, sl)][yr].append(nr)
        df.drop(columns=["dt"], inplace=True, errors="ignore")
    del raw; gc.collect()

    print(f"{'TP':>5}{'SL':>5}{'FULL':>9}{'2024':>9}{'2025(drought)':>15}")
    print("-" * 44)
    for tp in TPS:
        for sl in SLS:
            r = res[(tp, sl)]
            def m(k):
                a = np.array(r[k]); return a.mean() if len(a) else float("nan")
            mark = "  <== +2025" if m(2025) > 0 else ""
            print(f"{int(tp*100):>4}%{int(sl*100):>4}%{m('full'):>+8.1f}%{m(2024):>+8.1f}%{m(2025):>+13.1f}%{mark}")
    print("\nنبحث عن تهيئة موجبة في عمود 2025 (الجفاف) => الاستراتيجية تعيش على الحركات الصغيرة.")
    print("\nDONE_GRID.", flush=True)


if __name__ == "__main__":
    main()
