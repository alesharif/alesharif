#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Optimize the drought mean-reversion exit. Best entry = weekly-RSI cross>30 +
daily close>EMA10. Sweep TP x SL (first-touch, 60d) to maximize the 2025 drought
edge. Net of 1% round-trip + 1% stop slippage. Run: python experiments/mr_optimize.py
"""

from __future__ import annotations

import gc, sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from binance_sim import pit_universe as PIT               # noqa: E402

WINDOW_MS = 60*24*3600*1000; LIQ_MIN = 300_000
S, E, SF = "2022-06-01", "2026-06-01", "2022-09-01"
OS = 30
TPS = [0.10, 0.15, 0.20, 0.25, 0.30]; SLS = [0.08, 0.10, 0.12, 0.15, 0.20]
COST = 1.0; STOP_SLIP = 1.0


def ms(s): return int(pd.Timestamp(s, tz="UTC").timestamp()*1000)
def ema(a, n): return pd.Series(a).ewm(span=n, adjust=False).mean().to_numpy()


def rsi(c, n=14):
    d = np.diff(c, prepend=c[0]); up = np.where(d > 0, d, 0.0); dn = np.where(d < 0, -d, 0.0)
    ru = pd.Series(up).ewm(alpha=1/n, adjust=False).mean().to_numpy()
    rd = pd.Series(dn).ewm(alpha=1/n, adjust=False).mean().to_numpy()
    return 100 - 100/(1 + ru/np.where(rd == 0, 1e-9, rd))


def main():
    print(f"Optimize drought MR exit: entry weekly-RSI>{OS} + daily>EMA10, TPxSL grid\n", flush=True)
    raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), ms(S), ms(E), log=lambda *a: None)
    res = {(tp, sl): {"full": [], 2024: [], 2025: []} for tp in TPS for sl in SLS}
    nsig = 0
    for sym, df in raw.items():
        df = df.sort_values("time").reset_index(drop=True)
        df["dt"] = pd.to_datetime(df["time"], unit="ms")
        g = df.set_index("dt")
        wk = pd.DataFrame({"c": g["close"].resample("W").last(), "v": g["volume"].resample("W").sum(),
                           "t": g["time"].resample("W").last()}).dropna()
        dl = pd.DataFrame({"c": g["close"].resample("D").last(), "t": g["time"].resample("D").last()}).dropna()
        wc = wk["c"].to_numpy(); wt = wk["t"].to_numpy(); wv = wk["v"].to_numpy()
        if len(wc) < 25 or len(dl) < 30:
            df.drop(columns=["dt"], inplace=True, errors="ignore"); continue
        wr = rsi(wc, 14); wdv = wc*wv
        dc = dl["c"].to_numpy(); dtt = dl["t"].to_numpy(); dema = ema(dc, 10)
        T = df["time"].to_numpy(); H = df["high"].to_numpy(float); L = df["low"].to_numpy(float); C = df["close"].to_numpy(float)
        for w in range(15, len(wc)):
            if not (wr[w-1] <= OS and wr[w] > OS and int(wt[w]) >= ms(SF)):
                continue
            if np.nanmean(wdv[max(0, w-4):w]) <= LIQ_MIN:
                continue
            ent_t, ent_px = int(wt[w]), float(wc[w]); j0 = np.searchsorted(T, ent_t)
            if j0 >= len(T):
                continue
            di = np.searchsorted(dtt, ent_t, side="right") - 1
            if di < 1 or not (dc[di] > dema[di]):        # daily confirm
                continue
            nsig += 1
            yr = int(pd.Timestamp(ent_t, unit="ms").strftime("%Y"))
            end = min(j0 + int(np.searchsorted(T[j0:], ent_t+WINDOW_MS)) + 1, len(T))
            Hs = H[j0:end]; Ls = L[j0:end]; lastc = C[end-1] if end > j0 else ent_px
            for tp in TPS:
                hb = Hs >= ent_px*(1+tp); tph = int(np.argmax(hb)) if hb.any() else 10**9
                for sl in SLS:
                    lb = Ls <= ent_px*(1-sl); slh = int(np.argmax(lb)) if lb.any() else 10**9
                    if slh <= tph and slh < 10**9:
                        nr = -sl*100 - COST - STOP_SLIP
                    elif tph < 10**9:
                        nr = tp*100 - COST
                    else:
                        nr = (lastc/ent_px - 1)*100 - COST
                    res[(tp, sl)]["full"].append(nr)
                    if yr in (2024, 2025):
                        res[(tp, sl)][yr].append(nr)
        df.drop(columns=["dt"], inplace=True, errors="ignore")
    del raw; gc.collect()

    print(f"  {nsig} entry signals\n")
    print("2025 (drought) expectancy grid  [rows=TP, cols=SL]:")
    print("TP\\SL " + "".join(f"{int(s*100):>8}%" for s in SLS))
    best = None
    for tp in TPS:
        row = f"{int(tp*100):>4}% "
        for sl in SLS:
            a = np.array(res[(tp, sl)][2025]); m = a.mean() if len(a) else float("nan")
            row += f"{m:>+8.1f}"
            if best is None or m > best[0]:
                best = (m, tp, sl)
        print(row)
    bm, btp, bsl = best
    r = res[(btp, bsl)]
    wr2 = (np.array(r["full"]) > 0).mean()*100
    print(f"\nBEST 2025: TP{int(btp*100)}%/SL{int(bsl*100)}% -> 2025 {bm:+.1f}%, "
          f"FULL {np.mean(r['full']):+.1f}%, 2024 {np.mean(r[2024]):+.1f}%, "
          f"win {wr2:.0f}%, n {len(r['full'])}")
    print("\nDONE_MROPT.", flush=True)


if __name__ == "__main__":
    main()
