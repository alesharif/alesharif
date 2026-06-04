#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Does a 99-day coin filter help the drought mean-reversion?
Entry = weekly-RSI>30 + daily>EMA10 + TP15/SL15 (best MR). Filter variants on the
coin's 99-day return at entry (ret99):
  none | up99 (ret99>=0, strict) | not<-30% | not<-50% | ONLY<-50% (deep)
Tension: MR buys oversold (down) coins, so 'up99' may kill signals; 'not dead'
versions test avoiding only catastrophic decline. Per regime + n + win%.
Run:  python experiments/mr_99filter.py
"""

from __future__ import annotations

import gc, sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from binance_sim import pit_universe as PIT               # noqa: E402

WINDOW_MS = 60*24*3600*1000; LIQ_MIN = 300_000; TP = 0.15; SL = 0.15; OS = 30
S, E, SF = "2022-06-01", "2026-06-01", "2022-09-01"
COST = 1.0; STOP_SLIP = 1.0
FILTS = ["none", "up99", "not<-30%", "not<-50%", "ONLY<-50%"]


def ms(s): return int(pd.Timestamp(s, tz="UTC").timestamp()*1000)
def ema(a, n): return pd.Series(a).ewm(span=n, adjust=False).mean().to_numpy()


def rsi(c, n=14):
    d = np.diff(c, prepend=c[0]); up = np.where(d > 0, d, 0.0); dn = np.where(d < 0, -d, 0.0)
    ru = pd.Series(up).ewm(alpha=1/n, adjust=False).mean().to_numpy()
    rd = pd.Series(dn).ewm(alpha=1/n, adjust=False).mean().to_numpy()
    return 100 - 100/(1 + ru/np.where(rd == 0, 1e-9, rd))


def main():
    print("Drought MR + 99-day coin filter variants (entry RSI>30 + daily>EMA10, TP15/SL15)\n", flush=True)
    raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), ms(S), ms(E), log=lambda *a: None)
    res = {f: {"full": [], 2024: [], 2025: []} for f in FILTS}
    for sym, df in raw.items():
        df = df.sort_values("time").reset_index(drop=True)
        df["dt"] = pd.to_datetime(df["time"], unit="ms")
        g = df.set_index("dt")
        wk = pd.DataFrame({"c": g["close"].resample("W").last(), "v": g["volume"].resample("W").sum(),
                           "t": g["time"].resample("W").last()}).dropna()
        dl = pd.DataFrame({"c": g["close"].resample("D").last(), "t": g["time"].resample("D").last()}).dropna()
        wc = wk["c"].to_numpy(); wt = wk["t"].to_numpy(); wv = wk["v"].to_numpy()
        if len(wc) < 25 or len(dl) < 110:
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
            if di < 99 or not (dc[di] > dema[di]):
                continue
            ret99 = dc[di]/dc[di-99] - 1
            yr = int(pd.Timestamp(ent_t, unit="ms").strftime("%Y"))
            end = min(j0 + int(np.searchsorted(T[j0:], ent_t+WINDOW_MS)) + 1, len(T))
            Hs = H[j0:end]; Ls = L[j0:end]; lastc = C[end-1] if end > j0 else ent_px
            hb = Hs >= ent_px*(1+TP); lb = Ls <= ent_px*(1-SL)
            tph = int(np.argmax(hb)) if hb.any() else 10**9
            slh = int(np.argmax(lb)) if lb.any() else 10**9
            if slh <= tph and slh < 10**9:
                nr = -SL*100 - COST - STOP_SLIP
            elif tph < 10**9:
                nr = TP*100 - COST
            else:
                nr = (lastc/ent_px - 1)*100 - COST
            passes = {"none": True, "up99": ret99 >= 0, "not<-30%": ret99 >= -0.30,
                      "not<-50%": ret99 >= -0.50, "ONLY<-50%": ret99 < -0.50}
            for f in FILTS:
                if passes[f]:
                    res[f]["full"].append(nr)
                    if yr in (2024, 2025):
                        res[f][yr].append(nr)
        df.drop(columns=["dt"], inplace=True, errors="ignore")
    del raw; gc.collect()

    print(f"{'filter':<12}{'n':>7}{'FULL':>9}{'2024':>9}{'2025':>9}{'win%':>7}")
    print("-" * 52)
    for f in FILTS:
        r = res[f]
        def m(k):
            a = np.array(r[k]); return a.mean() if len(a) else float("nan")
        nn = len(r["full"]); wr2 = (np.array(r["full"]) > 0).mean()*100 if nn else float("nan")
        mark = " *" if m(2025) > 0 else ""
        print(f"{f:<12}{nn:>7}{m('full'):>+8.1f}%{m(2024):>+8.1f}%{m(2025):>+8.1f}%{wr2:>6.0f}%{mark}")
    print("\nهل تجنّب العملات الميتة (not<-X%) يرفع 2025 فوق +3.8%؟")
    print("\nDONE_MR99.", flush=True)


if __name__ == "__main__":
    main()
