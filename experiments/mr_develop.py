#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Develop the drought mean-reversion: several weekly-RSI oversold thresholds x a
DAILY entry confirmation. Entry = weekly RSI crosses up from <=OS, confirmed on
the daily frame; exit = fixed TP +15% / SL -15% (the drought winner), 60d window.
Goal: raise the thin 2025 (+2%) edge. Per regime FULL/2024/2025 + win%.
Run:  python experiments/mr_develop.py
"""

from __future__ import annotations

import gc, sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from binance_sim import pit_universe as PIT               # noqa: E402

WINDOW_MS = 60*24*3600*1000; LIQ_MIN = 300_000; TP = 0.15; SL = 0.15
S, E, SF = "2022-06-01", "2026-06-01", "2022-09-01"
OSS = [25, 30, 35, 40]; CONFIRMS = ["none", "dRSIup", "d>EMA10"]
COST = 1.0; STOP_SLIP = 1.0


def ms(s): return int(pd.Timestamp(s, tz="UTC").timestamp()*1000)
def ema(a, n): return pd.Series(a).ewm(span=n, adjust=False).mean().to_numpy()


def rsi(c, n=14):
    d = np.diff(c, prepend=c[0]); up = np.where(d > 0, d, 0.0); dn = np.where(d < 0, -d, 0.0)
    ru = pd.Series(up).ewm(alpha=1/n, adjust=False).mean().to_numpy()
    rd = pd.Series(dn).ewm(alpha=1/n, adjust=False).mean().to_numpy()
    return 100 - 100/(1 + ru/np.where(rd == 0, 1e-9, rd))


def main():
    print("Develop drought MR: weekly-RSI OS thresholds x daily confirm (TP15/SL15)\n", flush=True)
    raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), ms(S), ms(E), log=lambda *a: None)
    res = {(os_, cf): {"full": [], 2024: [], 2025: []} for os_ in OSS for cf in CONFIRMS}

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
        dc = dl["c"].to_numpy(); dt = dl["t"].to_numpy(); drsi = rsi(dc, 14); dema = ema(dc, 10)
        T = df["time"].to_numpy(); H = df["high"].to_numpy(float); L = df["low"].to_numpy(float); C = df["close"].to_numpy(float)
        for w in range(15, len(wc)):
            if int(wt[w]) < ms(SF) or np.nanmean(wdv[max(0, w-4):w]) <= LIQ_MIN:
                continue
            ent_t, ent_px = int(wt[w]), float(wc[w]); j0 = np.searchsorted(T, ent_t)
            if j0 >= len(T):
                continue
            di = np.searchsorted(dt, ent_t, side="right") - 1
            if di < 1:
                continue
            conf = {"none": True,
                    "dRSIup": drsi[di] > drsi[di-1],
                    "d>EMA10": dc[di] > dema[di]}
            yr = int(pd.Timestamp(ent_t, unit="ms").strftime("%Y"))
            # outcome (TP/SL first-touch, fixed) computed once
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
            for os_ in OSS:
                if not (wr[w-1] <= os_ and wr[w] > os_):
                    continue
                for cf in CONFIRMS:
                    if not conf[cf]:
                        continue
                    res[(os_, cf)]["full"].append(nr)
                    if yr in (2024, 2025):
                        res[(os_, cf)][yr].append(nr)
        df.drop(columns=["dt"], inplace=True, errors="ignore")
    del raw; gc.collect()

    print(f"{'OS':>4}{'confirm':>10}{'n':>7}{'FULL':>9}{'2024':>9}{'2025':>9}{'win%':>7}")
    print("-" * 56)
    for os_ in OSS:
        for cf in CONFIRMS:
            r = res[(os_, cf)]
            def m(k):
                a = np.array(r[k]); return a.mean() if len(a) else float("nan")
            nn = len(r["full"]); wr2 = (np.array(r["full"]) > 0).mean()*100 if nn else float("nan")
            mark = " *" if m(2025) > 0 else ""
            print(f"{os_:>4}{cf:>10}{nn:>7}{m('full'):>+8.1f}%{m(2024):>+8.1f}%{m(2025):>+8.1f}%{wr2:>6.0f}%{mark}")
    print("\n* = موجب في 2025. نبحث عن أعلى 2025 مع عدد صفقات معقول.")
    print("\nDONE_MRDEV.", flush=True)


if __name__ == "__main__":
    main()
