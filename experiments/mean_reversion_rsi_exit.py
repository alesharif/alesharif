#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Proper RANGE round-trip mean-reversion: buy weekly-RSI oversold, SELL at weekly-
RSI OVERBOUGHT (not a fixed TP). Stop -15%, 180-day window. This is the classic
'buy the bottom, sell the top inside the range' the assistant described for
ranging markets. Per regime: FULL | 2024 (trend) | 2025 (DROUGHT).
Run:  python experiments/mean_reversion_rsi_exit.py
"""

from __future__ import annotations

import gc, sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from binance_sim import pit_universe as PIT               # noqa: E402

WINDOW_MS = 180*24*3600*1000; LIQ_MIN = 300_000; SL = 0.15
S, E, SF = "2022-06-01", "2026-06-01", "2022-09-01"
OBS = [60, 65, 70]; COST = 1.0; STOP_SLIP = 1.0


def ms(s): return int(pd.Timestamp(s, tz="UTC").timestamp()*1000)


def rsi(c, n=14):
    d = np.diff(c, prepend=c[0]); up = np.where(d > 0, d, 0.0); dn = np.where(d < 0, -d, 0.0)
    ru = pd.Series(up).ewm(alpha=1/n, adjust=False).mean().to_numpy()
    rd = pd.Series(dn).ewm(alpha=1/n, adjust=False).mean().to_numpy()
    return 100 - 100/(1 + ru/np.where(rd == 0, 1e-9, rd))


def main():
    print("Range round-trip: buy weekly-RSI oversold, SELL at weekly-RSI overbought\n", flush=True)
    raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), ms(S), ms(E), log=lambda *a: None)
    res = {ob: {"full": [], 2024: [], 2025: []} for ob in OBS}
    nsig = 0
    for sym, df in raw.items():
        df = df.sort_values("time").reset_index(drop=True)
        df["dt"] = pd.to_datetime(df["time"], unit="ms")
        g = df.set_index("dt")
        wk = pd.DataFrame({"c": g["close"].resample("W").last(),
                           "v": g["volume"].resample("W").sum(),
                           "t": g["time"].resample("W").last()}).dropna()
        wc = wk["c"].to_numpy(); wt = wk["t"].to_numpy(); wv = wk["v"].to_numpy()
        if len(wc) < 25:
            df.drop(columns=["dt"], inplace=True, errors="ignore"); continue
        r = rsi(wc, 14); wdv = wc*wv
        T = df["time"].to_numpy(); L = df["low"].to_numpy(float); C = df["close"].to_numpy(float)
        for w in range(15, len(wc)):
            if not (r[w-1] <= 35 and r[w] > 35 and int(wt[w]) >= ms(SF)):
                continue
            if np.nanmean(wdv[max(0, w-4):w]) <= LIQ_MIN:
                continue
            nsig += 1
            ent_t, ent_px = int(wt[w]), float(wc[w]); j0 = np.searchsorted(T, ent_t)
            if j0 >= len(T):
                continue
            yr = int(pd.Timestamp(ent_t, unit="ms").strftime("%Y"))
            sl_px = ent_px*(1-SL)
            for ob in OBS:
                # first later week reaching overbought, within window
                we = None
                for k in range(w+1, len(wc)):
                    if int(wt[k]) - ent_t > WINDOW_MS:
                        break
                    if r[k] >= ob:
                        we = k; break
                horizon_t = int(wt[we]) if we is not None else ent_t + WINDOW_MS
                jh = np.searchsorted(T, horizon_t)
                seg = L[j0:min(jh, len(L))]
                if seg.size and seg.min() <= sl_px:        # stop hit before exit
                    nr = -SL*100 - COST - STOP_SLIP
                elif we is not None:
                    nr = (wc[we]/ent_px - 1)*100 - COST    # sold at overbought
                else:
                    nr = (C[min(jh, len(C)-1)]/ent_px - 1)*100 - COST   # timeout
                res[ob]["full"].append(nr)
                if yr in (2024, 2025):
                    res[ob][yr].append(nr)
        df.drop(columns=["dt"], inplace=True, errors="ignore")
    del raw; gc.collect()

    print(f"  {nsig} signals (SL {int(SL*100)}%, 180d window)\n")
    print(f"{'sell@RSI':>9}{'FULL':>9}{'2024':>9}{'2025(drought)':>15}{'win%':>7}")
    print("-" * 50)
    for ob in OBS:
        r = res[ob]
        def m(k):
            a = np.array(r[k]); return a.mean() if len(a) else float("nan")
        wr = (np.array(r["full"]) > 0).mean()*100 if r["full"] else float("nan")
        mark = "  <== +2025" if m(2025) > 0 else ""
        print(f"{ob:>8} {m('full'):>+8.1f}%{m(2024):>+8.1f}%{m(2025):>+13.1f}%{wr:>6.0f}%{mark}")
    print("\nبيع عند RSI مرتفع = الجولة الكاملة للنطاق. موجب 2025 => يصلح للأسواق العرضية.")
    print("\nDONE_MRX.", flush=True)


if __name__ == "__main__":
    main()
