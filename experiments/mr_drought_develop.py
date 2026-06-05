#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DEVELOP mean-reversion for the drought (2025) — it's the only drought winner.

Weekly-RSI MR is too thin (31 trades/yr). Test DAILY-RSI oversold-bounce for more
signals, several oversold thresholds, and tighter stops to cut the -45% DD. Each
config is run as a real 2025 PORTFOLIO (equity, return, drawdown, positive months)
so we optimize what capital actually does, not per-trade averages.

Entry: daily RSI(14) crosses up from <=OS, confirm close>EMA10, ret99>=-30%, liquid.
Exits swept: TP/SL pairs. Run:  python experiments/mr_drought_develop.py
"""

from __future__ import annotations

import gc, sys, itertools
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from binance_sim import pit_universe as PIT               # noqa: E402

LIQ_MIN = 300_000; DAY = 86400000; MWIN = 60*24*3600*1000
S, E = "2022-06-01", "2026-06-01"
W0, W1 = "2025-01-01", "2026-01-01"
START = 10_000.0; SLOTS = 8; ALLOC = 0.10; COST = 0.01; STOP_SLIP = 0.01
OS_SET = [25, 30, 35]
EXITS = [(0.15, 0.15), (0.15, 0.10), (0.20, 0.10), (0.20, 0.08)]   # (TP, SL)


def ms(s): return int(pd.Timestamp(s, tz="UTC").timestamp()*1000)
def ema(a, n): return pd.Series(a).ewm(span=n, adjust=False).mean().to_numpy()


def rsi(c, n=14):
    d = np.diff(c, prepend=c[0]); up = np.where(d > 0, d, 0.0); dn = np.where(d < 0, -d, 0.0)
    ru = pd.Series(up).ewm(alpha=1/n, adjust=False).mean().to_numpy()
    rd = pd.Series(dn).ewm(alpha=1/n, adjust=False).mean().to_numpy()
    return 100 - 100/(1 + ru/np.where(rd == 0, 1e-9, rd))


def main():
    print("Develop drought MR (daily-RSI) — 2025 portfolio, sweep OS x TP/SL\n", flush=True)
    raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), ms(S), ms(E), log=lambda *a: None)
    w0, w1 = ms(W0), ms(W1)
    # collect raw entries per OS (ent_t, ent_px path indices) once; exits computed per TP/SL
    ent = {os_: [] for os_ in OS_SET}      # (ent_t, ent_px, j0, j1ref, symkey) -> store arrays per sym
    store = {}                              # symkey -> (T,H,L,C)
    sk = 0
    for sym, df in raw.items():
        df = df.sort_values("time").reset_index(drop=True)
        df["dt"] = pd.to_datetime(df["time"], unit="ms")
        g = df.set_index("dt")
        d = pd.DataFrame({"c": g["close"].resample("D").last(), "v": g["volume"].resample("D").sum(),
                          "t": g["time"].resample("D").last()}).dropna()
        c = d["c"].to_numpy(); n = len(c)
        if n < 130:
            df.drop(columns=["dt"], inplace=True, errors="ignore"); continue
        v = d["v"].to_numpy(); t = d["t"].to_numpy(); dv = c*v
        r = rsi(c, 14); e10 = ema(c, 10)
        T = df["time"].to_numpy(); H = df["high"].to_numpy(float); L = df["low"].to_numpy(float); C = df["close"].to_numpy(float)
        store[sk] = (T, H, L, C)
        for i in range(100, n):
            if not (w0 <= int(t[i]) < w1):
                continue
            if np.nanmean(dv[max(0, i-30):i]) <= LIQ_MIN:
                continue
            if not (c[i] > e10[i] and (c[i]/c[i-99]-1) >= -0.30):
                continue
            for os_ in OS_SET:
                if r[i-1] <= os_ and r[i] > os_:
                    ent[os_].append((int(t[i]), float(c[i]), sk))
        df.drop(columns=["dt"], inplace=True, errors="ignore"); sk += 1
    del raw; gc.collect()

    def exit_ret(ent_t, ent_px, skk, TP, SL):
        T, H, L, C = store[skk]
        j0 = np.searchsorted(T, ent_t, side="right"); j1 = np.searchsorted(T, ent_t+MWIN, side="right")
        for k in range(j0, j1):
            if L[k] <= ent_px*(1-SL): return (-SL-COST-STOP_SLIP, int(T[k]))
            if H[k] >= ent_px*(1+TP): return (TP-COST, int(T[k]))
        ke = min(j1, len(C)-1); return ((C[ke]/ent_px-1)-COST, int(T[ke]))

    def portfolio(trades):       # trades: list (ent_t, exit_t, rr)
        ss = sorted(trades, key=lambda x: x[0]); equity = START; op = []; eqt = []; eqv = []; nt = 0
        for et, xt, rr in ss:
            op.sort()
            while op and op[0][0] <= et:
                a, pnl = op.pop(0); equity += pnl; eqt.append(a); eqv.append(equity)
            if len(op) >= SLOTS:
                continue
            stake = equity*ALLOC
            equity -= stake; op.append((xt, stake*(1+rr))); nt += 1
        for xt, pnl in sorted(op):
            equity += pnl; eqt.append(xt); eqv.append(equity)
        if not eqv:
            return None
        eqt = np.array(eqt); eqv = np.array(eqv); o = np.argsort(eqt); eqt = eqt[o]; eqv = eqv[o]
        s = pd.Series(eqv, index=pd.to_datetime(eqt, unit="ms")).resample("ME").last().dropna()
        rets = s.pct_change().dropna()*100
        peak = np.maximum.accumulate(eqv); dd = ((eqv-peak)/peak).min()*100
        wins = sum(1 for _, _, rr in ss if rr > 0)
        return eqv[-1], dd, (rets > 0).mean()*100 if len(rets) else float("nan"), nt, wins/len(ss)*100 if ss else float("nan")

    print(f"{'OS':>4}{'TP/SL':>9}{'إشارات':>8}{'نهائي$':>9}{'عائد%':>7}{'تراجع':>7}{'win%':>6}{'موجب%':>7}")
    print("-"*60)
    best = None
    for os_ in OS_SET:
        for TP, SL in EXITS:
            tr = []
            for et, px, skk in ent[os_]:
                rr, xt = exit_ret(et, px, skk, TP, SL)
                tr.append((et, xt, rr))
            res = portfolio(tr)
            if res is None:
                continue
            fin, dd, pos, nt, wr = res
            tag = ""
            if best is None or fin > best[0]:
                best = (fin, os_, TP, SL); tag = ""
            print(f"{os_:>4}{f'{int(TP*100)}/{int(SL*100)}':>9}{len(ent[os_]):>8}{fin:>8.0f}{(fin/START-1)*100:>+6.0f}%{dd:>+6.0f}%{wr:>5.0f}%{pos:>6.0f}%")
    print("-"*60)
    print(f"الأفضل: OS={best[1]}  TP/SL={int(best[2]*100)}/{int(best[3]*100)}  →  ${best[0]:,.0f}")
    print("الباسلاين (أسبوعي): +10% تراجع −45%. الهدف: عائد أعلى وتراجع أقل في 2025.")
    print("⚠️ متفائل بانحياز البقاء (مخفّف بالسيولة).")
    print("\nDONE_MRD.", flush=True)


if __name__ == "__main__":
    main()
