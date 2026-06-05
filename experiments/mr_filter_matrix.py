#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DEVELOP deep MR by REMOVING trend filters (user's idea: trend filters fight
mean-reversion, which by nature buys price that is DOWN). Test a matrix of which
filters to keep/drop, at OS=25 and OS=30, as a real 2025 portfolio.

Entry base: daily RSI(14) crosses up from <=OS, liquid. Toggle filters:
  EMA10   : close > EMA10        (short-term trend-up confirmation)
  ret99   : 99d return >= -30%   (avoid catastrophic decline / dead coins)
Quality alternatives (tested in place of trend filters):
  volX    : volume climax  v > 2.5*vma30   (capitulation)
  wick    : lower-wick rejection >= 0.4     (sellers rejected)
  belowE50: close < EMA50        (genuinely extended below the mean)
Report per config: trades, return, drawdown, win%, positive-months (2025).
Run:  python experiments/mr_filter_matrix.py
"""

from __future__ import annotations

import gc, sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from binance_sim import pit_universe as PIT               # noqa: E402

LIQ_MIN = 300_000; DAY = 86400000; MWIN = 60*24*3600*1000
S, E = "2022-06-01", "2026-06-01"
W0, W1 = "2025-01-01", "2026-01-01"
START = 10_000.0; SLOTS = 8; ALLOC = 0.10; COST = 0.01; STOP_SLIP = 0.01
TP, SL = 0.15, 0.15


def ms(s): return int(pd.Timestamp(s, tz="UTC").timestamp()*1000)
def ema(a, n): return pd.Series(a).ewm(span=n, adjust=False).mean().to_numpy()


def rsi(c, n=14):
    d = np.diff(c, prepend=c[0]); up = np.where(d > 0, d, 0.0); dn = np.where(d < 0, -d, 0.0)
    ru = pd.Series(up).ewm(alpha=1/n, adjust=False).mean().to_numpy()
    rd = pd.Series(dn).ewm(alpha=1/n, adjust=False).mean().to_numpy()
    return 100 - 100/(1 + ru/np.where(rd == 0, 1e-9, rd))


# config = (label, OS, requires-dict) ; requires keys among ema10,ret99,volX,wick,belowE50
CONFIGS = [
    ("OS25 base (EMA10+ret99)", 25, {"ema10": 1, "ret99": 1}),
    ("OS25 -EMA10",             25, {"ret99": 1}),
    ("OS25 -ret99",             25, {"ema10": 1}),
    ("OS25 NO trend filters",   25, {}),
    ("OS25 +volX",              25, {"volX": 1}),
    ("OS25 +wick",              25, {"wick": 1}),
    ("OS25 +belowE50",          25, {"belowE50": 1}),
    ("OS25 +volX+wick",         25, {"volX": 1, "wick": 1}),
    ("OS30 base (EMA10+ret99)", 30, {"ema10": 1, "ret99": 1}),
    ("OS30 NO trend filters",   30, {}),
    ("OS30 +volX+wick",         30, {"volX": 1, "wick": 1}),
    ("OS35 +volX+wick",         35, {"volX": 1, "wick": 1}),
]


def main():
    print("Remove trend filters from deep MR — 2025 portfolio matrix\n", flush=True)
    raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), ms(S), ms(E), log=lambda *a: None)
    w0, w1 = ms(W0), ms(W1)
    # entries: list of (ent_t, exit_t, rr, OS, feats-dict)
    rows = []
    for sym, df in raw.items():
        df = df.sort_values("time").reset_index(drop=True)
        df["dt"] = pd.to_datetime(df["time"], unit="ms")
        g = df.set_index("dt")
        d = pd.DataFrame({"o": g["open"].resample("D").first(), "h": g["high"].resample("D").max(),
                          "l": g["low"].resample("D").min(), "c": g["close"].resample("D").last(),
                          "v": g["volume"].resample("D").sum(), "t": g["time"].resample("D").last()}).dropna()
        c = d["c"].to_numpy(); n = len(c)
        if n < 130:
            df.drop(columns=["dt"], inplace=True, errors="ignore"); continue
        o = d["o"].to_numpy(); hi = d["h"].to_numpy(); lo = d["l"].to_numpy()
        v = d["v"].to_numpy(); t = d["t"].to_numpy(); dv = c*v
        r = rsi(c, 14); e10 = ema(c, 10); e50 = ema(c, 50)
        vma = pd.Series(v).rolling(30).mean().to_numpy()
        T = df["time"].to_numpy(); H = df["high"].to_numpy(float); L = df["low"].to_numpy(float); C = df["close"].to_numpy(float)
        for i in range(100, n):
            if not (w0 <= int(t[i]) < w1):
                continue
            if np.nanmean(dv[max(0, i-30):i]) <= LIQ_MIN:
                continue
            # which OS thresholds does this bar cross?
            crossed = [os_ for os_ in (25, 30, 35) if r[i-1] <= os_ and r[i] > os_]
            if not crossed:
                continue
            rng = max(hi[i]-lo[i], 1e-12)
            feats = {"ema10": c[i] > e10[i],
                     "ret99": (c[i]/c[i-99]-1) >= -0.30,
                     "volX": np.isfinite(vma[i]) and vma[i] > 0 and v[i] > 2.5*vma[i],
                     "wick": (min(o[i], c[i])-lo[i])/rng >= 0.4,
                     "belowE50": c[i] < e50[i]}
            ent_t = int(t[i]); ent_px = float(c[i])
            j0 = np.searchsorted(T, ent_t, side="right"); j1 = np.searchsorted(T, ent_t+MWIN, side="right")
            rr = None; xt = None
            for k in range(j0, j1):
                if L[k] <= ent_px*(1-SL): rr = -SL-COST-STOP_SLIP; xt = int(T[k]); break
                if H[k] >= ent_px*(1+TP): rr = TP-COST; xt = int(T[k]); break
            if rr is None:
                ke = min(j1, len(C)-1); rr = (C[ke]/ent_px-1)-COST; xt = int(T[ke])
            for os_ in crossed:
                rows.append((ent_t, xt, rr, os_, feats))
        df.drop(columns=["dt"], inplace=True, errors="ignore")
    del raw; gc.collect()

    def portfolio(trades):
        ss = sorted(trades, key=lambda x: x[0]); equity = START; op = []; eqt = []; eqv = []; nt = 0; wins = 0
        for et, xt, rr in ss:
            op.sort()
            while op and op[0][0] <= et:
                a, pnl = op.pop(0); equity += pnl; eqt.append(a); eqv.append(equity)
            if len(op) >= SLOTS:
                continue
            stake = equity*ALLOC; equity -= stake; op.append((xt, stake*(1+rr))); nt += 1
            wins += 1 if rr > 0 else 0
        for xt, pnl in sorted(op):
            equity += pnl; eqt.append(xt); eqv.append(equity)
        if not eqv:
            return None
        eqt = np.array(eqt); eqv = np.array(eqv); ordr = np.argsort(eqt); eqt = eqt[ordr]; eqv = eqv[ordr]
        s = pd.Series(eqv, index=pd.to_datetime(eqt, unit="ms")).resample("ME").last().dropna()
        rets = s.pct_change().dropna()*100
        peak = np.maximum.accumulate(eqv); dd = ((eqv-peak)/peak).min()*100
        return eqv[-1], dd, (rets > 0).mean()*100 if len(rets) else float("nan"), nt, wins/nt*100 if nt else float("nan")

    print(f"{'config':<26}{'صفقات':>7}{'عائد%':>8}{'تراجع':>8}{'win%':>7}{'موجب%':>7}")
    print("-"*63)
    for lab, os_, req in CONFIGS:
        sel = []
        for et, xt, rr, o_, feats in rows:
            if o_ != os_:
                continue
            if all(feats.get(k, False) for k in req):
                sel.append((et, xt, rr))
        res = portfolio(sel)
        if res is None:
            print(f"{lab:<26}{len(sel):>7}{'لا صفقات':>30}"); continue
        fin, dd, pos, nt, wr = res
        print(f"{lab:<26}{nt:>7}{(fin/START-1)*100:>+7.0f}%{dd:>+7.0f}%{wr:>6.0f}%{pos:>6.0f}%")
    print("-"*63)
    print("الباسلاين: OS25 base = +4% تراجع −12%. هل إزالة فلتر الاتجاه أو فلتر الجودة أفضل؟")
    print("⚠️ متفائل بانحياز البقاء (مخفّف بالسيولة).")
    print("\nDONE_MAT.", flush=True)


if __name__ == "__main__":
    main()
