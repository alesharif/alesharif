#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tame the -53% drawdown of the +10% MR signal (OS25 daily-RSI cross + lower-wick)
by attacking its real cause: CLUSTERING. In a drought everything is oversold at
once, so 8 correlated longs crater together. Test risk controls on 2025:

  SLOTS    : max concurrent positions (2/3/4/6/8)
  COOLDOWN : min days between NEW entries (de-cluster the book: 0/2/5/10)
  ALLOC    : fraction of equity per trade

Goal: keep ~+10% return while pulling drawdown from -53% toward -20%.
Run:  python experiments/mr_risk_control.py
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
START = 10_000.0; COST = 0.01; STOP_SLIP = 0.01; TP, SL = 0.15, 0.15; OS = 25


def ms(s): return int(pd.Timestamp(s, tz="UTC").timestamp()*1000)
def ema(a, n): return pd.Series(a).ewm(span=n, adjust=False).mean().to_numpy()


def rsi(c, n=14):
    d = np.diff(c, prepend=c[0]); up = np.where(d > 0, d, 0.0); dn = np.where(d < 0, -d, 0.0)
    ru = pd.Series(up).ewm(alpha=1/n, adjust=False).mean().to_numpy()
    rd = pd.Series(dn).ewm(alpha=1/n, adjust=False).mean().to_numpy()
    return 100 - 100/(1 + ru/np.where(rd == 0, 1e-9, rd))


def main():
    print("Risk-control the +10% MR signal (OS25 + lower-wick) — 2025\n", flush=True)
    raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), ms(S), ms(E), log=lambda *a: None)
    w0, w1 = ms(W0), ms(W1); trades = []      # (ent_t, exit_t, rr)
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
        o = d["o"].to_numpy(); lo = d["l"].to_numpy(); hi = d["h"].to_numpy(); v = d["v"].to_numpy(); t = d["t"].to_numpy(); dv = c*v
        r = rsi(c, 14)
        T = df["time"].to_numpy(); H = df["high"].to_numpy(float); L = df["low"].to_numpy(float); C = df["close"].to_numpy(float)
        for i in range(100, n):
            if not (w0 <= int(t[i]) < w1) or np.nanmean(dv[max(0, i-30):i]) <= LIQ_MIN:
                continue
            if not (r[i-1] <= OS and r[i] > OS):
                continue
            rng = max(hi[i]-lo[i], 1e-12)
            if (min(o[i], c[i])-lo[i])/rng < 0.4:       # lower-wick rejection
                continue
            ent_t = int(t[i]); ent_px = float(c[i])
            j0 = np.searchsorted(T, ent_t, side="right"); j1 = np.searchsorted(T, ent_t+MWIN, side="right")
            rr = None; xt = None
            for k in range(j0, j1):
                if L[k] <= ent_px*(1-SL): rr = -SL-COST-STOP_SLIP; xt = int(T[k]); break
                if H[k] >= ent_px*(1+TP): rr = TP-COST; xt = int(T[k]); break
            if rr is None:
                ke = min(j1, len(C)-1); rr = (C[ke]/ent_px-1)-COST; xt = int(T[ke])
            trades.append((ent_t, xt, rr))
        df.drop(columns=["dt"], inplace=True, errors="ignore")
    del raw; gc.collect()
    trades.sort(key=lambda x: x[0])
    print(f"إشارات الفتيل في 2025: {len(trades)}\n")

    def sim(slots, alloc, cooldown):
        equity = START; op = []; eqt = []; eqv = []; nt = 0; last_entry = -10**18
        cd = cooldown*DAY
        for et, xt, rr in trades:
            op.sort()
            while op and op[0][0] <= et:
                a, pnl = op.pop(0); equity += pnl; eqt.append(a); eqv.append(equity)
            if len(op) >= slots or (et - last_entry) < cd:
                continue
            stake = equity*alloc; equity -= stake; op.append((xt, stake*(1+rr))); nt += 1; last_entry = et
        for xt, pnl in sorted(op):
            equity += pnl; eqt.append(xt); eqv.append(equity)
        if not eqv:
            return None
        eqt = np.array(eqt); eqv = np.array(eqv); ordr = np.argsort(eqt); eqt = eqt[ordr]; eqv = eqv[ordr]
        s = pd.Series(eqv, index=pd.to_datetime(eqt, unit="ms")).resample("ME").last().dropna()
        rets = s.pct_change().dropna()*100
        peak = np.maximum.accumulate(eqv); dd = ((eqv-peak)/peak).min()*100
        ret = (eqv[-1]/START-1)*100
        return ret, dd, nt, ret/abs(dd) if dd != 0 else float("nan")

    print(f"{'slots':>6}{'alloc':>7}{'cooldown':>9}{'صفقات':>7}{'عائد%':>8}{'تراجع':>8}{'عائد/تراجع':>11}")
    print("-"*56)
    best = None
    for slots in (8, 6, 4, 3, 2):
        for alloc in (0.10, 0.06):
            for cd in (0, 2, 5, 10):
                res = sim(slots, alloc, cd)
                if res is None:
                    continue
                ret, dd, nt, ratio = res
                if best is None or ratio > best[0]:
                    best = (ratio, slots, alloc, cd, ret, dd, nt)
                print(f"{slots:>6}{alloc:>7.2f}{cd:>9}{nt:>7}{ret:>+7.0f}%{dd:>+7.0f}%{ratio:>+11.2f}")
    print("-"*56)
    b = best
    print(f"الأفضل مخاطرةً: slots={b[1]} alloc={b[2]:.2f} cooldown={b[3]}d → عائد {b[4]:+.0f}% تراجع {b[5]:+.0f}% ({b[6]} صفقة)")
    print("الباسلاين (8,0.10,0): +10% تراجع −53%. الهدف: نسبة عائد/تراجع أعلى.")
    print("⚠️ متفائل بانحياز البقاء (مخفّف بالسيولة).")
    print("\nDONE_RC.", flush=True)


if __name__ == "__main__":
    main()
