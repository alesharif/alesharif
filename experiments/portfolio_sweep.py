#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Improve the winner (breakout + gate, NO mean-reversion): can a STRICTER gate
and/or supply-scaled exposure lift return and cut the -52% drawdown?

Generate breakout signals once, then sweep:
  GATE PCTL : supply threshold = p-th percentile of supply-at-signal (50/65/80)
  EXPOSURE  : flat (ALLOC, SLOTS) vs supply-scaled (more slots/size when supply high)
Report total return, CAGR, max drawdown, % positive months, Calmar (CAGR/|DD|).
Run:  python experiments/portfolio_sweep.py
"""

from __future__ import annotations

import gc, sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from binance_sim import pit_universe as PIT               # noqa: E402

LIQ_MIN = 300_000; DAY = 86400000
S, E, SF = "2022-06-01", "2026-06-01", "2024-01-01"
BWIN = 120*24*3600*1000; B_SL = 0.12; B_TP = 2.0
START = 10_000.0


def ms(s): return int(pd.Timestamp(s, tz="UTC").timestamp()*1000)
def ema(a, n): return pd.Series(a).ewm(span=n, adjust=False).mean().to_numpy()


def main():
    print("Sweep gate strictness + exposure for breakout+gate (no MR)\n", flush=True)
    raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), ms(S), ms(E), log=lambda *a: None)
    runs = {}; tots = {}; sigs = []        # (ent_t, exit_t, ret_frac)
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
        for i in range(60, n):
            if np.nanmean(dv[max(0, i-30):i]) <= LIQ_MIN:
                continue
            day = int(t[i]) // DAY; tots[day] = tots.get(day, 0) + 1
            if c[i] / c[i-60] - 1 >= 0.50:
                runs[day] = runs.get(day, 0) + 1
        T = df["time"].to_numpy(); H = df["high"].to_numpy(float); L = df["low"].to_numpy(float); C = df["close"].to_numpy(float)
        sma = pd.Series(c).rolling(20).mean(); std = pd.Series(c).rolling(20).std()
        bbw = (4*std/sma).to_numpy(); q25 = pd.Series(bbw).rolling(100).quantile(0.25).to_numpy()
        e20 = ema(c, 20); e50 = ema(c, 50); e200 = ema(c, 200)
        macd = ema(c, 12)-ema(c, 26); sg = ema(macd, 9); hist = macd-sg
        for i in range(200, n-1):
            if int(t[i]) < ms(SF):
                continue
            if not (np.isfinite(q25[i-1]) and bbw[i-1] <= q25[i-1] and bbw[i] > bbw[i-1]
                    and hist[i] > hist[i-1] > hist[i-2] and macd[i] > macd[i-1]
                    and abs(e20[i]-e50[i])/c[i] < 0.03 and c[i] > o[i]
                    and c[i] > e200[i] and i >= 99 and c[i] >= c[i-99]):
                continue
            if np.nanmean(dv[max(0, i-30):i]) <= LIQ_MIN:
                continue
            P0 = c[i]; ent_t = int(t[i]); sl = P0*(1-B_SL); tp = P0*B_TP
            j0 = np.searchsorted(T, ent_t, side="right"); j1 = np.searchsorted(T, ent_t+BWIN, side="right")
            rr = None; xt = None
            for k in range(j0, j1):
                if L[k] <= sl: rr = -B_SL - 0.01; xt = int(T[k]); break
                if H[k] >= tp: rr = (B_TP-1) - 0.01; xt = int(T[k]); break
            if rr is None:
                ke = min(j1, len(C)-1); rr = (C[ke]/P0-1) - 0.01; xt = int(T[ke])
            sigs.append((ent_t, xt, rr))
        df.drop(columns=["dt"], inplace=True, errors="ignore")
    del raw; gc.collect()

    days = sorted(tots); sup_t = np.array([dd*DAY for dd in days])
    sup_v = np.array([runs.get(dd, 0)/tots[dd]*100 for dd in days])
    def supply(tt):
        i = np.searchsorted(sup_t, tt, side="left") - 1
        return sup_v[i] if i >= 0 else np.nan
    sup_at = np.array([supply(s[0]) for s in sigs])
    ss = sorted(zip(sigs, sup_at), key=lambda x: x[0][0])

    def simulate(thr, slots, alloc, scaled):
        equity = START; open_pos = []; eq_t = []; eq_v = []
        for (ent_t, exit_t, rr), sup in ss:
            open_pos.sort()
            while open_pos and open_pos[0][0] <= ent_t:
                xt, pnl = open_pos.pop(0); equity += pnl; eq_t.append(xt); eq_v.append(equity)
            if not (sup >= thr) or len(open_pos) >= slots:
                continue
            a = alloc
            if scaled:                                   # scale size with how far above thr
                a = alloc*min(2.0, 0.5 + (sup-thr)/max(thr, 1e-9))
                a = min(a, 0.20)
            stake = equity*a
            if stake <= 0:
                continue
            equity -= stake; open_pos.append((exit_t, stake*(1+rr)))
        open_pos.sort()
        for xt, pnl in open_pos:
            equity += pnl; eq_t.append(xt); eq_v.append(equity)
        eq_t = np.array(eq_t); eq_v = np.array(eq_v); order = np.argsort(eq_t)
        eq_t = eq_t[order]; eq_v = eq_v[order]
        s = pd.Series(eq_v, index=pd.to_datetime(eq_t, unit="ms")).resample("ME").last().dropna()
        rets = s.pct_change().dropna()*100
        peak = np.maximum.accumulate(eq_v); dd = ((eq_v-peak)/peak).min()*100
        yrs = (eq_t[-1]-eq_t[0])/(365.25*DAY); cagr = ((eq_v[-1]/START)**(1/yrs)-1)*100 if yrs > 0 else float("nan")
        calmar = cagr/abs(dd) if dd != 0 else float("nan")
        return eq_v[-1], cagr, dd, (rets > 0).mean()*100, calmar

    pct = {50: np.nanpercentile(sup_at, 50), 65: np.nanpercentile(sup_at, 65), 80: np.nanpercentile(sup_at, 80)}
    print(f"عتبات العرض: p50={pct[50]:.0f}%  p65={pct[65]:.0f}%  p80={pct[80]:.0f}%   |   إشارات {len(sigs)}\n")
    print(f"{'config':<30}{'نهائي$':>9}{'CAGR':>7}{'تراجع':>7}{'موجب%':>7}{'Calmar':>8}")
    print("-"*68)
    for p in (50, 65, 80):
        for slots, alloc, scaled, lab in [(8, 0.10, False, "flat 8x10%"),
                                          (6, 0.10, False, "flat 6x10%"),
                                          (8, 0.10, True, "supply-scaled")]:
            fin, cagr, dd, pos, cal = simulate(pct[p], slots, alloc, scaled)
            print(f"p{p} {lab:<25}{fin:>8.0f}{cagr:>+6.0f}%{dd:>+6.0f}%{pos:>6.0f}%{cal:>+8.2f}")
    print("\nCalmar = CAGR/|تراجع| (أعلى=أفضل مخاطرةً). الهدف: رفعه فوق نسخة p50-flat.")
    print("⚠️ متفائل بانحياز البقاء (مخفّف بالسيولة).")
    print("\nDONE_SWEEP.", flush=True)


if __name__ == "__main__":
    main()
