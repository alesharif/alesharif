#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HYBRID: hold BTC in alt-season (supply high), switch to the MR drought system
when supply is low. Breakout underperforms holding BTC even in alt-season, so the
'risk-on' engine is simply BTC. The MR sleeve preserves/grows capital in droughts
where BTC bleeds. Daily switch on the lagged absolute runner-supply gauge.

Compared to: pure BTC buy&hold, and (reference) the MR-only system.
Run:  python experiments/hybrid_btc_mr.py
"""

from __future__ import annotations

import gc, sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from binance_sim import pit_universe as PIT               # noqa: E402

LIQ_MIN = 300_000; DAY = 86400000; MWIN = 60*24*3600*1000
S, E, SF = "2022-06-01", "2026-06-01", "2022-09-01"
M_TP = 0.15; M_SL = 0.15; OS = 25; M_SLOTS = 2; ALLOC = 0.10
START = 10_000.0; THR = 12.0


def ms(s): return int(pd.Timestamp(s, tz="UTC").timestamp()*1000)
def ema(a, n): return pd.Series(a).ewm(span=n, adjust=False).mean().to_numpy()


def rsi(c, n=14):
    d = np.diff(c, prepend=c[0]); up = np.where(d > 0, d, 0.0); dn = np.where(d < 0, -d, 0.0)
    ru = pd.Series(up).ewm(alpha=1/n, adjust=False).mean().to_numpy()
    rd = pd.Series(dn).ewm(alpha=1/n, adjust=False).mean().to_numpy()
    return 100 - 100/(1 + ru/np.where(rd == 0, 1e-9, rd))


def main():
    print("HYBRID: BTC in alt-season + MR in drought, full cycle\n", flush=True)
    raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), ms(S), ms(E), log=lambda *a: None)
    runs = {}; tots = {}; mrsig = []; btc = None
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
        for i in range(60, n):
            if np.nanmean(dv[max(0, i-30):i]) <= LIQ_MIN:
                continue
            day = int(t[i]) // DAY; tots[day] = tots.get(day, 0) + 1
            if c[i] / c[i-60] - 1 >= 0.50:
                runs[day] = runs.get(day, 0) + 1
        r = rsi(c, 14)
        T = df["time"].to_numpy(); H = df["high"].to_numpy(float); L = df["low"].to_numpy(float); C = df["close"].to_numpy(float)
        for i in range(100, n):
            if int(t[i]) < ms(SF) or np.nanmean(dv[max(0, i-30):i]) <= LIQ_MIN:
                continue
            if not (r[i-1] <= OS and r[i] > OS):
                continue
            rng = max(hi[i]-lo[i], 1e-12)
            if (min(o[i], c[i])-lo[i])/rng < 0.4:
                continue
            ent_t = int(t[i]); ent_px = float(c[i])
            j0 = np.searchsorted(T, ent_t, side="right"); j1 = np.searchsorted(T, ent_t+MWIN, side="right")
            rr = None; xt = None
            for k in range(j0, j1):
                if L[k] <= ent_px*(1-M_SL): rr = -M_SL-0.02; xt = int(T[k]); break
                if H[k] >= ent_px*(1+M_TP): rr = M_TP-0.01; xt = int(T[k]); break
            if rr is None:
                ke = min(j1, len(C)-1); rr = (C[ke]/ent_px-1)-0.01; xt = int(T[ke])
            mrsig.append((int(t[i]) // DAY, int(xt) // DAY, rr))
        if sym == "BTCUSDT":
            btc = (t // DAY, c.copy())
        df.drop(columns=["dt"], inplace=True, errors="ignore")
    del raw; gc.collect()

    days = sorted(tots); sup_t = np.array(days); sup_v = np.array([runs.get(dd, 0)/tots[dd]*100 for dd in days])
    def supply(day):
        i = np.searchsorted(sup_t, day, side="right") - 1
        return sup_v[i] if i >= 0 else np.nan
    bdays, bpx = btc
    mr_by_day = {}
    for ed, xd, rr in mrsig:
        mr_by_day.setdefault(ed, []).append((xd, rr))

    def simulate(mode):       # mode: 'hybrid' | 'btc' | 'mr'
        cash = START; units = 0.0; openmr = []      # (exit_day, payout)
        eqv = []; eqd = []
        for k, day in enumerate(bdays):
            px = bpx[k]
            for j in range(len(openmr)-1, -1, -1):
                if openmr[j][0] <= day:
                    cash += openmr[j][1]; openmr.pop(j)
            on = supply(day) >= THR
            if mode == "btc":
                on = True
            if mode == "mr":
                on = False
            if on:                                    # risk-on: hold BTC
                if cash > 0:
                    units += cash/px; cash = 0.0
            else:                                     # drought: cash, sell BTC, run MR
                if units > 0:
                    cash += units*px; units = 0.0
                if mode != "btc":
                    for xd, rr in mr_by_day.get(day, []):
                        if len(openmr) < M_SLOTS and cash > 0:
                            stake = (cash + sum(p for _, p in openmr))*ALLOC
                            stake = min(stake, cash)
                            cash -= stake; openmr.append((xd, stake*(1+rr)))
            eq = cash + units*px + sum(p for _, p in openmr)
            eqv.append(eq); eqd.append(day*DAY)
        eqv = np.array(eqv); eqd = np.array(eqd)
        s = pd.Series(eqv, index=pd.to_datetime(eqd, unit="ms")).resample("ME").last().dropna()
        rets = s.pct_change().dropna()*100
        peak = np.maximum.accumulate(eqv); dd = ((eqv-peak)/peak).min()*100
        yrs = (eqd[-1]-eqd[0])/(365.25*DAY); cagr = ((eqv[-1]/START)**(1/yrs)-1)*100
        return eqv[-1], cagr, dd, (rets > 0).mean()*100

    print(f"عتبة={THR:.0f}%  |  إشارات MR={len(mrsig)}\n")
    print(f"{'النظام':<26}{'نهائي$':>10}{'عائد%':>8}{'CAGR':>7}{'تراجع':>8}{'موجب%':>7}")
    print("-"*66)
    for mode, lab in [("hybrid", "هجين (بيتكوين+ارتداد)"), ("btc", "احتفاظ بيتكوين"), ("mr", "ارتداد فقط (نقد بالصعود)")]:
        fin, cagr, dd, pos = simulate(mode)
        print(f"{lab:<26}{fin:>9,.0f}{(fin/START-1)*100:>+7.0f}%{cagr:>+6.0f}%{dd:>+7.0f}%{pos:>6.0f}%")
    print("-"*66)
    print("الفكرة: تلتقط صعود البيتكوين، وتنسحب لنظام الارتداد في الجفاف بدل النزيف.")
    print("⚠️ MR متفائل بانحياز البقاء؛ BTC واقعي. تقدير محافظ للهجين سيكون أدنى.")
    print("\nDONE_HYB.", flush=True)


if __name__ == "__main__":
    main()
