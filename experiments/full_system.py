#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""THE FULL SYSTEM — regime-routed, full cycle 2022-2026, honest equity curve.

Both halves are now OOS-validated in their own regime. Route by an ABSOLUTE
runner-supply gauge so each engine runs ONLY in its regime:
  supply >= THR  (alt-season) -> BREAKOUT  (TP100/SL12, up to 6 concurrent)
  supply <  THR  (drought)    -> MEAN-REV  (RSI25 cross + lower-wick, TP15/SL15,
                                            max 2 concurrent — clustering control)
Shared equity, 10% per trade, compounding. Benchmark = BTC buy & hold.
Run:  python experiments/full_system.py
"""

from __future__ import annotations

import gc, sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from binance_sim import pit_universe as PIT               # noqa: E402

LIQ_MIN = 300_000; DAY = 86400000
S, E, SF = "2022-06-01", "2026-06-01", "2022-09-01"
BWIN = 120*24*3600*1000; B_SL = 0.12; B_TP = 2.0; B_SLOTS = 6
MWIN = 60*24*3600*1000; M_TP = 0.15; M_SL = 0.15; OS = 25; M_SLOTS = 2
START = 10_000.0; ALLOC = 0.10; THR = 12.0


def ms(s): return int(pd.Timestamp(s, tz="UTC").timestamp()*1000)
def ema(a, n): return pd.Series(a).ewm(span=n, adjust=False).mean().to_numpy()


def rsi(c, n=14):
    d = np.diff(c, prepend=c[0]); up = np.where(d > 0, d, 0.0); dn = np.where(d < 0, -d, 0.0)
    ru = pd.Series(up).ewm(alpha=1/n, adjust=False).mean().to_numpy()
    rd = pd.Series(dn).ewm(alpha=1/n, adjust=False).mean().to_numpy()
    return 100 - 100/(1 + ru/np.where(rd == 0, 1e-9, rd))


def main():
    print("FULL regime-routed system, full cycle, vs BTC buy&hold\n", flush=True)
    raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), ms(S), ms(E), log=lambda *a: None)
    runs = {}; tots = {}; sigs = []          # (ent_t, exit_t, rr, engine)
    btc_eq = None
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
        T = df["time"].to_numpy(); H = df["high"].to_numpy(float); L = df["low"].to_numpy(float); C = df["close"].to_numpy(float)
        r = rsi(c, 14)
        # breakout
        if n >= 220:
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
                    if L[k] <= sl: rr = -B_SL-0.01; xt = int(T[k]); break
                    if H[k] >= tp: rr = (B_TP-1)-0.01; xt = int(T[k]); break
                if rr is None:
                    ke = min(j1, len(C)-1); rr = (C[ke]/P0-1)-0.01; xt = int(T[ke])
                sigs.append((ent_t, xt, rr, "B"))
        # mean-reversion (frozen drought rules)
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
            sigs.append((ent_t, xt, rr, "M"))
        if sym == "BTCUSDT":
            btc_eq = (d["t"].to_numpy(), c.copy())
        df.drop(columns=["dt"], inplace=True, errors="ignore")
    del raw; gc.collect()

    days = sorted(tots); sup_t = np.array([dd*DAY for dd in days])
    sup_v = np.array([runs.get(dd, 0)/tots[dd]*100 for dd in days])
    def supply(tt):
        i = np.searchsorted(sup_t, tt, side="left") - 1
        return sup_v[i] if i >= 0 else np.nan
    sigs.sort(key=lambda x: x[0])

    equity = START; op = []        # (exit_t, pnl, engine)
    eqt = []; eqv = []; nb = nm = 0
    for et, xt, rr, eng in sigs:
        op.sort(key=lambda z: z[0])
        while op and op[0][0] <= et:
            _, pnl, _e = op.pop(0); equity += pnl; eqt.append(_); eqv.append(equity)
        sup = supply(et); on = sup >= THR
        if eng == "B":
            if not on or sum(1 for z in op if z[2] == "B") >= B_SLOTS:
                continue
        else:
            if on or sum(1 for z in op if z[2] == "M") >= M_SLOTS:
                continue
        stake = equity*ALLOC; equity -= stake; op.append((xt, stake*(1+rr), eng))
        nb += eng == "B"; nm += eng == "M"
    for xt, pnl, _e in sorted(op, key=lambda z: z[0]):
        equity += pnl; eqt.append(xt); eqv.append(equity)
    eqt = np.array(eqt); eqv = np.array(eqv); ordr = np.argsort(eqt); eqt = eqt[ordr]; eqv = eqv[ordr]

    def stats(et_arr, ev_arr, label):
        s = pd.Series(ev_arr, index=pd.to_datetime(et_arr, unit="ms")).resample("ME").last().dropna()
        rets = s.pct_change().dropna()*100
        peak = np.maximum.accumulate(ev_arr); dd = ((ev_arr-peak)/peak).min()*100
        yrs = (et_arr[-1]-et_arr[0])/(365.25*DAY); cagr = ((ev_arr[-1]/ev_arr[0])**(1/yrs)-1)*100
        print(f"{label:<22} نهائي ${ev_arr[-1]:>9,.0f}  ({(ev_arr[-1]/ev_arr[0]-1)*100:>+5.0f}%)  CAGR {cagr:>+5.0f}%  تراجع {dd:>+4.0f}%  موجب {(rets>0).mean()*100:>3.0f}%")

    print(f"عتبة مطلقة={THR:.0f}%  |  دخول: اختراق {nb}  ارتداد {nm}\n")
    stats(eqt, eqv, "النظام الكامل")
    # BTC buy & hold over same span
    bt, bc = btc_eq
    m = (bt >= eqt[0]) & (bt <= eqt[-1])
    bt2 = bt[m]; bc2 = bc[m]; beq = START*bc2/bc2[0]
    stats(bt2, beq, "احتفاظ بيتكوين")
    print("\nالحكم: هل النظام المُوجَّه يتفوّق على الاحتفاظ بالبيتكوين معدّلاً بالتراجع؟")
    print("⚠️ متفائل بانحياز البقاء في صفقات النظام (مخفّف بالسيولة، غير مُلغى).")
    print("\nDONE_FULL.", flush=True)


if __name__ == "__main__":
    main()
