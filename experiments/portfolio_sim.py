#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""REAL portfolio equity curve — the honest answer to 'does it make money?'

Per-trade and per-month AVERAGES both distort (a 1-trade drought month is weighed
like a 476-trade alt-season month). Capital doesn't work that way. So simulate a
real account: fixed start capital, max N concurrent positions, fixed-fraction
sizing, regime-switched engines, compounding. Report the monthly EQUITY curve,
total return, CAGR, max drawdown, % positive months — on actual capital.

  GATE ON  -> BREAKOUT  (TP+100 / SL-12, 120d)
  GATE OFF -> MEAN-REV  (TP+15 / SL-15, 60d, wRSI>30 + daily>EMA10 + ret99>=-30%)

Sizing: each position = ALLOC fraction of CURRENT equity; up to SLOTS concurrent;
signals compete first-come; if no free slot or cash, the signal is skipped (real).
Run:  python experiments/portfolio_sim.py
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
MWIN = 60*24*3600*1000; M_TP = 0.15; M_SL = 0.15; OS = 30; COST = 0.01; STOP_SLIP = 0.01
START = 10_000.0; SLOTS = 8; ALLOC = 0.10        # 10% of equity per trade, 8 concurrent


def ms(s): return int(pd.Timestamp(s, tz="UTC").timestamp()*1000)
def ema(a, n): return pd.Series(a).ewm(span=n, adjust=False).mean().to_numpy()


def rsi(c, n=14):
    d = np.diff(c, prepend=c[0]); up = np.where(d > 0, d, 0.0); dn = np.where(d < 0, -d, 0.0)
    ru = pd.Series(up).ewm(alpha=1/n, adjust=False).mean().to_numpy()
    rd = pd.Series(dn).ewm(alpha=1/n, adjust=False).mean().to_numpy()
    return 100 - 100/(1 + ru/np.where(rd == 0, 1e-9, rd))


def main():
    print("REAL portfolio equity curve (regime-switched, concurrency-capped)\n", flush=True)
    raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), ms(S), ms(E), log=lambda *a: None)
    runs = {}; tots = {}
    sigs = []        # (ent_t, exit_t, ret_frac, engine)
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
            sigs.append((ent_t, xt, rr, "B"))
        wk = pd.DataFrame({"c": g["close"].resample("W").last(), "v": g["volume"].resample("W").sum(),
                           "t": g["time"].resample("W").last()}).dropna()
        wc = wk["c"].to_numpy(); wt = wk["t"].to_numpy(); wv = wk["v"].to_numpy()
        if len(wc) >= 25:
            wr = rsi(wc, 14); wdv = wc*wv; dema = ema(c, 10)
            for w in range(15, len(wc)):
                if not (wr[w-1] <= OS and wr[w] > OS and int(wt[w]) >= ms(SF)):
                    continue
                if np.nanmean(wdv[max(0, w-4):w]) <= LIQ_MIN:
                    continue
                ent_t = int(wt[w]); ent_px = float(wc[w])
                di = np.searchsorted(t, ent_t, side="right") - 1
                if di < 99 or not (c[di] > dema[di]) or (c[di]/c[di-99]-1) < -0.30:
                    continue
                j0 = np.searchsorted(T, ent_t, side="right"); j1 = np.searchsorted(T, ent_t+MWIN, side="right")
                rr = None; xt = None
                for k in range(j0, j1):
                    if L[k] <= ent_px*(1-M_SL): rr = -M_SL - COST - STOP_SLIP; xt = int(T[k]); break
                    if H[k] >= ent_px*(1+M_TP): rr = M_TP - COST; xt = int(T[k]); break
                if rr is None:
                    ke = min(j1, len(C)-1); rr = (C[ke]/ent_px-1) - COST; xt = int(T[ke])
                sigs.append((ent_t, xt, rr, "M"))
        df.drop(columns=["dt"], inplace=True, errors="ignore")
    del raw; gc.collect()

    days = sorted(tots); sup_t = np.array([dd*DAY for dd in days])
    sup_v = np.array([runs.get(dd, 0)/tots[dd]*100 for dd in days])
    def supply(tt):
        i = np.searchsorted(sup_t, tt, side="left") - 1
        return sup_v[i] if i >= 0 else np.nan
    thr = np.nanmedian([supply(s[0]) for s in sigs if s[3] == "B"])
    gate_on = lambda tt: supply(tt) >= thr

    def run(engine_filter, label):
        ss = sorted(sigs, key=lambda x: x[0])
        equity = START; open_pos = []           # list of (exit_t, pnl_at_close)
        eq_t = []; eq_v = []
        for ent_t, exit_t, rr, eng in ss:
            # close matured positions up to this entry time
            open_pos.sort()
            while open_pos and open_pos[0][0] <= ent_t:
                _, pnl = open_pos.pop(0); equity += pnl; eq_t.append(_); eq_v.append(equity)
            if not engine_filter(ent_t, eng):
                continue
            if len(open_pos) >= SLOTS:
                continue
            stake = equity*ALLOC
            if stake <= 0:
                continue
            equity -= stake                       # lock capital
            open_pos.append((exit_t, stake*(1+rr)))   # returns at exit
        open_pos.sort()
        for xt, pnl in open_pos:
            equity += pnl; eq_t.append(xt); eq_v.append(equity)
        eq_t = np.array(eq_t); eq_v = np.array(eq_v)
        order = np.argsort(eq_t); eq_t = eq_t[order]; eq_v = eq_v[order]
        # monthly equity (last value each month)
        s = pd.Series(eq_v, index=pd.to_datetime(eq_t, unit="ms"))
        meq = s.resample("ME").last().dropna()
        rets = meq.pct_change().dropna()*100
        peak = np.maximum.accumulate(eq_v); dd = ((eq_v-peak)/peak).min()*100
        yrs = (eq_t[-1]-eq_t[0])/(365.25*DAY)
        cagr = ((eq_v[-1]/START)**(1/yrs)-1)*100 if yrs > 0 else float("nan")
        print(f"\n### {label} ###")
        print(f"  رأس مال نهائي: ${eq_v[-1]:,.0f}  (بداية ${START:,.0f})  |  إجمالي {(eq_v[-1]/START-1)*100:+.0f}%")
        print(f"  CAGR {cagr:+.0f}%/سنة  |  أقصى تراجع {dd:.0f}%  |  أشهر موجبة {(rets>0).mean()*100:.0f}%  |  وسيط شهري {rets.median():+.1f}%")
        return meq

    print(f"عتبة العرض = {thr:.0f}%   |   إشارات: اختراق {sum(1 for s in sigs if s[3]=='B')}  ارتداد {sum(1 for s in sigs if s[3]=='M')}")
    print(f"إعدادات: ${START:,.0f} بداية، {SLOTS} صفقات متزامنة، {ALLOC*100:.0f}% لكل صفقة، تركيب")
    run(lambda tt, eng: eng == "B", "اختراق فقط (كل الريجيمات)")
    run(lambda tt, eng: eng == "B" and gate_on(tt), "اختراق + بوابة (يجلس في الجفاف)")
    run(lambda tt, eng: (eng == "B" and gate_on(tt)) or (eng == "M" and not gate_on(tt)),
        "المُجمّع: اختراق(بوابة) + ارتداد(جفاف)")
    print("\n⚠️ متفائل بانحياز البقاء (مخفّف بفلتر السيولة، غير مُلغى).")
    print("\nDONE_PORT.", flush=True)


if __name__ == "__main__":
    main()
