#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HYBRID v2: regime-routed by runner-supply gauge. Alt-season (supply>=THR) runs TWO
momentum engines together: squeeze breakout + big-green-day continuation. Drought
(supply<THR) runs mean-reversion. vs BTC buy&hold. Daily, $2,000 spot, cost 0.3% RT.
Run: python experiments/final_system_v2.py
"""
from __future__ import annotations
import gc, sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from binance_sim import pit_universe as PIT

S, E, SF = "2022-06-01", "2026-06-01", "2022-09-01"
DAY = 86400000; LIQ_MIN = 300_000; THR = 12.0; COST = 0.30; START = 2000.0
BWIN = 120*DAY; B_SL = 0.12; B_TP = 1.0
BG_THR = 0.07; BG_TRAIL = 0.20; BG_WIN = 120*DAY      # big-green-day
MWIN = 60*DAY; M_TP = 0.15; M_SL = 0.15; OS = 25
MOM_STAKE = 0.15; MOM_SLOTS = 6; M_STAKE = 0.10; M_SLOTS = 2


def ms(s): return int(pd.Timestamp(s, tz="UTC").timestamp()*1000)
def ema(a, n): return pd.Series(a).ewm(span=n, adjust=False).mean().to_numpy()


def rsi(c, n=14):
    d = np.diff(c, prepend=c[0]); up = np.where(d > 0, d, 0.0); dn = np.where(d < 0, -d, 0.0)
    ru = pd.Series(up).ewm(alpha=1/n, adjust=False).mean().to_numpy()
    rd = pd.Series(dn).ewm(alpha=1/n, adjust=False).mean().to_numpy()
    return 100 - 100/(1 + ru/np.where(rd == 0, 1e-9, rd))


def main():
    print("HYBRID v2: alt-season=(squeeze+big-green), drought=MR, vs BTC\n", flush=True)
    raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), ms(S), ms(E), log=lambda *a: None)
    runs = {}; tots = {}; SQ = []; BG = []; MR = []; btc = None
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
        o = d["o"].to_numpy(); hi = d["h"].to_numpy(); lo = d["l"].to_numpy(); v = d["v"].to_numpy(); t = d["t"].to_numpy(); dv = c*v
        for i in range(60, n):
            if np.nanmean(dv[max(0, i-30):i]) <= LIQ_MIN:
                continue
            day = int(t[i])//DAY; tots[day] = tots.get(day, 0)+1
            if c[i]/c[i-60]-1 >= 0.50:
                runs[day] = runs.get(day, 0)+1
        T = df["time"].to_numpy(); H = df["high"].to_numpy(float); L = df["low"].to_numpy(float); C = df["close"].to_numpy(float)
        r = rsi(c, 14); e50d = ema(c, 50)
        # squeeze breakout
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
                P0 = c[i]; ent = int(t[i]); slp = P0*(1-B_SL); tpp = P0*(1+B_TP)
                j0 = np.searchsorted(T, ent, side="right"); j1 = np.searchsorted(T, ent+BWIN, side="right")
                rr = None; xt = ent
                for k in range(j0, j1):
                    if L[k] <= slp: rr = -B_SL*100-COST; xt = int(T[k]); break
                    if H[k] >= tpp: rr = B_TP*100-COST; xt = int(T[k]); break
                if rr is None:
                    ke = min(j1, len(C)-1); rr = (C[ke]/P0-1)*100-COST; xt = int(T[ke])
                SQ.append((ent, xt, rr))
        # big-green-day continuation (daily-close trailing exit)
        for i in range(60, n-1):
            if int(t[i]) < ms(SF):
                continue
            if not (c[i]/c[i-1]-1 >= BG_THR and c[i] > e50d[i] and np.nanmean(dv[max(0, i-30):i]) > LIQ_MIN):
                continue
            ent = int(t[i]); P0 = c[i]; peak = c[i]; rr = None; xt = ent
            kend = min(n, i+1+120)                          # max hold 120 days
            for k in range(i+1, kend):
                peak = max(peak, c[k])
                if c[k] <= peak*(1-BG_TRAIL):
                    rr = (c[k]/P0-1)*100-COST; xt = int(t[k]); break
            if rr is None:
                ke = min(kend, n)-1; rr = (c[ke]/P0-1)*100-COST; xt = int(t[ke])
            BG.append((ent, xt, rr))
        # mean-reversion (drought)
        for i in range(100, n):
            if int(t[i]) < ms(SF) or np.nanmean(dv[max(0, i-30):i]) <= LIQ_MIN:
                continue
            if not (r[i-1] <= OS and r[i] > OS):
                continue
            rng = max(hi[i]-lo[i], 1e-12)
            if (min(o[i], c[i])-lo[i])/rng < 0.4:
                continue
            P0 = c[i]; ent = int(t[i]); j0 = np.searchsorted(T, ent, side="right"); j1 = np.searchsorted(T, ent+MWIN, side="right")
            rr = None; xt = ent
            for k in range(j0, j1):
                if L[k] <= P0*(1-M_SL): rr = -M_SL*100-COST; xt = int(T[k]); break
                if H[k] >= P0*(1+M_TP): rr = M_TP*100-COST; xt = int(T[k]); break
            if rr is None:
                ke = min(j1, len(C)-1); rr = (C[ke]/P0-1)*100-COST; xt = int(T[ke])
            MR.append((ent, xt, rr))
        if sym == "BTCUSDT":
            btc = (t//DAY, c.copy())
        df.drop(columns=["dt"], inplace=True, errors="ignore")
    del raw; gc.collect()

    days = sorted(tots); sup_t = np.array([dd*DAY for dd in days]); sup_v = np.array([runs.get(dd, 0)/tots[dd]*100 for dd in days])
    def supply(tt):
        i = np.searchsorted(sup_t, tt, side="left")-1
        return sup_v[i] if i >= 0 else np.nan
    bt, bc = btc
    for nm, lst in [("انقباض", SQ), ("يوم-أخضر", BG), ("ارتداد", MR)]:
        a = np.array([x[2] for x in lst])
        print(f"تشخيص {nm}: n={len(a)} متوسط {a.mean():+.1f}% وسيط {np.median(a):+.1f}% رابحة {(a>0).mean()*100:.0f}%", flush=True)

    def stats(eqt, eqv, lab):
        eqt = np.array(eqt); eqv = np.array(eqv); o = np.argsort(eqt); eqt = eqt[o]; eqv = eqv[o]
        s = pd.Series(eqv, index=pd.to_datetime(eqt, unit="ms")).resample("YE").last()
        yr = {}; prev = START
        for ts, val in s.items():
            yr[ts.year] = (val/prev-1)*100; prev = val
        peak = np.maximum.accumulate(eqv); dd = ((eqv-peak)/peak).min()*100
        yrs = (eqt[-1]-eqt[0])/(365.25*DAY); cagr = ((eqv[-1]/START)**(1/yrs)-1)*100
        ys = " ".join(f"{y}:{yr.get(y, float('nan')):+.0f}%" for y in [2023, 2024, 2025, 2026])
        print(f"{lab:<28}${eqv[-1]:>8,.0f} CAGR{cagr:>+5.0f}% سحب{dd:>+5.0f}%  {ys}")

    # portfolio: alt-season -> SQ+BG (shared MOM slots); drought -> MR
    sigs = sorted([(e, x, rr, 'M') for e, x, rr in (SQ+BG)] + [(e, x, rr, 'R') for e, x, rr in MR])
    cash = START; op = []; eqt = []; eqv = []
    for et, xt, rr, eng in sigs:
        op.sort(key=lambda z: z[0])
        while op and op[0][0] <= et:
            _0, p, _s, _e = op.pop(0); cash += p; eqt.append(_0); eqv.append(cash+sum(z[2] for z in op))
        equity = cash+sum(z[2] for z in op); on = supply(et) >= THR
        if eng == 'M':
            if not on or sum(1 for z in op if z[3] == 'M') >= MOM_SLOTS:
                continue
            stake = equity*MOM_STAKE
        else:
            if on or sum(1 for z in op if z[3] == 'R') >= M_SLOTS:
                continue
            stake = equity*M_STAKE
        stake = min(stake, cash)
        if stake < 1:
            continue
        cash -= stake; op.append((xt, stake*(1+rr/100), stake, eng))
    for z in sorted(op):
        cash += z[1]; eqt.append(z[0]); eqv.append(cash)
    beq = START*bc/bc[0]
    print(f"عتبة={THR:.0f}% | انقباض {len(SQ)} يوم-أخضر {len(BG)} ارتداد {len(MR)}\n")
    stats(eqt, eqv, "الهجين v2 (انقباض+أخضر+ارتداد)")
    stats(bt*DAY, beq, "احتفاظ بيتكوين")
    print("\n⚠️ متفائل بانحياز البقاء. BTC واقعي. عمولة 0.3% محسوبة.")
    print("\nDONE_V2.", flush=True)


if __name__ == "__main__":
    main()
