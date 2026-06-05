#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""User's 'Double DCA Recovery' formula, faithfully, on squeeze entries, 2025.

Budget $400. NO stop loss. Exit at +1% above original entry (sell all).
DOWN phase (scale-in as it falls from entry P0):
   initial $40 @ entry; +$60 @ -8%; +$100 @ -16%; +$100 @ -24%; +$100 @ -32%.
REBOUND phase (only if it dipped below entry and budget not fully deployed):
   measured from the running bottom: +3% -> invest 25% of remaining;
   +6% -> 25% of remaining; +9% -> 25% of remaining;
   within ~2% of entry -> invest ALL remaining (full $400 before returning to entry).
Entry trigger = squeeze breakout signal. Window 120d; if +1% never hit, mark to market.
Reports 2025 (and full sample): exit-rate, avg return, portfolio, worst losers.
Run:  python experiments/dca_double.py
"""

from __future__ import annotations

import gc, sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from binance_sim import pit_universe as PIT               # noqa: E402

LIQ_MIN = 300_000; WIN = 120*24*3600*1000; BUDGET = 400.0
S, E = "2022-06-01", "2026-06-01"; W0, W1 = "2025-01-01", "2026-01-01"
DOWN = [(0.08, 60.0), (0.16, 100.0), (0.24, 100.0), (0.32, 100.0)]
REB = [0.03, 0.06, 0.09]; TGT = 2.0          # exit = our strategy target +100% (2x entry)
STOP = 0.33                                   # hard stop: close all if price hits -33% from entry
START = 10_000.0; SLOTS = 6; ALLOC = 0.10


def ms(s): return int(pd.Timestamp(s, tz="UTC").timestamp()*1000)
def ema(a, n): return pd.Series(a).ewm(span=n, adjust=False).mean().to_numpy()


def simulate(Hs, Ls, lastc, P0):
    invested = 40.0; coins = 40.0/P0
    dfill = [False]*4; rdone = [False]*3; near = False; dipped = False
    bottom = P0; exitpx = P0*TGT
    for k in range(len(Hs)):
        lo = Ls[k]; hi = Hs[k]
        if lo < bottom:
            bottom = lo; rdone = [False]*3          # re-arm rebound from new bottom
        if lo < P0:
            dipped = True
        # down-fills (price falling through fixed levels)
        for di in range(4):
            lvl, amt = DOWN[di]; price = P0*(1-lvl)
            if not dfill[di] and lo <= price and invested < BUDGET-1e-9:
                a = min(amt, BUDGET-invested); coins += a/price; invested += a; dfill[di] = True
        if lo <= P0*(1-STOP):                         # hard stop at -33% from entry (after fills)
            spx = P0*(1-STOP); return (coins*spx-invested)/BUDGET*100, "stop", invested
        # rebound-fills (only after a real dip, measured from bottom)
        if dipped and invested < BUDGET-1e-9:
            rebpct = hi/bottom - 1
            for ri in range(3):
                if not rdone[ri] and rebpct >= REB[ri] and invested < BUDGET-1e-9:
                    fp = bottom*(1+REB[ri]); a = min(0.25*(BUDGET-invested), BUDGET-invested)
                    coins += a/fp; invested += a; rdone[ri] = True
            if not near and hi >= P0*0.98 and invested < BUDGET-1e-9:
                fp = P0*0.98; a = BUDGET-invested; coins += a/fp; invested += a; near = True
        if hi >= exitpx:                              # exit at our target +100% (2x entry)
            return (coins*exitpx-invested)/BUDGET*100, "exit", invested
    return (coins*lastc-invested)/BUDGET*100, "timeout", invested


def main():
    print("Double-DCA recovery formula on squeeze entries\n", flush=True)
    raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), ms(S), ms(E), log=lambda *a: None)
    rows = []     # (ent_t, yr, ret%, tag, invested, simple_ret%)
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
        o = d["o"].to_numpy(); v = d["v"].to_numpy(); t = d["t"].to_numpy()
        sma = pd.Series(c).rolling(20).mean(); std = pd.Series(c).rolling(20).std()
        bbw = (4*std/sma).to_numpy(); q25 = pd.Series(bbw).rolling(100).quantile(0.25).to_numpy()
        e20 = ema(c, 20); e50 = ema(c, 50); e200 = ema(c, 200)
        macd = ema(c, 12)-ema(c, 26); sg = ema(macd, 9); hist = macd-sg
        dv = c*v
        T = df["time"].to_numpy(); H = df["high"].to_numpy(float); L = df["low"].to_numpy(float); C = df["close"].to_numpy(float)
        for i in range(200, n-1):
            if not (np.isfinite(q25[i-1]) and bbw[i-1] <= q25[i-1] and bbw[i] > bbw[i-1]
                    and hist[i] > hist[i-1] > hist[i-2] and macd[i] > macd[i-1]
                    and abs(e20[i]-e50[i])/c[i] < 0.03 and c[i] > o[i]
                    and c[i] > e200[i] and i >= 99 and c[i] >= c[i-99]):
                continue
            if np.nanmean(dv[max(0, i-30):i]) <= LIQ_MIN:
                continue
            P0 = c[i]; ent_t = int(t[i]); yr = int(pd.Timestamp(ent_t, unit="ms").year)
            j0 = np.searchsorted(T, ent_t, side="right"); j1 = np.searchsorted(T, ent_t+WIN, side="right")
            Hs = H[j0:j1]; Ls = L[j0:j1]; lastc = C[min(j1, len(C)-1)]
            ret, tag, inv = simulate(Hs, Ls, lastc, P0)
            sret = None                              # our simple strategy: $400 @ entry, TP+100/SL-12
            for k in range(len(Hs)):
                if Ls[k] <= P0*0.88: sret = -12.0; break
                if Hs[k] >= P0*2.0: sret = 100.0; break
            if sret is None:
                sret = (lastc/P0-1)*100
            rows.append((ent_t, yr, ret, tag, inv, sret))
        df.drop(columns=["dt"], inplace=True, errors="ignore")
    del raw; gc.collect()

    def report(sel, label):
        if not sel:
            print(f"{label}: لا صفقات"); return
        rets = np.array([r[2] for r in sel]); tags = [r[3] for r in sel]
        ex = sum(t == "exit" for t in tags); to = sum(t == "timeout" for t in tags); st = sum(t == "stop" for t in tags)
        inv = np.array([r[4] for r in sel]); srets = np.array([r[5] for r in sel])
        # portfolio
        ss = sorted([(r[0], r[0]+WIN, r[2]) for r in sel])
        equity = START; op = []; eqv = []; nt = 0
        for et, xt, rr in ss:
            op.sort()
            while op and op[0][0] <= et:
                _, p = op.pop(0); equity += p; eqv.append(equity)
            if len(op) >= SLOTS:
                continue
            stake = equity*ALLOC; equity -= stake; op.append((xt, stake*(1+rr/100))); nt += 1
        for xt, p in sorted(op):
            equity += p; eqv.append(equity)
        eqv = np.array(eqv); dd = ((eqv-np.maximum.accumulate(eqv))/np.maximum.accumulate(eqv)).min()*100 if len(eqv) else float("nan")
        port = (eqv[-1]/START-1)*100 if len(eqv) else float("nan")
        print(f"\n### {label}: {len(sel)} صفقة ###")
        print(f"  هدف +100%: {ex} ({ex/len(sel)*100:.0f}%)  |  وقف −33%: {st} ({st/len(sel)*100:.0f}%)  |  عالقة بالنافذة: {to} ({to/len(sel)*100:.0f}%)")
        print(f"  متوسط العائد/صفقة: {rets.mean():+.1f}%   |   الوسيط: {np.median(rets):+.1f}%")
        print(f"  متوسط المستثمر فعلياً: ${inv.mean():.0f} من 400")
        print(f"  أسوأ 5 صفقات: {', '.join(f'{x:+.0f}%' for x in np.sort(rets)[:5])}")
        print(f"  محفظة (6 خانات، 10%): {port:+.0f}%   تراجع {dd:+.0f}%")
        print(f"  استراتيجيتنا البسيطة (دخول $400 دفعة، TP+100/SL−12) → متوسط {srets.mean():+.1f}%/صفقة")

    rows2025 = [r for r in rows if W0 <= pd.Timestamp(r[0], unit="ms").strftime('%Y-%m-%d') < W1 or r[1] == 2025]
    rows2025 = [r for r in rows if r[1] == 2025]
    report(rows2025, "2025 (جفاف)")
    report([r for r in rows if r[1] == 2024], "2024 (موسم بديل)")
    report(rows, "كامل العيّنة 2022-2026")
    print("\n⚠️ لا يوجد وقف خسارة — الصفقة العالقة تبقى بكامل $400 قرب القاع. انحياز البقاء يضخّم التفاؤل.")
    print("\nDONE_DCA.", flush=True)


if __name__ == "__main__":
    main()
