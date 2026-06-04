#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DEVELOP the winner: squeeze breakout system — optimize exits + liquidity filter,
measured across all three regimes (in-sample / 2024 bull / 2022-23 bear).

Baseline entry = Bollinger squeeze->expansion + MACD curling up + EMA20/50
converged + bullish candle (daily). Initial hard stop -10%. We DEVELOP it:
  EXITS:   TP+50 | TP+100 | TP+200 | TRAIL(activate +20%, trail 25% from peak)
  FILTER:  ALL coins  vs  LIQUID (avg $vol/day > $300k at entry; cuts survivorship
           bias and keeps only tradable names)
Cost applied: 1% round-trip + 1% extra when the hard stop is hit (realistic).
Run:  python experiments/develop_squeeze.py
"""

from __future__ import annotations

import gc, sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from binance_sim import pit_universe as PIT               # noqa: E402

SL = 0.10; WINDOW_MS = 90*24*3600*1000; LIQ_MIN = 300_000
REGIMES = {
    "in-sample": ("2022-06-01", "2026-06-01", "2022-09-01"),
    "2024-bull": ("2023-06-01", "2025-01-01", "2024-01-01"),
    "bear22-23": ("2022-06-01", "2023-12-01", "2022-10-01"),
}


def ms(s): return int(pd.Timestamp(s, tz="UTC").timestamp()*1000)
def ema(a, n): return pd.Series(a).ewm(span=n, adjust=False).mean().to_numpy()


def resample_d(df):
    g = df.set_index("dt")
    return pd.DataFrame({"o": g["open"].resample("D").first(),
                         "h": g["high"].resample("D").max(),
                         "l": g["low"].resample("D").min(),
                         "c": g["close"].resample("D").last(),
                         "v": g["volume"].resample("D").sum(),
                         "t": g["time"].resample("D").last()}).dropna()


def entries(d, sig_from):
    o = d["o"].to_numpy(); c = d["c"].to_numpy(); v = d["v"].to_numpy(); t = d["t"].to_numpy()
    n = len(c)
    if n < 120:
        return []
    sma = pd.Series(c).rolling(20).mean(); std = pd.Series(c).rolling(20).std()
    bbw = (4*std/sma).to_numpy(); q25 = pd.Series(bbw).rolling(100).quantile(0.25).to_numpy()
    e20 = ema(c, 20); e50 = ema(c, 50)
    macd = ema(c, 12)-ema(c, 26); sig = ema(macd, 9); hist = macd-sig
    dv = c * v                                           # daily $ volume
    out = []
    for i in range(100, n):
        if (np.isfinite(q25[i-1]) and bbw[i-1] <= q25[i-1] and bbw[i] > bbw[i-1]
                and hist[i] > hist[i-1] > hist[i-2] and macd[i] > macd[i-1]
                and abs(e20[i]-e50[i])/c[i] < 0.03 and c[i] > o[i] and int(t[i]) >= sig_from):
            liq = np.nanmean(dv[max(0, i-30):i]) if i > 0 else 0.0
            out.append((int(t[i]), float(c[i]), float(liq)))
    return out


def exits(H, L, C, j0, t, ent_px, end_t):
    """Return dict variant -> (gross_ret%, hit_hardstop)."""
    sl = ent_px*(1-SL)
    res = {}
    for name, tpmult in [("TP50", 1.5), ("TP100", 2.0), ("TP200", 3.0)]:
        tp = ent_px*tpmult; r = None; j = j0
        while j < len(t) and t[j] <= end_t:
            if L[j] <= sl: r = (-SL*100, True); break
            if H[j] >= tp: r = ((tpmult-1)*100, False); break
            j += 1
        if r is None:
            r = ((C[min(j, len(t)-1)]/ent_px-1)*100, False)
        res[name] = r
    # trailing: hard stop -10% until +20% reached, then trail 25% below peak
    stop = sl; peak = ent_px; armed = False; r = None; j = j0
    while j < len(t) and t[j] <= end_t:
        if H[j] > peak: peak = H[j]
        if not armed and peak >= ent_px*1.20:
            armed = True
        if armed:
            stop = max(stop, peak*0.75)
        if L[j] <= stop:
            hit_hard = (stop <= sl + 1e-12)
            r = ((stop/ent_px-1)*100, hit_hard); break
        j += 1
    if r is None:
        r = ((C[min(j, len(t)-1)]/ent_px-1)*100, False)
    res["TRAIL"] = r
    return res


def net(gross, hit_hard):
    return gross - 1.0 - (1.0 if hit_hard else 0.0)        # 1% RT + 1% stop slip


def main():
    print("DEVELOP squeeze winner: exits x liquidity, across 3 regimes (net @1%+1%)\n", flush=True)
    VARIANTS = ["TP50", "TP100", "TP200", "TRAIL"]
    # results[regime][variant][liq] -> list of net returns
    res = {rg: {v: {"all": [], "liq": []} for v in VARIANTS} for rg in REGIMES}
    counts = {}

    for rg, (s, e, sf) in REGIMES.items():
        raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), ms(s), ms(e), log=lambda *a: None)
        nsig = 0; nliq = 0
        for sym, df in raw.items():
            df = df.sort_values("time").reset_index(drop=True)
            df["dt"] = pd.to_datetime(df["time"], unit="ms")
            d = resample_d(df)
            sigs = entries(d, ms(sf))
            if not sigs:
                df.drop(columns=["dt"], inplace=True, errors="ignore"); continue
            t = df["time"].to_numpy(); H = df["high"].to_numpy(float)
            L = df["low"].to_numpy(float); C = df["close"].to_numpy(float)
            for ent_t, ent_px, liq in sigs:
                j0 = np.searchsorted(t, ent_t)
                if j0 >= len(t):
                    continue
                ex = exits(H, L, C, j0, t, ent_px, ent_t+WINDOW_MS)
                nsig += 1; isliq = liq > LIQ_MIN
                if isliq: nliq += 1
                for v in VARIANTS:
                    g, hh = ex[v]; nr = net(g, hh)
                    res[rg][v]["all"].append(nr)
                    if isliq: res[rg][v]["liq"].append(nr)
            df.drop(columns=["dt"], inplace=True, errors="ignore")
        counts[rg] = (nsig, nliq)
        print(f"  {rg}: {nsig} signals ({nliq} liquid >$300k/day)", flush=True)
        del raw; gc.collect()

    print(f"\n{'variant':<8}" + "".join(f"{rg+'/all':>13}{rg+'/liq':>13}" for rg in REGIMES))
    print("-" * 86)
    for v in VARIANTS:
        row = f"{v:<8}"
        for rg in REGIMES:
            for f in ("all", "liq"):
                a = np.array(res[rg][v][f])
                row += f"{(a.mean() if len(a) else float('nan')):>+12.1f}%"
        print(row, flush=True)
    print(f"\nالقيمة = التوقّع/صفقة (net @1%+1%). نبحث عن: أعلى توقّع يصمد في الأنظمة الثلاثة،")
    print("وهل فلتر السيولة (liq) يبقي الحافة (=> ليست مجرّد انحياز بقاء) ويحسّنها.")
    print("\nDONE_DEV.", flush=True)


if __name__ == "__main__":
    main()
