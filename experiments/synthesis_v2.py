#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SYNTHESIS V2 — develop the user's threads into ONE system, not reject them.

Established facts (all proven earlier):
  * breakout edge is real but regime-dependent (2024 +, 2025 -).
  * root cause of 2025: runner supply collapsed (fewer coins reach +50/100%).
  * scaled ENTRY ties simple (re-buy at entry erases the discount) -> drop it.
  * 71% of trades hit the stop, only 22% reach 2x  -> all-or-nothing TP100 wastes
    the many moves that reach +30/+60% but not +100% (esp. in 2025).

So we develop the EXIT and the GATE instead of the entry:
  SCALE-OUT exit: sell 1/3 at +30%, 1/3 at +60%, TRAIL final 1/3 (25% from peak)
                  -> banks 2025's smaller moves AND lets 2024's runners ride.
  REGIME GATE   : trailing-60d market runner-rate >= median -> sit out droughts.
  CONVICTION    : size by ATR%/ADX at entry (big-move predictors) -> bet on quality.

Compared on the SAME breakout entries, per regime, return on a fixed $ budget.
Run:  python experiments/synthesis_v2.py
"""

from __future__ import annotations

import gc, sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from binance_sim import pit_universe as PIT               # noqa: E402

LIQ_MIN = 300_000; WIN = 120*24*3600*1000; DAY = 86400000; BUDGET = 400.0
S, E, SF = "2022-06-01", "2026-06-01", "2024-01-01"
SL = 0.12                                   # hard stop on the whole position
T1, F1 = 0.30, 1/3                          # take 1/3 at +30%
T2, F2 = 0.60, 1/3                          # take 1/3 at +60%
TRAIL = 0.25                                # trail final third 25% from peak after +60%


def ms(s): return int(pd.Timestamp(s, tz="UTC").timestamp()*1000)
def ema(a, n): return pd.Series(a).ewm(span=n, adjust=False).mean().to_numpy()


def adx(h, l, c, n=14):
    up = np.diff(h, prepend=h[0]); dn = -np.diff(l, prepend=l[0])
    pdm = np.where((up > dn) & (up > 0), up, 0.0); ndm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = np.maximum(h - l, np.maximum(abs(h - np.roll(c, 1)), abs(l - np.roll(c, 1)))); tr[0] = h[0]-l[0]
    atr = pd.Series(tr).ewm(alpha=1/n, adjust=False).mean().to_numpy()
    pdi = 100*pd.Series(pdm).ewm(alpha=1/n, adjust=False).mean().to_numpy()/np.where(atr == 0, 1e-9, atr)
    ndi = 100*pd.Series(ndm).ewm(alpha=1/n, adjust=False).mean().to_numpy()/np.where(atr == 0, 1e-9, atr)
    dx = 100*np.abs(pdi-ndi)/np.where(pdi+ndi == 0, 1e-9, pdi+ndi)
    return pd.Series(dx).ewm(alpha=1/n, adjust=False).mean().to_numpy()


def simple_exit(Hs, Ls, lastc, P0):
    tp = P0*2; sl = P0*(1-SL)
    for k in range(len(Hs)):
        if Ls[k] <= sl: return (sl/P0-1)*100
        if Hs[k] >= tp: return (tp/P0-1)*100
    return (lastc/P0-1)*100


def scaleout_exit(Hs, Ls, lastc, P0):
    """1/3 @ +30%, 1/3 @ +60%, trail final 1/3 (25% from peak after +60%).
    Hard stop -12% on whatever remains. Returns % gain on full budget."""
    sl = P0*(1-SL); l1 = P0*(1+T1); l2 = P0*(1+T2)
    coins = BUDGET/P0; cash = 0.0
    s1 = s2 = False; trailing = False; peak = P0
    for k in range(len(Hs)):
        hi = Hs[k]; lo = Ls[k]
        # stop on remaining position (check low first = conservative)
        if lo <= sl:
            cash += coins*sl; coins = 0.0; break
        if not s1 and hi >= l1:                       # bank first third
            sell = (BUDGET/P0)*F1; cash += sell*l1; coins -= sell; s1 = True
        if not s2 and hi >= l2:                        # bank second third, start trailing
            sell = (BUDGET/P0)*F2; cash += sell*l2; coins -= sell; s2 = True
            trailing = True; peak = hi
        if trailing:
            peak = max(peak, hi)
            if lo <= peak*(1-TRAIL):                    # trail the final third
                cash += coins*peak*(1-TRAIL); coins = 0.0; break
    if coins > 0:
        cash += coins*lastc
    return (cash/BUDGET-1)*100


def main():
    print("SYNTHESIS V2: scale-out exit + regime gate + conviction (breakout)\n", flush=True)
    raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), ms(S), ms(E), log=lambda *a: None)
    runs = {}; tots = {}                               # trailing runner-rate series
    rows = []                                          # (ent_t, yr, simple%, scaleout%, atrp, adxv)
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
        o = d["o"].to_numpy(); hd = d["h"].to_numpy(); ld = d["l"].to_numpy()
        v = d["v"].to_numpy(); t = d["t"].to_numpy(); dv = c*v
        for i in range(60, n):                          # trailing 60d runner contribution
            if np.nanmean(dv[max(0, i-30):i]) <= LIQ_MIN:
                continue
            day = int(t[i]) // DAY; tots[day] = tots.get(day, 0) + 1
            if c[i] / c[i-60] - 1 >= 0.50:
                runs[day] = runs.get(day, 0) + 1
        sma = pd.Series(c).rolling(20).mean(); std = pd.Series(c).rolling(20).std()
        bbw = (4*std/sma).to_numpy(); q25 = pd.Series(bbw).rolling(100).quantile(0.25).to_numpy()
        e20 = ema(c, 20); e50 = ema(c, 50); e200 = ema(c, 200)
        macd = ema(c, 12)-ema(c, 26); sg = ema(macd, 9); hist = macd-sg
        atrp = pd.Series(np.maximum(hd-ld, np.maximum(abs(hd-np.roll(c, 1)), abs(ld-np.roll(c, 1))))
                         ).ewm(alpha=1/14, adjust=False).mean().to_numpy()/np.where(c == 0, 1e-9, c)*100
        adxv = adx(hd, ld, c, 14)
        T = df["time"].to_numpy(); H = df["high"].to_numpy(float); L = df["low"].to_numpy(float); C = df["close"].to_numpy(float)
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
            P0 = c[i]; ent_t = int(t[i]); yr = int(pd.Timestamp(ent_t, unit="ms").year)
            j0 = np.searchsorted(T, ent_t, side="right"); j1 = np.searchsorted(T, ent_t+WIN, side="right")
            Hs = H[j0:j1]; Ls = L[j0:j1]; lastc = C[min(j1, len(C)-1)]
            rows.append((ent_t, yr, simple_exit(Hs, Ls, lastc, P0),
                         scaleout_exit(Hs, Ls, lastc, P0), atrp[i], adxv[i]))
        df.drop(columns=["dt"], inplace=True, errors="ignore")
    del raw; gc.collect()

    days = sorted(tots); sup_t = np.array([dd*DAY for dd in days])
    sup_v = np.array([runs.get(dd, 0)/tots[dd]*100 for dd in days])
    def supply(tt):
        i = np.searchsorted(sup_t, tt, side="left") - 1
        return sup_v[i] if i >= 0 else np.nan
    sup_of = np.array([supply(r[0]) for r in rows]); thr = np.nanmedian(sup_of)
    gate = sup_of >= thr
    adxa = np.array([r[5] for r in rows]); atra = np.array([r[4] for r in rows])
    yrs = np.array([r[1] for r in rows]); simp = np.array([r[2] for r in rows]); sco = np.array([r[3] for r in rows])

    def line(name, ret, sel):
        s = sel
        out = f"{name:<22}"
        for y in [2024, 2025, None]:
            m = s & (yrs == y) if y else s
            a = ret[m]
            out += f"{a.mean() if len(a) else float('nan'):>+8.1f}%({len(a):>4})"
        print(out)

    allm = np.ones(len(rows), bool)
    print(f"عتبة العرض (الوسيط) = {thr:.0f}%   |   إجمالي الإشارات = {len(rows)}\n")
    print(f"{'system':<22}{'2024':>13}{'2025':>13}{'الكل':>13}")
    print("-"*61)
    line("A SIMPLE TP100/SL12", simp, allm)
    line("B SCALE-OUT", sco, allm)
    line("C SCALE-OUT + GATE", sco, gate)
    line("A SIMPLE + GATE", simp, gate)
    print("\n— تحجيم القناعة (على B داخل البوابة): الوزن ∝ ADX، ثابت متوسط الوزن —")
    g = gate
    base = sco[g]; w = adxa[g]; w = w/np.nanmean(w)           # normalized weights, mean 1
    for y, lab in [(2024, "2024"), (2025, "2025"), (None, "الكل")]:
        m = (yrs[g] == y) if y else np.ones(g.sum(), bool)
        eq = base[m].mean() if m.any() else float("nan")
        wt = np.nansum(base[m]*w[m])/m.sum() if m.any() else float("nan")
        print(f"  {lab:<6} متساوٍ {eq:>+7.1f}%   |   موزون-ADX {wt:>+7.1f}%")
    print("\nالأرقام = متوسط العائد لكل صفقة على ميزانية ثابتة (وليس مجموع السنة).")
    print("⚠️ متفائل بانحياز البقاء (مخفّف بفلتر السيولة، غير مُلغى).")
    print("\nDONE_V2.", flush=True)


if __name__ == "__main__":
    main()
