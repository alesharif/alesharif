#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TWO-REGIME COMBINED system — the user's actual goal: keep working across
regimes instead of sitting out for months.

Established: breakout has an edge only when runner supply is high (gate ON);
in the drought (gate OFF) it loses. Mean-reversion is the complementary tool
that works in the drought. So SWITCH engines by the SAME slow gate:

  GATE ON  (trailing-60d runner-rate >= median)  -> BREAKOUT  (TP100/SL12, daily)
  GATE OFF (drought)                              -> MEAN-REV  (wRSI cross>30 +
                                                    daily>EMA10 + ret99>=-30%, TP15/SL15)

Output: monthly table (which engine, avg trade return, n) so we SEE whether the
drought months are now covered instead of negative. Compared to breakout-only.
Run:  python experiments/combined_regime.py
"""

from __future__ import annotations

import gc, sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from binance_sim import pit_universe as PIT               # noqa: E402

LIQ_MIN = 300_000; DAY = 86400000
S, E, SF = "2022-06-01", "2026-06-01", "2024-01-01"
# breakout exit
BWIN = 120*24*3600*1000; B_SL = 0.12; B_TP = 2.0
# mean-reversion exit
MWIN = 60*24*3600*1000; M_TP = 0.15; M_SL = 0.15; OS = 30; COST = 1.0; STOP_SLIP = 1.0


def ms(s): return int(pd.Timestamp(s, tz="UTC").timestamp()*1000)
def ema(a, n): return pd.Series(a).ewm(span=n, adjust=False).mean().to_numpy()


def rsi(c, n=14):
    d = np.diff(c, prepend=c[0]); up = np.where(d > 0, d, 0.0); dn = np.where(d < 0, -d, 0.0)
    ru = pd.Series(up).ewm(alpha=1/n, adjust=False).mean().to_numpy()
    rd = pd.Series(dn).ewm(alpha=1/n, adjust=False).mean().to_numpy()
    return 100 - 100/(1 + ru/np.where(rd == 0, 1e-9, rd))


def main():
    print("TWO-REGIME COMBINED: breakout (gate ON) + mean-reversion (gate OFF)\n", flush=True)
    raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), ms(S), ms(E), log=lambda *a: None)
    runs = {}; tots = {}
    bre = []        # (ent_t, ret%)   breakout
    mrt = []        # (ent_t, ret%)   mean-reversion
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
        # ---- breakout signals ----
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
            r = None
            for k in range(j0, j1):
                if L[k] <= sl: r = -B_SL*100 - 1.0; break
                if H[k] >= tp: r = (B_TP-1)*100 - 1.0; break
            if r is None:
                r = (C[min(j1, len(C)-1)]/P0-1)*100 - 1.0
            bre.append((ent_t, r))
        # ---- mean-reversion signals ----
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
                Hs = H[j0:j1]; Ls = L[j0:j1]; lastc = C[min(j1, len(C)-1)]
                hb = Hs >= ent_px*(1+M_TP); lb = Ls <= ent_px*(1-M_SL)
                tph = int(np.argmax(hb)) if hb.any() else 10**9
                slh = int(np.argmax(lb)) if lb.any() else 10**9
                if slh <= tph and slh < 10**9:
                    nr = -M_SL*100 - COST - STOP_SLIP
                elif tph < 10**9:
                    nr = M_TP*100 - COST
                else:
                    nr = (lastc/ent_px-1)*100 - COST
                mrt.append((ent_t, nr))
        df.drop(columns=["dt"], inplace=True, errors="ignore")
    del raw; gc.collect()

    days = sorted(tots); sup_t = np.array([dd*DAY for dd in days])
    sup_v = np.array([runs.get(dd, 0)/tots[dd]*100 for dd in days])
    def supply(tt):
        i = np.searchsorted(sup_t, tt, side="left") - 1
        return sup_v[i] if i >= 0 else np.nan
    allb = np.array([supply(x[0]) for x in bre]); thr = np.nanmedian(allb)
    gate_on = lambda tt: supply(tt) >= thr

    def mon(tt): return pd.Timestamp(tt, unit="ms").strftime("%Y-%m")
    # combined stream: breakout when gate ON, MR when gate OFF
    comb = [("B", t_, r) for t_, r in bre if gate_on(t_)] + \
           [("M", t_, r) for t_, r in mrt if not gate_on(t_)]
    bonly = [("B", t_, r) for t_, r in bre]                 # breakout regardless of regime

    def by_month(stream):
        mm = {}
        for eng, t_, r in stream:
            k = mon(t_); mm.setdefault(k, {"B": [], "M": []})[eng].append(r)
        return mm
    cm = by_month(comb); bm = by_month(bonly)

    months = sorted(set(list(cm) + list(bm)))
    months = [m for m in months if m >= "2024-01"]
    print(f"عتبة العرض = {thr:.0f}%   |   اختراق={len(bre)}  ارتداد={len(mrt)}\n")
    print(f"{'شهر':<9}{'محرّك':>7}{'صفقات':>7}{'عائد/صفقة':>11}   ||  {'اختراق فقط':>11}")
    print("-"*58)
    cl = []; bl = []
    for k in months:
        c_ = cm.get(k, {"B": [], "M": []}); b_ = bm.get(k, {"B": [], "M": []})
        cb = c_["B"]; cmr = c_["M"]
        if len(cb) >= len(cmr):
            eng = "اختراق"; arr = cb
        else:
            eng = "ارتداد"; arr = cmr
        # combined month return = mean of all trades active that month
        allc = cb + cmr; cret = np.mean(allc) if allc else float("nan")
        ball = b_["B"]; bret = np.mean(ball) if ball else float("nan")
        if allc: cl.append(cret)
        if ball: bl.append(bret)
        cflag = "" if not np.isfinite(cret) or cret >= 0 else "  ←سالب"
        print(f"{k:<9}{eng:>7}{len(allc):>7}{cret:>+10.1f}%   ||  {bret:>+10.1f}%{cflag}")
    cl = np.array(cl); bl = np.array(bl)
    print("-"*58)
    print(f"المُجمّع : متوسط شهري {np.nanmean(cl):+.1f}%  |  أشهر سالبة {(cl<0).sum()}/{len(cl)}")
    print(f"اختراق فقط: متوسط شهري {np.nanmean(bl):+.1f}%  |  أشهر سالبة {(bl<0).sum()}/{len(bl)}")
    print("\nالهدف: المُجمّع يقلّل الأشهر السالبة بتغطية الجفاف بالارتداد بدل الجلوس/الخسارة.")
    print("⚠️ متوسط لكل صفقة (وزن متساوٍ)؛ متفائل بانحياز البقاء (مخفّف بالسيولة).")
    print("\nDONE_COMB.", flush=True)


if __name__ == "__main__":
    main()
