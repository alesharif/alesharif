#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Portfolio equity curve for 2025 ONLY — the realistic forward regime (drought).

User's point: the future likely looks like 2025 (drought), not 2024's alt-season
explosion. So judge the system on 2025 alone: entries in 2025-01..2025-12, real
capital, concurrency cap, compounding. Does it make money in a drought year?

Systems: breakout-only | breakout+gate | combined(+MR) | MR-only.
Run:  python experiments/portfolio_2025.py
"""

from __future__ import annotations

import gc, sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from binance_sim import pit_universe as PIT               # noqa: E402

LIQ_MIN = 300_000; DAY = 86400000
S, E, SF = "2022-06-01", "2026-06-01", "2024-06-01"
BWIN = 120*24*3600*1000; B_SL = 0.12; B_TP = 2.0
MWIN = 60*24*3600*1000; M_TP = 0.15; M_SL = 0.15; OS = 30; COST = 0.01; STOP_SLIP = 0.01
START = 10_000.0; SLOTS = 8; ALLOC = 0.10
W0 = "2025-01-01"; W1 = "2026-01-01"          # the test window (entries inside)


def ms(s): return int(pd.Timestamp(s, tz="UTC").timestamp()*1000)
def ema(a, n): return pd.Series(a).ewm(span=n, adjust=False).mean().to_numpy()


def rsi(c, n=14):
    d = np.diff(c, prepend=c[0]); up = np.where(d > 0, d, 0.0); dn = np.where(d < 0, -d, 0.0)
    ru = pd.Series(up).ewm(alpha=1/n, adjust=False).mean().to_numpy()
    rd = pd.Series(dn).ewm(alpha=1/n, adjust=False).mean().to_numpy()
    return 100 - 100/(1 + ru/np.where(rd == 0, 1e-9, rd))


def main():
    print("Portfolio equity — 2025 ONLY (drought = realistic forward regime)\n", flush=True)
    raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), ms(S), ms(E), log=lambda *a: None)
    w0, w1 = ms(W0), ms(W1)
    runs = {}; tots = {}; sigs = []        # (ent_t, exit_t, ret_frac, engine)
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
            if not (w0 <= int(t[i]) < w1):
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
                if not (wr[w-1] <= OS and wr[w] > OS and w0 <= int(wt[w]) < w1):
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
    thr = np.nanmedian([supply(s[0]) for s in sigs if s[3] == "B"]) if any(s[3]=="B" for s in sigs) else 10.0
    gate_on = lambda tt: supply(tt) >= thr

    def run(flt, label):
        ss = sorted(sigs, key=lambda x: x[0]); equity = START; open_pos = []; eq_t = []; eq_v = []; ntk = 0
        for ent_t, exit_t, rr, eng in ss:
            open_pos.sort()
            while open_pos and open_pos[0][0] <= ent_t:
                xt, pnl = open_pos.pop(0); equity += pnl; eq_t.append(xt); eq_v.append(equity)
            if not flt(ent_t, eng) or len(open_pos) >= SLOTS:
                continue
            stake = equity*ALLOC
            if stake <= 0:
                continue
            equity -= stake; open_pos.append((exit_t, stake*(1+rr))); ntk += 1
        for xt, pnl in sorted(open_pos):
            equity += pnl; eq_t.append(xt); eq_v.append(equity)
        if not eq_v:
            print(f"  {label:<34} لا صفقات"); return
        eq_t = np.array(eq_t); eq_v = np.array(eq_v); order = np.argsort(eq_t)
        eq_t = eq_t[order]; eq_v = eq_v[order]
        s = pd.Series(eq_v, index=pd.to_datetime(eq_t, unit="ms")).resample("ME").last().dropna()
        rets = s.pct_change().dropna()*100
        peak = np.maximum.accumulate(eq_v); dd = ((eq_v-peak)/peak).min()*100
        pos = (rets > 0).mean()*100 if len(rets) else float("nan")
        print(f"  {label:<34} نهائي ${eq_v[-1]:>7,.0f}  ({(eq_v[-1]/START-1)*100:>+5.0f}%)  تراجع {dd:>+4.0f}%  موجب {pos:>3.0f}%  صفقات {ntk}")

    nb = sum(1 for s in sigs if s[3] == "B"); nm = sum(1 for s in sigs if s[3] == "M")
    print(f"نافذة 2025 فقط  |  عتبة العرض={thr:.0f}%  |  إشارات: اختراق {nb}  ارتداد {nm}")
    print(f"إعدادات: ${START:,.0f}، {SLOTS} متزامنة، {ALLOC*100:.0f}%/صفقة\n")
    run(lambda tt, e: e == "B", "اختراق فقط")
    run(lambda tt, e: e == "B" and gate_on(tt), "اختراق + بوابة")
    run(lambda tt, e: (e == "B" and gate_on(tt)) or (e == "M" and not gate_on(tt)), "المُجمّع (اختراق+ارتداد)")
    run(lambda tt, e: e == "M", "ارتداد فقط")
    print("\nالسؤال: هل أيّ نسخة تربح في سنة جفاف كاملة، أم الأصدق الجلوس؟")
    print("⚠️ متفائل بانحياز البقاء (مخفّف بالسيولة).")
    print("\nDONE_2025.", flush=True)


if __name__ == "__main__":
    main()
