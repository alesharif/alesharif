#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Third strategy candidate: WEEKLY-RSI MEAN-REVERSION for ranging/drought markets.

Breakout fails in the 2025 drought (no follow-through). Mean-reversion is the
complementary tool: buy the oversold bounce inside a range. Entry = weekly RSI(14)
crosses up out of oversold (<=35 -> >35), enter at weekly close. Exit = first
touch of TP / SL (60d window), net of 1% round-trip + 1% stop slippage. Liquid
coins only. Reported per regime: FULL | 2024 (trend) | 2025 (DROUGHT).
Hypothesis: positive in 2025 where breakout was negative.
Run:  python experiments/mean_reversion.py
"""

from __future__ import annotations

import gc, sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from binance_sim import pit_universe as PIT               # noqa: E402

WINDOW_MS = 60*24*3600*1000; LIQ_MIN = 300_000
S, E, SF = "2022-06-01", "2026-06-01", "2022-09-01"
TPS = [0.15, 0.25, 0.40]; SLS = [0.10, 0.15]
COST = 1.0; STOP_SLIP = 1.0


def ms(s): return int(pd.Timestamp(s, tz="UTC").timestamp()*1000)


def rsi(c, n=14):
    d = np.diff(c, prepend=c[0]); up = np.where(d > 0, d, 0.0); dn = np.where(d < 0, -d, 0.0)
    ru = pd.Series(up).ewm(alpha=1/n, adjust=False).mean().to_numpy()
    rd = pd.Series(dn).ewm(alpha=1/n, adjust=False).mean().to_numpy()
    return 100 - 100/(1 + ru/np.where(rd == 0, 1e-9, rd))


def main():
    print("Weekly-RSI MEAN-REVERSION (3rd strategy) per regime\n", flush=True)
    raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), ms(S), ms(E), log=lambda *a: None)
    res = {(tp, sl): {"full": [], 2024: [], 2025: []} for tp in TPS for sl in SLS}
    nsig = 0
    for sym, df in raw.items():
        df = df.sort_values("time").reset_index(drop=True)
        df["dt"] = pd.to_datetime(df["time"], unit="ms")
        g = df.set_index("dt")
        wk = pd.DataFrame({"c": g["close"].resample("W").last(),
                           "v": g["volume"].resample("W").sum(),
                           "t": g["time"].resample("W").last()}).dropna()
        wc = wk["c"].to_numpy(); wt = wk["t"].to_numpy(); wv = wk["v"].to_numpy()
        if len(wc) < 25:
            df.drop(columns=["dt"], inplace=True, errors="ignore"); continue
        r = rsi(wc, 14); wdv = wc*wv
        T = df["time"].to_numpy(); H = df["high"].to_numpy(float); L = df["low"].to_numpy(float); C = df["close"].to_numpy(float)
        for w in range(15, len(wc)):
            if not (r[w-1] <= 35 and r[w] > 35):           # weekly RSI exits oversold
                continue
            if int(wt[w]) < ms(SF):
                continue
            if np.nanmean(wdv[max(0, w-4):w]) <= LIQ_MIN:  # ~last month liquidity
                continue
            ent_t, ent_px = int(wt[w]), float(wc[w]); j0 = np.searchsorted(T, ent_t)
            if j0 >= len(T):
                continue
            nsig += 1
            end = min(j0 + int(np.searchsorted(T[j0:], ent_t+WINDOW_MS)) + 1, len(T))
            Hs = H[j0:end]; Ls = L[j0:end]; lastc = C[end-1] if end > j0 else ent_px
            yr = int(pd.Timestamp(ent_t, unit="ms").strftime("%Y"))
            for tp in TPS:
                hb = Hs >= ent_px*(1+tp); tph = int(np.argmax(hb)) if hb.any() else 10**9
                for sl in SLS:
                    lb = Ls <= ent_px*(1-sl); slh = int(np.argmax(lb)) if lb.any() else 10**9
                    if slh <= tph and slh < 10**9:
                        nr = -sl*100 - COST - STOP_SLIP
                    elif tph < 10**9:
                        nr = tp*100 - COST
                    else:
                        nr = (lastc/ent_px - 1)*100 - COST
                    res[(tp, sl)]["full"].append(nr)
                    if yr in (2024, 2025):
                        res[(tp, sl)][yr].append(nr)
        df.drop(columns=["dt"], inplace=True, errors="ignore")
    del raw; gc.collect()

    print(f"  {nsig} weekly-RSI oversold-bounce signals\n")
    print(f"{'TP':>5}{'SL':>5}{'FULL':>9}{'2024':>9}{'2025(drought)':>15}")
    print("-" * 44)
    for tp in TPS:
        for sl in SLS:
            r = res[(tp, sl)]
            def m(k):
                a = np.array(r[k]); return a.mean() if len(a) else float("nan")
            mark = "  <== +2025" if m(2025) > 0 else ""
            print(f"{int(tp*100):>4}%{int(sl*100):>4}%{m('full'):>+8.1f}%{m(2024):>+8.1f}%{m(2025):>+13.1f}%{mark}")
    print("\nموجب في 2025 => الارتداد يربح في الجفاف => استراتيجية ثالثة مكمّلة.")
    print("سالب في 2025 => 2025 كان هبوطاً لا نطاقاً (الارتداد = سكين ساقط أيضاً).")
    print("\nDONE_MR.", flush=True)


if __name__ == "__main__":
    main()
