#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Is there a SIGN that a falling breakout trade has bottomed? For down-first
breakout entries (2024->2026-03), find the true bottom (4h, 180d), and test
whether a CAPITULATION signature marks it: volume climax (vol vs 30-bar avg) and
RSI extreme AT the bottom vs DURING the fall. High lift => detectable bottom.
NOTE: survivorship — coins that died (no bottom) are absent; inflates any signal.
Run:  python experiments/bottom_signature.py
"""

from __future__ import annotations

import gc, sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from binance_sim import pit_universe as PIT               # noqa: E402

LIQ_MIN = 300_000; HOLD_MS = 180*24*3600*1000
S, E = "2023-08-01", "2026-06-01"
SIG_FROM = int(pd.Timestamp("2024-01-01", tz="UTC").timestamp()*1000)
SIG_TO = int(pd.Timestamp("2026-03-01", tz="UTC").timestamp()*1000)


def ms(s): return int(pd.Timestamp(s, tz="UTC").timestamp()*1000)
def ema(a, n): return pd.Series(a).ewm(span=n, adjust=False).mean().to_numpy()


def rsi(c, n=14):
    d = np.diff(c, prepend=c[0]); up = np.where(d > 0, d, 0.0); dn = np.where(d < 0, -d, 0.0)
    ru = pd.Series(up).ewm(alpha=1/n, adjust=False).mean().to_numpy()
    rd = pd.Series(dn).ewm(alpha=1/n, adjust=False).mean().to_numpy()
    return 100 - 100/(1 + ru/np.where(rd == 0, 1e-9, rd))


def main():
    print("Bottom signature: does a capitulation (volume/RSI) mark the bottom?\n", flush=True)
    raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), ms(S), ms(E), log=lambda *a: None)
    bot_vol = []; fall_vol = []; bot_rsi = []; bounce_after = []
    nb = 0
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
        # 4h arrays
        T = df["time"].to_numpy(); H = df["high"].to_numpy(float); L = df["low"].to_numpy(float)
        C4 = df["close"].to_numpy(float); V4 = df["volume"].to_numpy(float)
        vma = pd.Series(V4).rolling(30).mean().to_numpy()
        r4 = rsi(C4, 14)
        for i in range(200, n-1):
            if not (SIG_FROM <= int(t[i]) <= SIG_TO):
                continue
            if not (np.isfinite(q25[i-1]) and bbw[i-1] <= q25[i-1] and bbw[i] > bbw[i-1]
                    and hist[i] > hist[i-1] > hist[i-2] and macd[i] > macd[i-1]
                    and abs(e20[i]-e50[i])/c[i] < 0.03 and c[i] > o[i]
                    and c[i] > e200[i] and i >= 99 and c[i] >= c[i-99]):
                continue
            if np.nanmean(dv[max(0, i-30):i]) <= LIQ_MIN:
                continue
            P0 = c[i]; ent_t = int(t[i])
            j0 = np.searchsorted(T, ent_t, side="right"); j1 = np.searchsorted(T, ent_t+HOLD_MS, side="right")
            seg_l = L[j0:j1]
            if len(seg_l) < 10:
                continue
            dn = np.argmax(seg_l <= P0*0.95) if (seg_l <= P0*0.95).any() else 10**9
            up = np.argmax(H[j0:j1] >= P0*1.05) if (H[j0:j1] >= P0*1.05).any() else 10**9
            if not (dn < up):                          # down-first only
                continue
            botpos = int(np.argmin(seg_l)); botidx = j0 + botpos
            if botpos < 3 or vma[botidx] <= 0 or not np.isfinite(vma[botidx]):
                continue
            nb += 1
            bot_vol.append(V4[botidx]/vma[botidx])     # volume climax ratio at bottom
            bot_rsi.append(r4[botidx])
            # fall bars (entry -> just before bottom): their volume ratios
            for jj in range(j0, botidx):
                if np.isfinite(vma[jj]) and vma[jj] > 0:
                    fall_vol.append(V4[jj]/vma[jj])
            # bounce after bottom (high after bottom / bottom low)
            aft = H[botidx+1:j1]
            bounce_after.append((aft.max()/seg_l[botpos]-1)*100 if len(aft) else 0.0)
        df.drop(columns=["dt"], inplace=True, errors="ignore")
    del raw; gc.collect()

    bv = np.array(bot_vol); fv = np.array(fall_vol); br = np.array(bot_rsi)
    print(f"عدد القيعان المفحوصة: {nb}\n")
    print("##### علامة الحجم (Capitulation) #####")
    print(f"  متوسّط نسبة الحجم عند القاع:     {bv.mean():.2f}×")
    print(f"  متوسّط نسبة الحجم أثناء النزول:  {fv.mean():.2f}×")
    print(f"  انفجار حجم >3× عند القاع:        {(bv>3).mean()*100:.0f}%")
    print(f"  انفجار حجم >3× أثناء النزول:     {(fv>3).mean()*100:.0f}%")
    lift = (bv>3).mean()/max((fv>3).mean(),1e-9)
    print(f"  >>> Lift = {lift:.1f}×  (>2 = علامة مفيدة؛ ~1 = لا علامة)")
    print(f"\n##### RSI عند القاع (4h) #####")
    print(f"  متوسّط RSI عند القاع: {br.mean():.0f}   |   الوسيط: {np.median(br):.0f}")
    print(f"  RSI<20 عند القاع: {(br<20).mean()*100:.0f}%   RSI<30: {(br<30).mean()*100:.0f}%")
    print("\nملاحظة: انحياز البقاء — العملات التي ماتت بلا قاع غائبة (يضخّم أي علامة).")
    print("\nDONE_BOTSIG.", flush=True)


if __name__ == "__main__":
    main()
