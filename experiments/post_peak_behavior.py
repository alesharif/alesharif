#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""POST-PEAK behavior — after a pump tops, do conditions stay 'flipped'?

The user's insight: distinguishing the real top's sell-burst from a mid-pump
pullback is hard live. BUT if after the true peak the move STAYS DOWN (rarely
re-pumps), then exiting on a coincident sell signal is SAFE (we miss no
re-pump). A mid-pump pullback instead recovers quickly to a new high.

For each real pump (taken trades, MFE>=10%) we find the price peak, then measure:
  * re-pump?  : after dropping >=5% from the peak, does price return to within
                1% of the peak again inside the window?  (low share = tops final)
  * dd_60/240 : drawdown from peak after 1h / 4h
  * flip_dur  : minutes price stays below peak-5% before any recovery to peak-2%
  * of_persist: is executed order flow (CVD) net-negative for the 60min after peak?

If re-pump share is LOW and flip_dur LONG -> tops are 'final/persistent' -> a
confirmed-flip exit is reliable even if only coincident.
Run:  python experiments/post_peak_behavior.py
"""

from __future__ import annotations

import gc, os, sys, io, urllib.request, zipfile, socket
socket.setdefaulttimeout(60)
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from binance_sim import paper_bot_strategy as S          # noqa: E402
from binance_sim import portfolio_backtest as PB          # noqa: E402
from binance_sim import pit_universe as PIT               # noqa: E402
from binance_sim import hires_data as HR                  # noqa: E402

MAX_CONC = 8
FOUR = 4 * 3600 * 1000
HOLD_H = 48
FEAR = 1.15
ATR_MIN = 0.05
ADX_MIN = 40.0
MFE_MIN = 10.0
AGG_DIR = "data/cache/aggtrades"

MONTHS = {"2026-04": ("2026-04-01", "2026-05-01"),
          "2026-05": ("2026-05-01", "2026-06-01")}


def parse(d):
    return int(pd.Timestamp(d, tz="UTC").timestamp() * 1000)


def load_agg(sym, start_ms, end_ms):
    days = pd.date_range(pd.Timestamp(start_ms, unit="ms").floor("D"),
                         pd.Timestamp(end_ms, unit="ms").floor("D"), freq="D")
    T = []; Q = []; SE = []
    for d in days:
        ds = d.strftime("%Y-%m-%d"); cf = f"{AGG_DIR}/{sym}-{ds}.npz"
        if os.path.exists(cf):
            try:
                z = np.load(cf); T.append(z["t"]); Q.append(z["q"]); SE.append(z["s"]); continue
            except Exception:
                pass
        return None   # only use already-cached agg from prior run
    if not T:
        return None
    t = np.concatenate(T); q = np.concatenate(Q); s = np.concatenate(SE)
    if len(t) == 0:
        return None
    if t.max() > 1e14:
        t = t // 1000
    m = (t >= start_ms) & (t <= end_ms)
    return t[m], q[m], s[m]


def reconstruct(mname, s, e):
    start, end = parse(s), parse(e); ff = start - PB.WARMUP_BARS * FOUR
    raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), ff, end, log=lambda *a: None)
    ps = {sym: S.attach_entry_signal(S.compute_features(df.copy())).set_index("close_time")
          for sym, df in raw.items()}
    times = sorted({int(t) for df in ps.values() for t in df.index if start <= t <= end})
    sr = PB.build_stable_ratio(ff, end)
    sblock = {t: (np.isfinite(v) and v > FEAR) for t, v in sr.items()}
    trades = []; open_until = {}
    for t in times:
        open_until = {sy: u for sy, u in open_until.items() if u > t}
        if sblock.get(t, False) or len(open_until) >= MAX_CONC:
            continue
        cands = []
        for sym, df in ps.items():
            if sym in open_until or t not in df.index:
                continue
            row = df.loc[t]
            if not bool(row["entry_signal"]):
                continue
            price = float(row["close"]); atr = float(row["atr"]); adx = float(row["adx"])
            if atr / price < ATR_MIN or adx < ADX_MIN:
                continue
            cands.append((sym, price, float(row["vol_pit"])))
        cands.sort(key=lambda x: x[2], reverse=True)
        for sym, price, _ in cands[:MAX_CONC - len(open_until)]:
            open_until[sym] = t + HOLD_H * 3600 * 1000
            trades.append((sym, t, price))
    del ps, raw; gc.collect()
    return trades


def main():
    repump = []; dd60 = []; dd240 = []; flipdur = []; ofpersist = []
    n = 0
    for mname, (s, e) in MONTHS.items():
        trades = reconstruct(mname, s, e)
        print(f"  {mname}: {len(trades)} trades", flush=True)
        for sym, t0, ep in trades:
            d = HR.load_range(sym, "1m", t0 + 1, t0 + HOLD_H * 3600 * 1000)
            if d is None or len(d) < 120:
                continue
            hi = d["high"].to_numpy(float); lo = d["low"].to_numpy(float)
            cl = d["close"].to_numpy(float); tm = d["time"].to_numpy()
            mfe = (hi.max() / ep - 1) * 100
            if mfe < MFE_MIN:
                continue
            n += 1
            pk = int(np.argmax(hi)); peak = hi[pk]; peak_t = int(tm[pk])
            post_hi = hi[pk + 1:]; post_lo = lo[pk + 1:]
            # re-pump: after dropping >=5% from peak, does it get back within 1% of peak?
            rp = 0
            if len(post_lo):
                dropped = np.where(post_lo <= peak * 0.95)[0]
                if len(dropped):
                    after = post_hi[dropped[0]:]
                    rp = 1 if (len(after) and after.max() >= peak * 0.99) else 0
                else:
                    rp = 0   # never dropped 5% -> still up there (not a clean top yet)
            repump.append(rp)
            # drawdown after 1h / 4h
            def dd(mins):
                seg = post_lo[:mins]
                return (seg.min() / peak - 1) * 100 if len(seg) else 0.0
            dd60.append(dd(60)); dd240.append(dd(240))
            # flip duration: minutes below peak*0.95 before any return to peak*0.98
            fd = 0
            for j in range(len(post_lo)):
                if post_lo[j] <= peak * 0.95:
                    fd += 1
                if post_hi[j] >= peak * 0.98 and fd > 0:
                    break
            flipdur.append(fd)
            # order-flow persistence (if aggTrades cached): CVD net over 60min after peak
            ag = load_agg(sym, peak_t, peak_t + 60 * 60 * 1000)
            if ag is not None and len(ag[0]) > 20:
                at, aq, asell = ag
                net = np.where(asell == 1, -aq, aq).sum()
                ofpersist.append(1 if net < 0 else 0)
        gc.collect()

    if not repump:
        print("no pumps."); return
    rp = np.array(repump); print(f"\n##### POST-PEAK BEHAVIOR ({n} pumps, MFE>={MFE_MIN}%) #####")
    print(f"re-pump share (drops 5% then returns to peak): {rp.mean()*100:.0f}%   "
          f"=> tops are FINAL {100-rp.mean()*100:.0f}% of the time")
    print(f"median drawdown 1h after peak:  {np.median(dd60):.1f}%")
    print(f"median drawdown 4h after peak:  {np.median(dd240):.1f}%")
    print(f"median flip duration (min below peak-5%): {np.median(flipdur):.0f} min")
    print(f"share staying down >2h after peak: {(np.array(flipdur)>120).mean()*100:.0f}%")
    if ofpersist:
        print(f"order-flow net-SELL in 60min after peak: {np.mean(ofpersist)*100:.0f}% of pumps "
              f"(n={len(ofpersist)})")
    print("\nتفسير: re-pump منخفض + flip duration طويل = القمم نهائية وتبقى منقلبة")
    print("       => الخروج عند تأكّد الانقلاب آمن (لا نفوّت إعادة انفجار)")
    print("\nDONE_POSTPEAK.", flush=True)


if __name__ == "__main__":
    main()
