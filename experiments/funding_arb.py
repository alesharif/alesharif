#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Funding-rate arbitrage (delta-neutral) — REAL historical funding from OKX.

Long spot + short perp of equal size => price-neutral; you RECEIVE funding every
8h when the funding rate is positive (you are short), and PAY when negative.
This measures the TRUE realized yield on the position notional, net of trading
fees, including the periods funding flips negative — no guru promises.

Two policies:
  * always-on : hold continuously, collect funding (pos adds, neg subtracts),
                pay one round-trip fee. Raw funding capture.
  * pos-only  : hold only while funding is positive; pay a round-trip fee EACH
                time you toggle in/out (avoids negative funding but costs fees).

Capital note: yield is on NOTIONAL. If spot+short need separate full funding
(2x capital, no unified margin), halve the % return.
Run:  python experiments/funding_arb.py
"""

from __future__ import annotations

import time, urllib.request, json
import numpy as np

INSTS = ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "XRP-USDT-SWAP", "SOL-USDT-SWAP"]
DAYS = 730                                   # ~2 years lookback
SPOT_FEE = 0.10                              # % taker per spot leg
FUT_FEE = 0.05                               # % taker per perp leg
RT_FEE = 2 * (SPOT_FEE + FUT_FEE)            # round-trip (enter+exit both legs) = 0.30%


def fetch_funding(inst, start_ms):
    """All (time_ms, rate) for inst back to start_ms, ascending."""
    out = {}; after = ""
    while True:
        url = (f"https://www.okx.com/api/v5/public/funding-rate-history?"
               f"instId={inst}&limit=100" + (f"&after={after}" if after else ""))
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            j = json.loads(urllib.request.urlopen(req, timeout=25).read())
        except Exception as e:
            print(f"   fetch err {inst}: {e}"); break
        data = j.get("data", [])
        if not data:
            break
        for d in data:
            out[int(d["fundingTime"])] = float(d["realizedRate"])
        oldest = min(int(d["fundingTime"]) for d in data)
        after = str(oldest)
        if oldest <= start_ms or len(data) < 100:
            break
        time.sleep(0.15)
    items = sorted((t, r) for t, r in out.items() if t >= start_ms)
    return items


def analyze(inst):
    start_ms = int((time.time() - DAYS * 86400) * 1000)
    items = fetch_funding(inst, start_ms)
    if len(items) < 50:
        print(f"{inst:<16} insufficient data ({len(items)})"); return
    t = np.array([x[0] for x in items]); r = np.array([x[1] for x in items])
    days = (t[-1] - t[0]) / 86400000
    yrs = days / 365.25
    n = len(r)
    neg_share = (r < 0).mean() * 100
    avg8h = r.mean() * 100

    # always-on: collect every period (sign-aware), one round-trip fee
    gross = r.sum() * 100                         # % of notional over the whole span
    net_on = gross - RT_FEE
    ann_on = net_on / yrs

    # pos-only: hold while positive; fee each time we toggle on
    pos = r > 0
    entries = int(np.sum(pos[1:] & ~pos[:-1]) + (1 if pos[0] else 0))   # runs of positive
    pos_sum = r[pos].sum() * 100
    net_pos = pos_sum - entries * RT_FEE
    ann_pos = net_pos / yrs

    print(f"{inst.replace('-USDT-SWAP',''):<6}{n:>6}{days:>6.0f}d"
          f"{neg_share:>7.0f}%{avg8h:>8.3f}%{ann_on:>9.1f}%{ann_on/12:>8.2f}%"
          f"{ann_pos:>9.1f}%{entries:>6}")


def main():
    print("FUNDING-RATE ARBITRAGE (delta-neutral) — real OKX funding, ~2y\n", flush=True)
    print(f"fees: spot {SPOT_FEE}%/leg, perp {FUT_FEE}%/leg, round-trip {RT_FEE}%\n")
    print(f"{'coin':<6}{'n8h':>6}{'span':>7}{'neg%':>7}{'avg8h':>8}"
          f"{'ANN_on':>9}{'/mo':>8}{'ANN_pos':>9}{'toggl':>6}")
    print("-" * 68)
    for inst in INSTS:
        try:
            analyze(inst)
        except Exception as e:
            print(f"{inst:<16} err {e}")
    print("-" * 68)
    print("ANN_on = عائد سنوي للتحييد المستمر (بعد رسوم دخول/خروج مرّة).")
    print("ANN_pos = سياسة 'فقط عند التمويل الموجب' (برسوم كل تبديل).")
    print("العائد على الحجم (notional). لو رأس المال مزدوج (بلا هامش موحّد) اقسم على 2.")
    print("\nDONE_FUNDING.", flush=True)


if __name__ == "__main__":
    main()
