#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Velocity probe — CHUNKED & RESUMABLE.

Same question as before (do pumps show an early 15m velocity burst that
separates them from non-pumps?) but processed in symbol-chunks. Each chunk's
partial counts are saved to disk; if the run stops, rerun and it resumes from
the next unfinished chunk. A final pass aggregates all chunks and prints the
discrimination report.

Run repeatedly until it prints DONE_VELO:  python experiments/velocity_probe_chunked.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from binance_sim import pit_universe as PIT               # noqa: E402
from binance_sim import hires_data as HR                  # noqa: E402

FWD = 16            # 16 x 15m = 4h forward
PUMP = 20.0
CHUNK = 40          # symbols per chunk
OUT = "results_velo"
START, END = "2025-04-01", "2025-05-01"


def parse(d):
    return int(pd.Timestamp(d, tz="UTC").timestamp() * 1000)


def process_symbol(sym, start, end):
    warm = start - 5 * 24 * 3600 * 1000
    df = HR.load_range(sym, "15m", warm, end)
    if df is None or len(df) < 60:
        return None
    c = df["close"].to_numpy(float); h = df["high"].to_numpy(float)
    v = df["volume"].to_numpy(float); t = df["time"].to_numpy()
    vavg = pd.Series(v).rolling(40).mean().shift(1).to_numpy()
    n = len(c)
    # accumulate sums (so chunks can be merged): for each metric keep pump/normal
    acc = {"pv1": [], "pv2": [], "pv4": [], "pvol": [],
           "nv1": [], "nv2": [], "nv4": [], "nvol": []}
    for i in range(8, n - FWD):
        if not (start <= t[i] <= end):
            continue
        fwd = (h[i + 1:i + 1 + FWD].max() / c[i] - 1) * 100
        pump = fwd >= PUMP
        norm = fwd < 5.0
        if not (pump or norm):
            continue
        v1 = (c[i] / c[i - 1] - 1) * 100
        v2 = (c[i] / c[i - 2] - 1) * 100
        v4 = (c[i] / c[i - 4] - 1) * 100
        vr = v[i] / vavg[i] if vavg[i] > 0 else 0
        p = "p" if pump else "n"
        acc[f"{p}v1"].append(v1); acc[f"{p}v2"].append(v2)
        acc[f"{p}v4"].append(v4); acc[f"{p}vol"].append(vr)
    return acc


def main():
    os.makedirs(OUT, exist_ok=True)
    start, end = parse(START), parse(END)
    syms = PIT.list_all_usdt_symbols()
    chunks = [syms[i:i + CHUNK] for i in range(0, len(syms), CHUNK)]
    print(f"{len(syms)} symbols in {len(chunks)} chunks of {CHUNK}", flush=True)

    for ci, chunk in enumerate(chunks):
        cf = f"{OUT}/chunk_{ci:03d}.json"
        if os.path.exists(cf):
            continue
        merged = {k: [] for k in ("pv1", "pv2", "pv4", "pvol",
                                  "nv1", "nv2", "nv4", "nvol")}
        for sym in chunk:
            a = process_symbol(sym, start, end)
            if a:
                for k in merged:
                    merged[k].extend(a[k])
        # store compact summary per chunk (counts + sums + sumsq) to keep files small
        out = {}
        for k, arr in merged.items():
            arr = np.array(arr, dtype=float)
            out[k] = [len(arr), float(arr.sum()), float((arr * arr).sum()),
                      [float(x) for x in arr[:0]]]  # no raw; stats only
            # keep thresholds counts for catch-rate
            out[k + "_ge"] = {str(thr): int((arr >= thr).sum()) for thr in (3, 5, 8)}
        json.dump(out, open(cf, "w"))
        np_pump = out["pvol"][0]
        print(f"  chunk {ci+1}/{len(chunks)} done (pump bars +vol n={np_pump})", flush=True)

    # all chunks present? aggregate
    files = [f"{OUT}/chunk_{i:03d}.json" for i in range(len(chunks))]
    if not all(os.path.exists(f) for f in files):
        missing = sum(1 for f in files if not os.path.exists(f))
        print(f"\n{missing} chunks still missing — rerun to resume.", flush=True)
        return

    agg = {}
    for f in files:
        d = json.load(open(f))
        for k, val in d.items():
            if k.endswith("_ge"):
                agg.setdefault(k, {}).update(
                    {t: agg.get(k, {}).get(t, 0) + val[t] for t in val})
            else:
                if k not in agg:
                    agg[k] = [0, 0.0, 0.0]
                agg[k][0] += val[0]; agg[k][1] += val[1]; agg[k][2] += val[2]

    def mean_std(stat):
        n, s, ss = stat[0], stat[1], stat[2]
        if n == 0:
            return 0, 0
        m = s / n
        var = max(ss / n - m * m, 0)
        return m, var ** 0.5

    print("\n=== VELOCITY at launch (15m): pump vs normal (all chunks) ===")
    print(f"{'metric':<14}{'PUMP mean':>11}{'NORMAL mean':>13}{'separation':>12}")
    print("-" * 50)
    for key, name in [("v1", "vel 15min"), ("v2", "vel 30min"),
                      ("v4", "vel 60min"), ("vol", "vol burst x")]:
        pm, ps = mean_std(agg["p" + key])
        nm, ns = mean_std(agg["n" + key])
        sd = (ps + ns) / 2 or 1
        sep = (pm - nm) / sd
        flag = "  <== strong" if abs(sep) > 0.4 else ""
        print(f"{name:<14}{pm:>10.2f}%{nm:>12.2f}%{sep:>11.2f}{flag}")

    print("\n=== EARLY-CATCH potential (velocity >= X) ===")
    for key in ("v2", "v4"):
        pge = agg.get("p" + key + "_ge", {}); nge = agg.get("n" + key + "_ge", {})
        pn = agg["p" + key][0]; nn = agg["n" + key][0]
        for thr in ("3", "5", "8"):
            pc = pge.get(thr, 0) / pn * 100 if pn else 0
            nc = nge.get(thr, 0) / nn * 100 if nn else 0
            lbl = "30min" if key == "v2" else "60min"
            print(f"  {lbl} vel>={thr}% : catches {pc:.0f}% of pumps, "
                  f"false on {nc:.0f}% of normals")
    print("\nDONE_VELO", flush=True)


if __name__ == "__main__":
    main()
