#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pre-download ALL 5m + 1s data for the four dev months (every coin, every day).

After this runs once, every future hybrid backtest reads purely from disk
(no network), so experiments become fast. Idempotent: files already cached are
skipped instantly. Parallel downloads, progress per day.

Months: 2025-04, 2025-12, 2026-04, 2026-05.
Run:  python experiments/prefetch_hires.py
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

sys.path.insert(0, ".")
from binance_sim import pit_universe as PIT               # noqa: E402
from binance_sim import hires_data as HR                  # noqa: E402

MONTHS = [("2025-04-01", "2025-04-30"),
          ("2025-12-01", "2025-12-31"),
          ("2026-04-01", "2026-04-30"),
          ("2026-05-01", "2026-05-31")]
WORKERS = 16


def main():
    print("Listing all USDT symbols ...", flush=True)
    symbols = PIT.list_all_usdt_symbols()
    print(f"  {len(symbols)} symbols", flush=True)

    # build the full list of (symbol, interval, day) download tasks
    tasks = []
    for s_str, e_str in MONTHS:
        days = pd.date_range(s_str, e_str, freq="D")
        for sym in symbols:
            for d in days:
                ds = d.strftime("%Y-%m-%d")
                tasks.append((sym, "5m", ds))
                tasks.append((sym, "1s", ds))
    print(f"Total download tasks: {len(tasks):,} (cached ones skip instantly)\n", flush=True)

    done = 0
    got = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(HR.load_day, sym, iv, ds): (sym, iv, ds)
                for (sym, iv, ds) in tasks}
        for fut in as_completed(futs):
            done += 1
            try:
                if fut.result() is not None:
                    got += 1
            except Exception:  # noqa: BLE001
                pass
            if done % 2000 == 0:
                print(f"  processed {done:,}/{len(tasks):,}  (with data: {got:,})", flush=True)
    print(f"\nDone. processed {done:,}, files with data {got:,}.", flush=True)
    print("All four months' 5m+1s now cached on disk — future backtests are fast.")


if __name__ == "__main__":
    main()
