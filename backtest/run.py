#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run.py — نقطة تشغيل المحاكاة التاريخية.

أمثلة:
  python3 -m backtest.run --capital 5000 --year 2025
  python3 -m backtest.run --capital 5000 --year 2025 --max-symbols 15   # تجربة سريعة
"""
import argparse
import time

from .engine import Backtester
from .report import build_report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--capital', type=float, default=5000.0)
    ap.add_argument('--year', type=int, default=2025)
    ap.add_argument('--max-symbols', type=int, default=None,
                    help='حدّ عدد العملات (للتجربة السريعة)')
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()

    t0 = time.time()
    bt = Backtester(start_capital=args.capital, year=args.year,
                    max_symbols=args.max_symbols, verbose=not args.quiet)
    bt.preload_4h(workers=args.workers)
    bt.preload_daily(workers=args.workers)
    bt.run()
    elapsed = time.time() - t0

    summary = build_report(bt.state, bt.equity_log, args.capital, year=args.year)
    print()
    print(summary)
    print(f"\n⏱️ زمن التشغيل: {elapsed/60:.1f} دقيقة")


if __name__ == '__main__':
    main()
