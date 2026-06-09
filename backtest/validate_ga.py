#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""يعيد تحقّق أفضل جينوم من GA على كامل العملات عبر 2023/2024/2025."""
import json
import numpy as np
from backtest import engine as E
from backtest import optimizer as O
from backtest.experiments import build_stable_ratio_map
from backtest.engine import Backtester, PLATFORM

GENOME = json.load(open('backtest/output/ga_best.json'))['genome']
CAP = 5000.0


def run_year(year):
    bt = Backtester(start_capital=CAP, year=year, verbose=False)
    bt.preload_4h(workers=6)
    bt.preload_daily(workers=6)
    bt.stable_times, bt.stable_ratio = build_stable_ratio_map(year)
    O.apply_genome(bt, GENOME)
    bt.run()
    h = bt.state[PLATFORM]['history']
    pnls = np.array([t.get('net_pnl', 0.0) for t in h], dtype=float)
    gp = float(pnls[pnls > 0].sum()); gl = float(-pnls[pnls <= 0].sum())
    eq = bt.state[PLATFORM]['liquid_capital']
    return {
        'year': year,
        'sharpe': O._sharpe(bt.equity_log, CAP),
        'return_pct': (eq - CAP) / CAP * 100,
        'pf': (gp / gl) if gl > 0 else 9.99,
        'maxdd': O._maxdd(bt.equity_log, CAP),
        'trades': len(h),
        'wins': int((pnls > 0).sum()),
    }


if __name__ == '__main__':
    print("التحقّق من جينوم GA على كامل العملات:")
    print(json.dumps(GENOME, ensure_ascii=False))
    res = []
    for y in [2023, 2024, 2025]:
        m = run_year(y)
        res.append(m)
        tag = 'TEST' if y == 2025 else 'train'
        print(f"  {y} [{tag:5s}] Sharpe={m['sharpe']:+.2f} | عائد={m['return_pct']:+.1f}% | "
              f"PF={m['pf']:.2f} | DD={m['maxdd']:.1f}% | صفقات={m['trades']} | "
              f"WR={m['wins']/max(m['trades'],1)*100:.0f}%", flush=True)
    json.dump(res, open('backtest/output/ga_validate_full.json', 'w'),
              ensure_ascii=False, indent=2)
    print("=== تم — حُفظ ga_validate_full.json ===")
