#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments.py — تشغيل تجارب إصلاح الاستراتيجية (كل تجربة في عملية مستقلة).

كل تجربة = تعديلات على ثوابت strategy_core (S) و/أو بوابة نظام السوق (regime gate).
البيانات مخزّنة محلياً (لا تحميل) → كل تجربة ~10-12 دقيقة.

التشغيل:
  python3 -m backtest.experiments --exp E1
"""
import argparse
import time
import csv

import numpy as np

from . import strategy_core as S
from . import engine as E
from .engine import Backtester, PLATFORM


# ═══════════════════════════════════════════════════════════════════
# تعريف التجارب
#   overrides: قيم تُكتب على ثوابت S قبل التشغيل
#   regime:    None | 'ema50' | 'ema99'  (بوابة BTC: لا دخول إلا في اتجاه صاعد)
# ═══════════════════════════════════════════════════════════════════
EXPERIMENTS = {
    'BASE': dict(desc='الأساس (نسخة v3.27 كما هي)', overrides={}, regime=None),

    # ── معالجة جانب الخسارة ──
    'E1': dict(desc='Stop Loss أضيق -2.5%',
               overrides={'STOP_LOSS_PCT': -0.025}, regime=None),
    'E2': dict(desc='تفعيل Early Exit (6h / peak<2.5%)',
               overrides={'EARLY_EXIT_ENABLED': True}, regime=None),

    # ── معالجة جانب الربح (ترك الرابح يركض) ──
    'E3': dict(desc='التقاط الربح: TP +6%, trailing يعطي 1.5% فقط, تفعيل +2%',
               overrides={'TP_THRESHOLD': 0.06, 'TRAILING_RATIO': 0.985,
                          'TRAILING_ACTIVATE_PCT': 0.02}, regime=None),

    # ── فلتر نظام السوق (أعلى قناعة: 2025 كله نزفَ) ──
    'E4': dict(desc='بوابة BTC: دخول فقط عندما BTC ≥ EMA50 اليومي',
               overrides={}, regime='ema50'),

    # ── دمج أفضل التخمينات ──
    'E5': dict(desc='دمج: بوابة BTC + SL -2.5% + Early Exit',
               overrides={'STOP_LOSS_PCT': -0.025, 'EARLY_EXIT_ENABLED': True},
               regime='ema50'),
    'E6': dict(desc='دمج: بوابة BTC + التقاط ربح أفضل (E3)',
               overrides={'TP_THRESHOLD': 0.06, 'TRAILING_RATIO': 0.985,
                          'TRAILING_ACTIVATE_PCT': 0.02}, regime='ema50'),
}


class RegimeBacktester(Backtester):
    """نفس المحرك + بوابة نظام السوق (BTC trend) قبل أي scan."""
    regime_mode = None

    def _regime_ok(self, t_ms):
        sd = self.get_daily('BTCUSDT')
        if sd is None:
            return True
        di = self._daily_asof_idx(sd, t_ms)
        if di < 0:
            return True
        if self.regime_mode == 'ema50':
            return bool(sd.close[di] >= sd.ema50[di])
        if self.regime_mode == 'ema99':
            return bool(sd.close[di] >= sd.ema99[di])
        return True

    def scan(self, t_ms):
        if self.regime_mode and not self._regime_ok(t_ms):
            return []
        return super().scan(t_ms)


def _max_drawdown(equity_log, capital):
    if not equity_log:
        return 0.0, 0.0
    eq = np.array([e for _, e in equity_log], dtype=float)
    eq = np.concatenate([[capital], eq])
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak)
    i = int(np.argmin(dd))
    return float(dd[i]), float(dd[i] / peak[i] * 100) if peak[i] > 0 else 0.0


def compute_metrics(state, equity_log, capital):
    h = state[PLATFORM]['history']
    pnls = np.array([t.get('net_pnl', 0.0) for t in h], dtype=float)
    n = len(pnls)
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    gp = float(wins.sum())
    gl = float(-losses.sum())
    pf = (gp / gl) if gl > 0 else float('inf')
    final_eq = state[PLATFORM]['liquid_capital']
    dd_usd, dd_pct = _max_drawdown(equity_log, capital)
    return {
        'trades': n,
        'final_equity': final_eq,
        'return_pct': (final_eq - capital) / capital * 100,
        'wr': (len(wins) / n * 100) if n else 0.0,
        'pf': pf,
        'net': float(pnls.sum()),
        'maxdd_pct': dd_pct,
    }


def write_trades(state, path):
    h = state[PLATFORM]['history']
    cols = ['symbol', 'exit_reason', 'duration_h', 'size_usd', 'peak_pct',
            'net_pnl', 'pnl_pct']
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(cols)
        for t in h:
            w.writerow([t.get('symbol'), t.get('exit_reason'),
                        round(t.get('duration_h', 0), 2), round(t.get('size_usd', 0), 2),
                        round(t.get('peak_pct', 0), 2), round(t.get('net_pnl', 0), 4),
                        round(t.get('pnl_pct', 0), 3)])


def run_one(exp_id, capital=5000.0, year=2025, workers=4):
    cfg = EXPERIMENTS[exp_id]
    # تطبيق overrides على S (هذه العملية مستقلة → آمن)
    for k, v in cfg['overrides'].items():
        setattr(S, k, v)

    t0 = time.time()
    cls = RegimeBacktester if cfg['regime'] else Backtester
    bt = cls(start_capital=capital, year=year, verbose=False)
    if cfg['regime']:
        bt.regime_mode = cfg['regime']
    bt.preload_4h(workers=workers)
    bt.preload_daily(workers=workers)
    bt.run()
    m = compute_metrics(bt.state, bt.equity_log, capital)
    elapsed = (time.time() - t0) / 60

    out = (
        f"[{exp_id}] {cfg['desc']}\n"
        f"  overrides={cfg['overrides']} regime={cfg['regime']}\n"
        f"  العائد={m['return_pct']:+.1f}%  | نهائي=${m['final_equity']:,.0f}  | "
        f"صفقات={m['trades']}  | WR={m['wr']:.1f}%  | PF={m['pf']:.2f}  | "
        f"maxDD={m['maxdd_pct']:.1f}%  | ({elapsed:.1f}د)\n"
    )
    with open(f'backtest/output/exp_{exp_id}_summary.txt', 'w') as f:
        f.write(out)
    write_trades(bt.state, f'backtest/output/exp_{exp_id}_trades.csv')
    print(out, flush=True)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', required=True, choices=list(EXPERIMENTS.keys()))
    ap.add_argument('--capital', type=float, default=5000.0)
    ap.add_argument('--year', type=int, default=2025)
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()
    run_one(args.exp, args.capital, args.year, args.workers)


if __name__ == '__main__':
    main()
