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
import pandas as pd

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

    # ── فلتر "مقياس الخوف" عبر حجم العملات المستقرة (بديل عملي لفلتر BTC) ──
    'E7': dict(desc='Stable-Ratio Fear Filter @1.15 (وحده)',
               overrides={}, regime={'mode': 'stable', 'threshold': 1.15}),
    'E8': dict(desc='Stable-Ratio Fear @1.10 (أكثر حساسية)',
               overrides={}, regime={'mode': 'stable', 'threshold': 1.10}),
    'E9': dict(desc='Stable-Ratio Fear @1.30 (محافظ)',
               overrides={}, regime={'mode': 'stable', 'threshold': 1.30}),
    'E10': dict(desc='Stable-Ratio @1.15 + SL-2.5% + Early Exit',
                overrides={'STOP_LOSS_PCT': -0.025, 'EARLY_EXIT_ENABLED': True},
                regime={'mode': 'stable', 'threshold': 1.15}),

    # ── فلتر جودة الدخول لكل عملة (ATR/ADX على 4h) ──
    'E11': dict(desc='ATR/ADX جودة الدخول (ATR≥5% و ADX≥40) وحده',
                overrides={}, regime=None,
                quality={'atr_min': 0.05, 'adx_min': 40.0}),
    'E12': dict(desc='ATR/ADX أساسي (ATR≥2% و ADX≥25) وحده',
                overrides={}, regime=None,
                quality={'atr_min': 0.02, 'adx_min': 25.0}),
    'E13': dict(desc='ATR/ADX(5%/40) + Fear@1.15 + SL-2.5% + EarlyExit (كل شيء)',
                overrides={'STOP_LOSS_PCT': -0.025, 'EARLY_EXIT_ENABLED': True},
                regime={'mode': 'stable', 'threshold': 1.15},
                quality={'atr_min': 0.05, 'adx_min': 40.0}),
    'E14': dict(desc='ATR/ADX(5%/40) + التقاط ربح أفضل (TP+6%/trail1.5%/act+2%)',
                overrides={'TP_THRESHOLD': 0.06, 'TRAILING_RATIO': 0.985,
                           'TRAILING_ACTIVATE_PCT': 0.02}, regime=None,
                quality={'atr_min': 0.05, 'adx_min': 40.0}),

    # ── خروج ديناميكي مرتبط بالـ ATR (Chandelier) فوق فلتر الجودة ──
    # الهدف: التقاط الانفجارات الكبيرة بدل الخروج عند +2.4%
    'F1': dict(desc='Quality + ATR exit (sl×1.5, trail×2.5)',
               overrides={}, regime=None,
               quality={'atr_min': 0.05, 'adx_min': 40.0},
               atr_exit={'sl': 1.5, 'trail': 2.5, 'dead_h': None}),
    'F2': dict(desc='Quality + ATR exit (sl×2.0, trail×3.0) أوسع',
               overrides={}, regime=None,
               quality={'atr_min': 0.05, 'adx_min': 40.0},
               atr_exit={'sl': 2.0, 'trail': 3.0, 'dead_h': None}),
    'F3': dict(desc='Quality + ATR exit (sl×1.5, trail×2.0) أضيق trail',
               overrides={}, regime=None,
               quality={'atr_min': 0.05, 'adx_min': 40.0},
               atr_exit={'sl': 1.5, 'trail': 2.0, 'dead_h': None}),
    'F4': dict(desc='Quality + ATR exit (sl×1.5, trail×2.5) + dead-cut 48h',
               overrides={}, regime=None,
               quality={'atr_min': 0.05, 'adx_min': 40.0},
               atr_exit={'sl': 1.5, 'trail': 2.5, 'dead_h': 48}),

    # ── إشارة دخول جديدة: زخم/اختراق (بدل divergence المرتد) ──
    'G1': dict(desc='اختراق Donchian20 (وحده، بدل divergence)',
               overrides={}, regime=None,
               entry={'mode': 'breakout', 'lookback': 20, 'mom': 10}),
    'G2': dict(desc='اختراق + فلتر جودة ATR/ADX(5%/40)',
               overrides={}, regime=None,
               quality={'atr_min': 0.05, 'adx_min': 40.0},
               entry={'mode': 'breakout', 'lookback': 20, 'mom': 10}),
    'G3': dict(desc='اختراق + قوة نسبية مقابل BTC + جودة ATR/ADX',
               overrides={}, regime=None,
               quality={'atr_min': 0.05, 'adx_min': 40.0},
               entry={'mode': 'breakout_rs', 'lookback': 20, 'mom': 10, 'rs': 30}),
    'G4': dict(desc='اختراق+RS+جودة + فلتر الخوف + SL-2.5% + EarlyExit (كامل)',
               overrides={'STOP_LOSS_PCT': -0.025, 'EARLY_EXIT_ENABLED': True},
               regime={'mode': 'stable', 'threshold': 1.15},
               quality={'atr_min': 0.05, 'adx_min': 40.0},
               entry={'mode': 'breakout_rs', 'lookback': 20, 'mom': 10, 'rs': 30}),
}

# أزواج العملات المستقرة (DAIUSDT غير متوفر على المرآة — يُتخطى تلقائياً)
STABLE_PAIRS = ['USDCUSDT', 'FDUSDUSDT', 'TUSDUSDT', 'DAIUSDT']
STABLE_LOOKBACK = 30


def build_stable_ratio_map(year):
    """
    يبني نسبة الخوف لكل شمعة 4h: vol(آخر شمعة مكتملة) / متوسط آخر 30 (سببي shift(1)).
    صفر نظر مستقبلي. يُرجع (open_times_sorted: np.int64[], ratios: float[]).
    البوابة تستخدم open_time للبحث عن آخر شمعة مكتملة قبل t_ms.
    """
    from datetime import datetime, timezone
    from . import data
    s = datetime(year - 1, 11, 1, tzinfo=timezone.utc)
    e = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
    vol = {}
    for p in STABLE_PAIRS:
        df = data.get_klines_df(p, '4h', s, e)
        if df is None or df.empty:
            continue
        for ot, qv in zip(df['open_time'], df['quote_volume']):
            vol[int(ot)] = vol.get(int(ot), 0.0) + float(qv)
    if len(vol) < STABLE_LOOKBACK + 2:
        return np.array([], dtype=np.int64), np.array([], dtype=float)
    items = sorted(vol.items())
    times = np.array([t for t, _ in items], dtype=np.int64)
    vols = np.array([v for _, v in items], dtype=float)
    avg = pd.Series(vols).rolling(STABLE_LOOKBACK).mean().shift(1).to_numpy()
    ratio = np.where((np.isfinite(avg)) & (avg > 0), vols / avg, np.nan)
    return times, ratio


class RegimeBacktester(Backtester):
    """نفس المحرك + بوابة نظام السوق (BTC trend أو Stable-Ratio Fear) قبل أي scan."""
    regime_mode = None          # 'ema50' | 'ema99' | 'stable'
    stable_threshold = 1.15
    stable_times = None         # np.int64[]  open_times
    stable_ratio = None         # float[]

    def _regime_ok(self, t_ms):
        if self.regime_mode == 'stable':
            if self.stable_times is None or len(self.stable_times) == 0:
                return True
            idx = int(np.searchsorted(self.stable_times, t_ms, side='left')) - 1
            if idx < 0:
                return True  # لا بيانات بعد → اسمح
            r = self.stable_ratio[idx]
            if not np.isfinite(r):
                return True  # فشل/نقص → لا تمنع (مطابق fetch_stable_ratio)
            return bool(r <= self.stable_threshold)  # خوف (r>عتبة) → امنع
        # بوابات BTC
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
    regime = cfg['regime']
    cls = RegimeBacktester if regime else Backtester
    bt = cls(start_capital=capital, year=year, verbose=False)
    if regime:
        if isinstance(regime, dict) and regime.get('mode') == 'stable':
            bt.regime_mode = 'stable'
            bt.stable_threshold = regime.get('threshold', 1.15)
            bt.stable_times, bt.stable_ratio = build_stable_ratio_map(year)
        else:
            bt.regime_mode = regime  # 'ema50' / 'ema99'
    q = cfg.get('quality')
    if q:
        bt.quality_filter = True
        bt.atr_ratio_min = q['atr_min']
        bt.adx_min = q['adx_min']
    ax = cfg.get('atr_exit')
    if ax:
        bt.atr_exit = True
        bt.atr_sl_mult = ax['sl']
        bt.atr_trail_mult = ax['trail']
        bt.atr_dead_h = ax.get('dead_h')
    en = cfg.get('entry')
    if en:
        bt.entry_mode = en['mode']
        bt.breakout_lookback = en.get('lookback', 20)
        bt.breakout_mom = en.get('mom', 10)
        bt.rs_lookback = en.get('rs', 30)
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
    tag = f'{exp_id}_{year}'
    with open(f'backtest/output/exp_{tag}_summary.txt', 'w') as f:
        f.write(out)
    write_trades(bt.state, f'backtest/output/exp_{tag}_trades.csv')
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
