#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
optimizer.py — مُحسّن جيني (Genetic Algorithm) لاكتشاف أفضل تركيبة استراتيجية.

الهدف: أعلى Sharpe (أقل مخاطرة) مع تحقّق صارم خارج العيّنة.
  • التدريب/التطوّر: 2023 + 2024  (دالة اللياقة).
  • الاختبار النهائي: 2025 محجوبة تماماً (لا تُستخدم في الاختيار).
السرعة: تُحمّل بيانات 4h/daily مرة واحدة (نطاق واسع) وتُعاد لكل تقييم.

التشغيل:
  python3 -m backtest.optimizer --pop 14 --gen 6 --cap 140
"""
import argparse
import json
import random
import time
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from . import data
from . import strategy_core as S
from . import engine as E
from .engine import Backtester, Symbol4H, SymbolDaily, PLATFORM


# ═══════════════════════════════════════════════════════════════════
# فضاء البحث (الجينات)
# ═══════════════════════════════════════════════════════════════════
TF_LIST = ['2h', '4h', '6h', '12h', '1d']   # أطر الإشارة (1h مُسقط على كامل العملات لثقله)

SPACE = {
    'signal_tf':         list(TF_LIST),
    'entry_mode':        ['breakout', 'breakout_rs', 'divergence'],
    'breakout_lookback': [10, 15, 20, 30, 40],
    'breakout_mom':      [5, 10, 15, 20],
    'rs_lookback':       [15, 30, 45, 60],
    'rs_margin':         [0.0, 0.02, 0.05, 0.10],
    'quality':           [True, False],
    'atr_ratio_min':     [0.02, 0.03, 0.05, 0.07],
    'adx_min':           [20, 25, 30, 40, 50],
    'sl':                [-0.02, -0.025, -0.03, -0.04, -0.05],
    'tp':                [0.06, 0.10, 0.15, 0.20, 0.25, 0.30],
    'trail_act':         [0.02, 0.03, 0.05],
    'trail_ratio':       [0.97, 0.98, 0.985],
    'early_exit':        [True, False],
    'fear':              [None, 1.10, 1.15, 1.30],
    'max_pos':           [3, 5, 8, 12, 16],
}

TRAIN_YEARS = [2023, 2024]
TEST_YEAR = 2025
MIN_TRADES = 12          # حد أدنى للصفقات/سنة (يتجنّب تركيبات بلا تداول)
CAPITAL = 5000.0


# ═══════════════════════════════════════════════════════════════════
# حزمة البيانات (تُحمّل مرة واحدة، نطاق واسع، تُعاد لكل تقييم)
# ═══════════════════════════════════════════════════════════════════
class Bundle:
    def __init__(self, cap=140, workers=8, log=print):
        self.log = log
        allp = data.list_usdt_pairs()
        uni = []
        for sym in allp:
            base = sym[:-4] if sym.endswith('USDT') else sym
            if base.upper() in S.NEVER_TOUCH_ASSETS or S.is_leveraged_token(base):
                continue
            uni.append(sym)
        # نضمن وجود BTCUSDT (للقوة النسبية) + عيّنة محدودة للسرعة
        sample = uni if (cap is None or cap <= 0) else uni[:cap]
        if 'BTCUSDT' not in sample:
            sample = ['BTCUSDT'] + sample[:cap - 1]
        self.req_universe = sample
        self.s4_start = datetime(2022, 10, 1, tzinfo=timezone.utc)
        self.d_start = datetime(2022, 4, 1, tzinfo=timezone.utc)
        self.end = datetime(2026, 1, 1, tzinfo=timezone.utc)
        self.tf_data = {tf: {} for tf in TF_LIST}   # tf -> {sym: Symbol4H}
        self.daily = {}
        self.k15 = {}
        self._k15_loaded = {}
        self._daily_missing = set()
        self._load(workers)
        self.stable_times, self.stable_ratio = self._build_stable()

    def _load(self, workers):
        self.log(f"تحميل {len(TF_LIST)} أطر لـ {len(self.req_universe)} عملة (مرة واحدة)...")

        def loadtf(sym, tf):
            return sym, tf, data.get_klines_df(sym, tf, self.s4_start, self.end)

        def loadd(sym):
            return sym, data.get_klines_df(sym, '1d', self.d_start, self.end)

        # نحمّل كل (عملة، إطار) — البيانات اليومية للفلاتر تبقى منفصلة
        for tf in TF_LIST:
            done = 0
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(loadtf, s, tf): s for s in self.req_universe}
                for fut in as_completed(futs):
                    sym, _tf, df = fut.result()
                    if df is not None and len(df) >= 50:
                        self.tf_data[_tf][sym] = Symbol4H(sym, df)
                    done += 1
            self.log(f"  {tf}: {len(self.tf_data[tf])} عملة")
        # العملات التي لها 4h على الأقل = الكون المرجعي
        self.universe = [s for s in self.req_universe if s in self.tf_data['4h']]
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for fut in as_completed({ex.submit(loadd, s): s for s in self.universe}):
                sym, df = fut.result()
                if df is not None and len(df) >= 60:
                    self.daily[sym] = SymbolDaily(sym, df)
                else:
                    self._daily_missing.add(sym)
        self.log(f"جاهز: daily={len(self.daily)} | universe={len(self.universe)} عملة.")

    def _build_stable(self):
        pairs = ['USDCUSDT', 'FDUSDUSDT', 'TUSDUSDT', 'DAIUSDT']
        vol = {}
        for p in pairs:
            df = data.get_klines_df(p, '4h', self.s4_start, self.end)
            if df is None or df.empty:
                continue
            for ot, qv in zip(df['open_time'], df['quote_volume']):
                vol[int(ot)] = vol.get(int(ot), 0.0) + float(qv)
        if len(vol) < 32:
            return np.array([], dtype=np.int64), np.array([], dtype=float)
        items = sorted(vol.items())
        times = np.array([t for t, _ in items], dtype=np.int64)
        vols = np.array([v for _, v in items], dtype=float)
        avg = pd.Series(vols).rolling(30).mean().shift(1).to_numpy()
        ratio = np.where((np.isfinite(avg)) & (avg > 0), vols / avg, np.nan)
        return times, ratio


# ═══════════════════════════════════════════════════════════════════
# تطبيق الجينوم + التقييم
# ═══════════════════════════════════════════════════════════════════
def apply_genome(bt, g):
    # ثوابت S (تُضبط في كل تقييم لتفادي حالة عالقة)
    S.STOP_LOSS_PCT = g['sl']
    S.TP_THRESHOLD = g['tp']
    S.TRAILING_ACTIVATE_PCT = g['trail_act']
    S.TRAILING_RATIO = g['trail_ratio']
    S.EARLY_EXIT_ENABLED = g['early_exit']
    S.MAX_OPEN_POSITIONS = g['max_pos']
    # سمات المحرك
    bt.entry_mode = g['entry_mode']
    bt.breakout_lookback = g['breakout_lookback']
    bt.breakout_mom = g['breakout_mom']
    bt.rs_lookback = g['rs_lookback']
    bt.rs_margin = g['rs_margin']
    bt.quality_filter = g['quality']
    bt.atr_ratio_min = g['atr_ratio_min']
    bt.adx_min = g['adx_min']
    bt.fear_block_threshold = g['fear']


def _sharpe(equity_log, capital):
    if len(equity_log) < 5:
        return 0.0
    eq = np.array([e for _, e in equity_log], dtype=float)
    eq = np.concatenate([[capital], eq])
    rets = np.diff(eq) / eq[:-1]
    if rets.std() < 1e-9:
        return 0.0
    return float(rets.mean() / rets.std() * np.sqrt(365))


def _maxdd(equity_log, capital):
    if not equity_log:
        return 0.0
    eq = np.concatenate([[capital], [e for _, e in equity_log]])
    peak = np.maximum.accumulate(eq)
    return float(((eq - peak) / peak).min() * 100)


def evaluate_year(bundle, year, g):
    """يقيّم جينوم على سنة. يعيد dict(metrics) مع reuse للبيانات."""
    tf = g.get('signal_tf', '4h')
    tf_dict = bundle.tf_data.get(tf, bundle.tf_data['4h'])
    bt = Backtester(start_capital=CAPITAL, year=year,
                    universe=list(bundle.universe), verbose=False)
    # حقن البيانات (تخطّي التحميل) — بيانات إطار الإشارة المختار
    bt.data4h = tf_dict
    bt.signal_tf = tf
    bt.ms_sig, bt.bars_24h = E.tf_params(tf)
    bt.daily = bundle.daily
    bt._daily_missing = bundle._daily_missing
    bt.k15 = bundle.k15
    bt._k15_loaded = bundle._k15_loaded
    # الكون = عملات لها بيانات في هذا الإطار (مع ضمان BTC للقوة النسبية)
    uni = [s for s in bundle.universe if s in tf_dict]
    if 'BTCUSDT' in tf_dict and 'BTCUSDT' not in uni:
        uni.append('BTCUSDT')
    bt.universe = uni
    bt.stable_times = bundle.stable_times
    bt.stable_ratio = bundle.stable_ratio
    apply_genome(bt, g)
    bt.run()
    h = bt.state[PLATFORM]['history']
    pnls = np.array([t.get('net_pnl', 0.0) for t in h], dtype=float)
    gp = float(pnls[pnls > 0].sum()); gl = float(-pnls[pnls <= 0].sum())
    final_eq = bt.state[PLATFORM]['liquid_capital']
    return {
        'sharpe': _sharpe(bt.equity_log, CAPITAL),
        'return_pct': (final_eq - CAPITAL) / CAPITAL * 100,
        'pf': (gp / gl) if gl > 0 else (9.99 if gp > 0 else 0.0),
        'maxdd': _maxdd(bt.equity_log, CAPITAL),
        'trades': len(h),
    }


OBJECTIVE = 'sharpe'   # 'sharpe' (متّسق منخفض المخاطرة) أو 'return' (تعظيم العائد/Calmar)


def to_daily(annual_pct):
    """عائد سنوي% → متوسط يومي% مركّب (365 يوماً)."""
    return ((1.0 + annual_pct / 100.0) ** (1.0 / 365.0) - 1.0) * 100.0


def to_monthly(annual_pct):
    return ((1.0 + annual_pct / 100.0) ** (1.0 / 12.0) - 1.0) * 100.0


def fitness(bundle, g, cache):
    key = json.dumps(g, sort_keys=True)
    if key in cache:
        return cache[key]
    yr = {}
    for y in TRAIN_YEARS:
        yr[y] = evaluate_year(bundle, y, g)
    sh = [yr[y]['sharpe'] for y in TRAIN_YEARS]
    rt = [yr[y]['return_pct'] for y in TRAIN_YEARS]
    tr = [yr[y]['trades'] for y in TRAIN_YEARS]
    dd = min(yr[y]['maxdd'] for y in TRAIN_YEARS)  # الأسوأ (أكثر سلبية)
    if min(tr) < MIN_TRADES:
        fit = -50.0 + min(tr) * 0.01           # يوجّه GA بعيداً عن "لا تداول"
    elif OBJECTIVE == 'return':
        # تعظيم متوسط العائد الشهري مع حارس تراجع (Calmar-style)
        m_avg = float(np.mean([to_monthly(r) for r in rt]))
        m_min = float(np.min([to_monthly(r) for r in rt]))
        fit = 0.7 * m_avg + 0.3 * m_min          # كافئ العائد، اطلب اتّساقاً جزئياً
        fit -= max(0.0, (-dd - 40.0)) * 0.5      # عقوبة تراجع فقط بعد -40% (عدواني)
    else:
        fit = float(np.mean(sh)) - 0.5 * float(np.std(sh))   # Sharpe متّسق
        fit -= max(0.0, (-dd - 20.0)) * 0.05                  # عقوبة تراجع قوية >20%
    res = {'fit': fit, 'train': yr}
    cache[key] = res
    return res


# ═══════════════════════════════════════════════════════════════════
# عمليات الخوارزمية الجينية
# ═══════════════════════════════════════════════════════════════════
def rand_genome():
    return {k: random.choice(v) for k, v in SPACE.items()}


def crossover(a, b):
    return {k: (a[k] if random.random() < 0.5 else b[k]) for k in SPACE}


def mutate(g, rate=0.25):
    h = dict(g)
    for k in SPACE:
        if random.random() < rate:
            h[k] = random.choice(SPACE[k])
    return h


def tournament(pop_scored, k=3):
    return max(random.sample(pop_scored, k), key=lambda x: x[1]['fit'])[0]


def _checkpoint(best_overall, gen, gens):
    g, r = best_overall
    out = {'genome': g, 'generation': f'{gen}/{gens}',
           'train': {str(y): r['train'][y] for y in TRAIN_YEARS},
           'fit': r['fit'], 'ts': datetime.now(timezone.utc).isoformat()}
    try:
        with open('backtest/output/ga_checkpoint.json', 'w', encoding='utf-8') as f:
            json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    except Exception:
        pass


def _target_met(r, target_ret, target_dd):
    """الهدف: في وضع العائد = متوسط شهري ≥ target_ret%؛ غير ذلك = كل سنة ≥ target_ret%."""
    rets = [r['train'][y]['return_pct'] for y in TRAIN_YEARS]
    dds = [r['train'][y]['maxdd'] for y in TRAIN_YEARS]
    if min(dds) < -target_dd:
        return False
    if OBJECTIVE == 'return':
        return float(np.mean([to_monthly(x) for x in rets])) >= target_ret
    return min(rets) >= target_ret


def run_ga(bundle, pop_size=14, gens=30, elite=3, patience=8,
           target_ret=35.0, target_dd=30.0, log=print):
    cache = {}
    pop = [rand_genome() for _ in range(pop_size)]
    best_overall = None
    no_improve = 0
    for gen in range(gens):
        scored = []
        for g in pop:
            r = fitness(bundle, g, cache)
            scored.append((g, r))
        scored.sort(key=lambda x: x[1]['fit'], reverse=True)
        best_g, best_r = scored[0]
        improved = best_overall is None or best_r['fit'] > best_overall[1]['fit'] + 1e-6
        if improved:
            best_overall = (best_g, best_r)
            no_improve = 0
        else:
            no_improve += 1
        sh = [best_r['train'][y]['sharpe'] for y in TRAIN_YEARS]
        rt = [best_r['train'][y]['return_pct'] for y in TRAIN_YEARS]
        mo = float(np.mean([to_monthly(x) for x in rt]))
        da = float(np.mean([to_daily(x) for x in rt]))
        log(f"[جيل {gen+1}/{gens}] لياقة={best_r['fit']:.3f} (ثبات {no_improve}) | "
            f"عائد=({rt[0]:+.0f}%,{rt[1]:+.0f}%) ≈{mo:+.1f}%/شهر ≈{da:+.2f}%/يوم | "
            f"Sharpe={tuple(round(x,2) for x in sh)} | tf={best_g['signal_tf']} | {_short(best_g)}", flush=True)
        _checkpoint(best_overall, gen + 1, gens)
        # شرط الإيقاف: بلوغ الهدف أو ثبات طويل
        if _target_met(best_overall[1], target_ret, target_dd):
            log(f"🎯 بلوغ الهدف عند الجيل {gen+1}! (كل سنوات التدريب ≥{target_ret}%)", flush=True)
            break
        if no_improve >= patience:
            log(f"⏹️ توقّف: لا تحسّن منذ {patience} أجيال (تقارب).", flush=True)
            break
        # الجيل التالي: نخبة + نسل + دم جديد (تنويع)
        nxt = [g for g, _ in scored[:elite]]
        ps = scored
        while len(nxt) < pop_size - 2:
            child = mutate(crossover(tournament(ps), tournament(ps)))
            nxt.append(child)
        nxt += [rand_genome(), rand_genome()]   # حقن عشوائي لتجنّب التقارب المبكر
        pop = nxt
    return best_overall, cache


def _short(g):
    return (f"{g['entry_mode']},bo{g['breakout_lookback']}/{g['breakout_mom']},"
            f"rs{g['rs_lookback']}+{g['rs_margin']},q{int(g['quality'])}"
            f"{g['atr_ratio_min']}/{g['adx_min']},sl{g['sl']},tp{g['tp']},"
            f"tr{g['trail_act']}/{g['trail_ratio']},ee{int(g['early_exit'])},"
            f"fear{g['fear']},mp{g['max_pos']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pop', type=int, default=16)
    ap.add_argument('--gen', type=int, default=40)
    ap.add_argument('--cap', type=int, default=140)
    ap.add_argument('--patience', type=int, default=10)
    ap.add_argument('--target-ret', type=float, default=35.0)
    ap.add_argument('--target-dd', type=float, default=30.0)
    ap.add_argument('--objective', choices=['sharpe', 'return'], default='sharpe')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    global OBJECTIVE
    OBJECTIVE = args.objective
    print(f"الهدف: {OBJECTIVE} | target={args.target_ret} | target_dd={args.target_dd}", flush=True)
    random.seed(args.seed); np.random.seed(args.seed)
    t0 = time.time()

    bundle = Bundle(cap=args.cap, workers=8)
    (best_g, best_r), cache = run_ga(bundle, args.pop, args.gen,
                                     patience=args.patience,
                                     target_ret=args.target_ret,
                                     target_dd=args.target_dd)

    print("\n" + "=" * 60)
    print("أفضل جينوم (مُحسّن على 2023+2024):")
    print(json.dumps(best_g, ensure_ascii=False))
    print("-" * 60)
    # أداء التدريب + الاختبار المحجوب (2025)
    rows = []
    for y in TRAIN_YEARS:
        m = best_r['train'][y]
        rows.append((y, 'train', m))
    test_m = evaluate_year(bundle, TEST_YEAR, best_g)
    rows.append((TEST_YEAR, 'TEST', test_m))
    for y, tag, m in rows:
        print(f"  {y} [{tag:5s}] عائد={m['return_pct']:+.1f}% ≈{to_monthly(m['return_pct']):+.1f}%/شهر "
              f"≈{to_daily(m['return_pct']):+.2f}%/يوم | Sharpe={m['sharpe']:+.2f} | "
              f"PF={m['pf']:.2f} | DD={m['maxdd']:.1f}% | صفقات={m['trades']}")
    print("=" * 60)
    out = {'genome': best_g,
           'train': {str(y): best_r['train'][y] for y in TRAIN_YEARS},
           'test_2025': test_m, 'evals': len(cache),
           'elapsed_min': (time.time() - t0) / 60,
           'note': 'عيّنة محدودة للبحث — يجب إعادة التحقّق على كامل العملات'}
    with open('backtest/output/ga_best.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    print(f"⏱️ {out['elapsed_min']:.1f}د | تقييمات={len(cache)} | حُفظ backtest/output/ga_best.json")


if __name__ == '__main__':
    main()
