"""
محرك تداول لحظي (intraday) على 15m لعملات سائلة.
الهدف: عشرات الصفقات الصغيرة يومياً تتراكم — مسار بلا رافعة نحو عائد يومي أعلى.
الرسوم واقعية (0.075%/جهة) — العدو الأكبر للتردد العالي، فنقيس الصافي بأمانة.

يعيد استخدام data.get_klines_df + Symbol4H (مؤشرات RSI/MACD/ATR/ADX على 15m).
"""
import sys
import time
import json
import argparse
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from backtest import data
from backtest.engine import Symbol4H
import backtest.strategy_core as S

MS_15M = 15 * 60 * 1000
TAKER_FEE = S.EFFECTIVE_FEE_BINANCE    # 0.075%/جهة (أمر سوق)
MAKER_FEE = 0.0002                     # 0.02%/جهة (أمر Limit) — السكالبرز الحقيقيون
BARS_PER_DAY = 96                      # 15m
TRAIN_YEARS = (2023, 2024)
TEST_YEAR = 2025


def log(m):
    print(m, flush=True)


# ════════════════════════════════════════════════════════════════════
#  بيانات
# ════════════════════════════════════════════════════════════════════
class IntradayData:
    """يحمّل 15m لكامل النطاق ويرتّب الكون حسب السيولة (أعلى N)."""

    def __init__(self, top_n=60, workers=8, req_universe=None):
        self.top_n = top_n
        self.start = datetime(2022, 12, 1, tzinfo=timezone.utc)   # هامش إحماء للمؤشرات
        self.end = datetime(2026, 1, 1, tzinfo=timezone.utc)
        self.bars = {}        # sym -> Symbol4H (على 15m)
        uni = req_universe or data.get_universe()
        log(f"intraday: تحميل 15m لـ {len(uni)} عملة لاختيار أعلى {top_n} سيولة...")

        def load(sym):
            df = data.get_klines_df(sym, '15m', self.start, self.end)
            return sym, df

        loaded = {}
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for fut in as_completed({ex.submit(load, s): s for s in uni}):
                sym, df = fut.result()
                if df is not None and len(df) >= 2000:
                    loaded[sym] = df
        # ترتيب حسب متوسط حجم التداول بالدولار (سيولة) — scalping يحتاج سيولة
        ranked = sorted(loaded.items(),
                        key=lambda kv: float(np.median(kv[1]['quote_volume'].to_numpy())),
                        reverse=True)
        top = ranked[:top_n]
        self.csum = {}      # مجموع تراكمي للإغلاق (لـ SMA الاتجاه بسرعة O(1))
        for sym, df in top:
            b = Symbol4H(sym, df)
            self.bars[sym] = b
            cs = np.zeros(b.n + 1, dtype=np.float64)
            np.cumsum(b.close, out=cs[1:])
            self.csum[sym] = cs
        self.universe = [s for s, _ in top]
        log(f"intraday: جاهز — {len(self.universe)} عملة سائلة على 15m.")


# ════════════════════════════════════════════════════════════════════
#  الباكتست اللحظي
# ════════════════════════════════════════════════════════════════════
DEFAULT = {
    'entry':       'breakout',   # breakout | revert
    'lookback':    20,           # شموع المرجع للاختراق/التطرّف
    'rsi_min':     55,           # اختراق: RSI ≥ (زخم)
    'rsi_os':      30,           # ارتداد: RSI ≤ (تشبّع بيعي)
    'vol_mult':    1.5,          # حجم الشمعة ≥ mult × متوسط (تأكيد)
    'sl':          -0.010,       # وقف
    'tp':          0.015,        # هدف
    'trail':       0.008,        # trailing من القمة (0=معطّل)
    'max_hold':    16,           # أقصى احتفاظ (شموع 15m) = 4h
    'max_pos':     5,            # مراكز متزامنة
    'trend_sma':   200,          # فلتر اتجاه: ادخل فقط فوق SMA(هذا) — 0=معطّل (~2 يوم)
}


def _year_bounds(year):
    s = int(datetime(year, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    e = int(datetime(year + 1, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    return s, e


def run_intraday(idata, year, p, start_capital=5000.0):
    """يشغّل سنة كاملة على 15m. يعيد مقاييس الأداء (صافي الرسوم)."""
    s_ms, e_ms = _year_bounds(year)
    syms = idata.universe
    bars = idata.bars
    # فهارس البداية لكل عملة (أول شمعة داخل السنة)
    L = int(p['lookback'])
    entry = p['entry']
    rsi_min = p['rsi_min']; rsi_os = p['rsi_os']
    vol_mult = p['vol_mult']
    sl = p['sl']; tp = p['tp']; trail = p['trail']
    max_hold = int(p['max_hold']); max_pos = int(p['max_pos'])
    tsma = int(p.get('trend_sma', 0))
    csum = idata.csum

    # بناء شبكة زمنية موحّدة على خطوات 15m
    equity = start_capital
    liquid = start_capital
    positions = {}        # sym -> dict(entry, qty, peak, opened_bar)
    trades = []           # (pnl_pct,)
    eq_curve = []         # equity لكل خطوة (لـ Sharpe/DD)
    n_steps = 0

    # نطاق الزمن
    t = s_ms
    # خرائط فهرسة سريعة: لكل عملة مصفوفة close_time
    ct = {s: bars[s].close_time for s in syms}
    # مؤشر تتبّع موضع كل عملة (pointer) لتفادي البحث الثنائي المتكرر
    ptr = {}
    for s in syms:
        ptr[s] = int(np.searchsorted(ct[s], s_ms, side='left'))

    daily_eq = []         # equity في نهاية كل يوم (لعائد يومي)
    last_day = None

    while t < e_ms:
        # idx لكل عملة: آخر شمعة مغلقة عند الزمن t
        # نتقدّم pointer حتى close_time > t
        # نقيّم على الشمعة المغلقة i حيث close_time[i] == t تقريباً
        for s in syms:
            arr = ct[s]
            i = ptr[s]
            while i < len(arr) and arr[i] <= t:
                i += 1
            ptr[s] = i
        # i-1 = آخر شمعة مغلقة
        # ── إدارة المراكز المفتوحة (خروج) ──
        for s in list(positions.keys()):
            b = bars[s]; i = ptr[s] - 1
            if i < 0 or i >= b.n:
                continue
            pos = positions[s]
            hi = b.high[i]; lo = b.low[i]; cl = b.close[i]
            ep = pos['entry']
            exit_px = None; exit_fee = TAKER_FEE
            # SL أولاً (تحفّظ) — أمر سوق = taker
            if lo <= ep * (1 + sl):
                exit_px = ep * (1 + sl); exit_fee = TAKER_FEE
            elif hi >= ep * (1 + tp):
                exit_px = ep * (1 + tp); exit_fee = MAKER_FEE   # هدف Limit = maker
            else:
                if cl > pos['peak']:
                    pos['peak'] = cl
                if trail > 0 and cl <= pos['peak'] * (1 - trail):
                    exit_px = cl; exit_fee = TAKER_FEE
                elif (i - pos['opened_bar']) >= max_hold:
                    exit_px = cl; exit_fee = TAKER_FEE
            if exit_px is not None:
                proceeds = exit_px * pos['qty'] * (1 - exit_fee)
                liquid += proceeds
                pnl = (proceeds - pos['cost']) / pos['cost']
                trades.append(pnl)
                del positions[s]

        # ── دخول جديد ──
        slots = max_pos - len(positions)
        if slots > 0:
            cands = []
            for s in syms:
                if s in positions:
                    continue
                b = bars[s]; i = ptr[s] - 1
                if i < L + 2 or i >= b.n:
                    continue
                cl = b.close[i]
                # فلتر الاتجاه: السعر فوق SMA(tsma) — تداول مع الاتجاه الصاعد فقط
                if tsma > 0:
                    if i < tsma:
                        continue
                    cs = csum[s]
                    sma = (cs[i + 1] - cs[i + 1 - tsma]) / tsma
                    if cl <= sma:
                        continue
                if entry == 'breakout':
                    ref = b.high[i - L:i].max()
                    vol_ok = b.volume[i] >= vol_mult * b.volume[i - L:i].mean()
                    if cl > ref and b.rsi[i] >= rsi_min and vol_ok:
                        cands.append((s, b.rsi[i], i, cl))
                else:  # revert (شراء الانخفاض داخل اتجاه صاعد)
                    if b.rsi[i] <= rsi_os and b.close[i] > b.open[i]:
                        cands.append((s, -b.rsi[i], i, cl))
            # رتّب حسب القوة (RSI للزخم، أو الأكثر تشبّعاً للارتداد)
            cands.sort(key=lambda x: x[1], reverse=True)
            for s, _score, i, cl in cands[:slots]:
                size = equity / max_pos
                if size > liquid:
                    size = liquid
                if size < 10:
                    continue
                qty = size / cl
                cost = cl * qty * (1 + MAKER_FEE)   # دخول Limit = maker
                if cost > liquid:
                    continue
                liquid -= cost
                positions[s] = {'entry': cl, 'qty': qty, 'peak': cl,
                                'opened_bar': i, 'cost': cost}

        # ── تحديث الإكويتي (mark-to-market) ──
        mtm = liquid
        for s, pos in positions.items():
            b = bars[s]; i = ptr[s] - 1
            px = b.close[i] if 0 <= i < b.n else pos['entry']
            mtm += px * pos["qty"] * (1 - TAKER_FEE)
        equity = mtm
        eq_curve.append(equity)
        day = t // (BARS_PER_DAY * MS_15M)
        if last_day is None:
            last_day = day
        elif day != last_day:
            daily_eq.append(equity)
            last_day = day
        n_steps += 1
        t += MS_15M

    # تصفية ما تبقّى بسعر الإغلاق الأخير
    for s, pos in positions.items():
        b = bars[s]; i = min(ptr[s] - 1, b.n - 1)
        proceeds = b.close[i] * pos["qty"] * (1 - TAKER_FEE)
        liquid += proceeds
        trades.append((proceeds - pos['cost']) / pos['cost'])
    equity = liquid

    return _metrics(start_capital, equity, eq_curve, daily_eq, trades)


def _metrics(start_cap, end_eq, eq_curve, daily_eq, trades):
    ret_pct = (end_eq / start_cap - 1) * 100
    arr = np.array(eq_curve, dtype=np.float64)
    if len(arr) > 1:
        peak = np.maximum.accumulate(arr)
        dd = ((arr - peak) / peak).min() * 100
    else:
        dd = 0.0
    # Sharpe على العوائد اليومية
    de = np.array(daily_eq, dtype=np.float64)
    if len(de) > 2:
        dr = np.diff(de) / de[:-1]
        sharpe = float(np.mean(dr) / (np.std(dr) + 1e-9) * np.sqrt(365))
        avg_daily = float(np.mean(dr) * 100)
    else:
        sharpe = 0.0; avg_daily = 0.0
    wins = [x for x in trades if x > 0]
    wr = len(wins) / len(trades) * 100 if trades else 0.0
    gp = sum(x for x in trades if x > 0)
    gl = -sum(x for x in trades if x < 0)
    pf = gp / gl if gl > 0 else 0.0
    ndays = max(1, len(daily_eq) + 1)
    return {
        'return_pct': ret_pct,
        'avg_daily_pct': avg_daily,
        'monthly_pct': ((1 + ret_pct / 100) ** (1 / 12) - 1) * 100 if ret_pct > -100 else -100,
        'daily_comp_pct': ((1 + ret_pct / 100) ** (1 / 365) - 1) * 100 if ret_pct > -100 else -100,
        'sharpe': sharpe,
        'maxdd': dd,
        'trades': len(trades),
        'trades_per_day': len(trades) / ndays,
        'win_rate': wr,
        'pf': pf,
    }


def _fmt(year, tag, m):
    return (f"  {year} [{tag:5s}] عائد={m['return_pct']:+.1f}% "
            f"≈{m['monthly_pct']:+.1f}%/شهر ≈{m['daily_comp_pct']:+.2f}%/يوم | "
            f"Sharpe={m['sharpe']:+.2f} | DD={m['maxdd']:.1f}% | "
            f"صفقات={m['trades']} ({m['trades_per_day']:.1f}/يوم) | WR={m['win_rate']:.0f}% | PF={m['pf']:.2f}")


# ════════════════════════════════════════════════════════════════════
#  GA لحظي — يبحث في آلاف التركيبات (لا حدس)
# ════════════════════════════════════════════════════════════════════
import random as _rnd

ISPACE = {
    'entry':     ['breakout', 'revert'],
    'lookback':  [10, 20, 30, 40, 60],
    'rsi_min':   [50, 55, 60, 65, 70],
    'rsi_os':    [20, 25, 30, 35],
    'vol_mult':  [1.0, 1.5, 2.0, 3.0],
    'sl':        [-0.005, -0.008, -0.012, -0.02, -0.03],
    'tp':        [0.008, 0.012, 0.02, 0.03, 0.05],
    'trail':     [0.0, 0.005, 0.008, 0.015, 0.025],
    'max_hold':  [8, 16, 32, 48, 96],
    'max_pos':   [3, 5, 8],
    'trend_sma': [0, 100, 200, 400],
}


def _rand():
    return {k: _rnd.choice(v) for k, v in ISPACE.items()}


def _cross(a, b):
    return {k: (a[k] if _rnd.random() < 0.5 else b[k]) for k in ISPACE}


def _mut(g, rate=0.25):
    g = dict(g)
    for k in ISPACE:
        if _rnd.random() < rate:
            g[k] = _rnd.choice(ISPACE[k])
    return g


def _ifit(idata, g, cache):
    key = json.dumps(g, sort_keys=True)
    if key in cache:
        return cache[key]
    yr = {y: run_intraday(idata, y, g) for y in TRAIN_YEARS}
    mo = [yr[y]['monthly_pct'] for y in TRAIN_YEARS]
    dd = min(yr[y]['maxdd'] for y in TRAIN_YEARS)
    tr = min(yr[y]['trades'] for y in TRAIN_YEARS)
    if tr < 20:                              # يحتاج نشاطاً معقولاً
        fit = -100 + tr
    else:
        fit = 0.7 * float(np.mean(mo)) + 0.3 * float(np.min(mo))
        fit -= max(0.0, (-dd - 35.0)) * 0.5  # حارس تراجع
    res = {'fit': fit, 'train': yr}
    cache[key] = res
    return res


def run_iga(idata, pop=16, gens=25, elite=3, patience=8, seed=7):
    _rnd.seed(seed); np.random.seed(seed)
    cache = {}
    popu = [_rand() for _ in range(pop)]
    best = None; noimp = 0
    for gen in range(gens):
        scored = sorted(((g, _ifit(idata, g, cache)) for g in popu),
                        key=lambda x: x[1]['fit'], reverse=True)
        bg, br = scored[0]
        if best is None or br['fit'] > best[1]['fit'] + 1e-9:
            best = (bg, br); noimp = 0
        else:
            noimp += 1
        mo = [br['train'][y]['monthly_pct'] for y in TRAIN_YEARS]
        da = [br['train'][y]['daily_comp_pct'] for y in TRAIN_YEARS]
        tpd = np.mean([br['train'][y]['trades_per_day'] for y in TRAIN_YEARS])
        log(f"[جيل {gen+1}/{gens}] لياقة={br['fit']:.2f} (ثبات {noimp}) | "
            f"≈{np.mean(mo):+.1f}%/شهر ≈{np.mean(da):+.2f}%/يوم | "
            f"{tpd:.1f}صفقة/يوم | {bg['entry']},sma{bg['trend_sma']},tp{bg['tp']},sl{bg['sl']},mp{bg['max_pos']}")
        try:
            with open('backtest/output/iga_best.json', 'w', encoding='utf-8') as f:
                json.dump({'genome': best[0], 'train': best[1]['train']}, f,
                          ensure_ascii=False, indent=2, default=str)
        except Exception:
            pass
        if noimp >= patience:
            log(f"⏹️ توقّف: ثبات {patience} أجيال."); break
        nxt = [g for g, _ in scored[:elite]]
        while len(nxt) < pop - 2:
            a = min(_rnd.sample(scored, 3), key=lambda x: -x[1]['fit'])[0]
            b = min(_rnd.sample(scored, 3), key=lambda x: -x[1]['fit'])[0]
            nxt.append(_mut(_cross(a, b)))
        nxt += [_rand(), _rand()]
        popu = nxt
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--top', type=int, default=60)
    ap.add_argument('--year', type=int, default=0, help='0 = كل السنوات')
    ap.add_argument('--ga', action='store_true')
    ap.add_argument('--pop', type=int, default=16)
    ap.add_argument('--gen', type=int, default=25)
    args = ap.parse_args()
    t0 = time.time()
    idata = IntradayData(top_n=args.top)
    if args.ga:
        log(f"\nGA لحظي: maker={MAKER_FEE*100:.3f}% taker={TAKER_FEE*100:.3f}% | top={args.top}")
        log("-" * 70)
        best_g, best_r = run_iga(idata, pop=args.pop, gens=args.gen)
        log("\n" + "=" * 70)
        log("أفضل تركيبة لحظية: " + json.dumps(best_g, ensure_ascii=False))
        for y in [2023, 2024]:
            log(_fmt(y, 'train', best_r['train'][y]))
        tm = run_intraday(idata, TEST_YEAR, best_g)
        log(_fmt(TEST_YEAR, 'TEST', tm))
        log("=" * 70)
        log(f"⏱️ {(time.time()-t0)/60:.1f}د")
        return
    p = dict(DEFAULT)
    log("\nتركيبة افتراضية: " + json.dumps(p, ensure_ascii=False))
    log("-" * 70)
    years = [args.year] if args.year else [2023, 2024, 2025]
    for y in years:
        tag = 'TEST' if y == TEST_YEAR else 'train'
        m = run_intraday(idata, y, p)
        log(_fmt(y, tag, m))
    log("-" * 70)
    log(f"⏱️ {(time.time()-t0)/60:.1f}د")


if __name__ == '__main__':
    main()
