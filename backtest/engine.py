#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
engine.py — محرك المحاكاة التاريخية لاستراتيجية Real Bot v3.27 (Binance).

يعيد تنفيذ منطق search_signals + إدارة الصفقات بدقة:
  - مسح الإشارات على شموع 4h المغلقة كل 4 ساعات.
  - إدارة الخروج على شموع 15m (SL/TP/Trailing) عبر check_exit.
  - فلاتر: الحجم، v3 divergence، OBV، A+D اليومي، F3، EMA99، نسبة الحجم، sigma.
  - التحجيم بقانون الجذر التربيعي + cooldown هرمي.

التحسين: نحسب MACD/RSI/OBV على كامل سلسلة 4h لكل عملة مرة واحدة (ewm متقارب
→ مكافئ لنافذة 200 ضمن دقة الفاصلة العائمة) ثم نأخذ القيم عند الـ pivots.
يُتحقَّق من التكافؤ مقابل calc_signals الأصلية في validate.py.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

from . import data
from . import strategy_core as S

MS_4H = 4 * 3600 * 1000
MS_15M = 15 * 60 * 1000
MS_DAY = 24 * 3600 * 1000

PLATFORM = 'binance'
FEE = S.EFFECTIVE_FEE_BINANCE

# نطاقات التحميل
WARMUP_4H_START = datetime(2024, 11, 1, tzinfo=timezone.utc)
WARMUP_1D_START = datetime(2024, 5, 1, tzinfo=timezone.utc)


# ═══════════════════════════════════════════════════════════════════
# تحميل + precompute بيانات العملة
# ═══════════════════════════════════════════════════════════════════

def _atr_ratio_series(high, low, close, period=14):
    """ATR(Wilder)/price — تذبذب نسبي. سببي (يستخدم الماضي فقط)."""
    h = np.asarray(high, float); l = np.asarray(low, float); c = np.asarray(close, float)
    prev_close = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum.reduce([h - l, np.abs(h - prev_close), np.abs(l - prev_close)])
    atr = pd.Series(tr).ewm(alpha=1 / period, adjust=False).mean().to_numpy()
    return atr / np.where(c > 0, c, np.nan)


def _adx_series(high, low, close, period=14):
    """ADX(Wilder) — قوة الاتجاه 0..100. سببي."""
    h = np.asarray(high, float); l = np.asarray(low, float); c = np.asarray(close, float)
    n = len(c)
    if n < period * 2:
        return np.full(n, np.nan)
    up = np.zeros(n); dn = np.zeros(n)
    up[1:] = h[1:] - h[:-1]; dn[1:] = l[:-1] - l[1:]
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    prev_close = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum.reduce([h - l, np.abs(h - prev_close), np.abs(l - prev_close)])
    atr = pd.Series(tr).ewm(alpha=1 / period, adjust=False).mean().to_numpy()
    pdi = 100 * pd.Series(plus_dm).ewm(alpha=1 / period, adjust=False).mean().to_numpy() / np.where(atr > 1e-12, atr, np.nan)
    mdi = 100 * pd.Series(minus_dm).ewm(alpha=1 / period, adjust=False).mean().to_numpy() / np.where(atr > 1e-12, atr, np.nan)
    dx = 100 * np.abs(pdi - mdi) / np.where((pdi + mdi) > 1e-12, pdi + mdi, np.nan)
    return pd.Series(np.nan_to_num(dx)).ewm(alpha=1 / period, adjust=False).mean().to_numpy()


def check_exit_atr(pos, current_price, now_ts, atr_price, sl_mult, trail_mult,
                   candle=None, dead_h=None, max_hold_days=30):
    """
    خروج ديناميكي مرتبط بالـ ATR (Chandelier) — لالتقاط الانفجارات الكبيرة.
      • الوقف الابتدائي = entry - sl_mult * ATR
      • بعد ارتفاع السعر: الوقف = peak - trail_mult * ATR (يتبع القمة)
      • لا خروج زمني عند 24h (يقتل المنفجرات) — فقط max_hold + dead-cut اختياري.
    atr_price: ATR بوحدة السعر وقت الدخول (atr_ratio*entry). يستخدم low/high الشمعة.
    """
    now = now_ts
    entry = pos['entry_price']
    check_high = candle['high'] if (candle and 'high' in candle) else current_price
    check_low = candle['low'] if (candle and 'low' in candle) else current_price
    if check_high > pos['peak']:
        pos['peak'] = check_high
    # الوقف: الأعلى بين الابتدائي والمتحرك من القمة
    init_stop = entry - sl_mult * atr_price
    trail_stop = pos['peak'] - trail_mult * atr_price
    eff_stop = max(init_stop, trail_stop)
    if check_low <= eff_stop:
        reason = 'ATR_TRAIL' if pos['peak'] > entry + 1e-12 and trail_stop >= init_stop else 'ATR_STOP'
        # سعر الخروج = الوقف (أو low لو فجوة تحته)
        exit_price = eff_stop if check_low <= eff_stop else check_low
        return {'exit_reason': reason, 'exit_price': min(eff_stop, check_high if check_high > 0 else eff_stop),
                'exit_time': now}
    dur_h = (now - pos['entry_time']) / 3600
    # dead-cut اختياري: لم يرتفع فوق entry+0.5*ATR خلال dead_h ساعة
    if dead_h is not None and dur_h >= dead_h and pos['peak'] < entry + 0.5 * atr_price:
        return {'exit_reason': 'ATR_DEAD', 'exit_price': current_price, 'exit_time': now}
    if dur_h >= max_hold_days * 24:
        return {'exit_reason': 'TIME_LIMIT', 'exit_price': current_price, 'exit_time': now}
    return None


class Symbol4H:
    """شموع 4h لعملة مع مؤشرات MACD/RSI/OBV/ATR/ADX محسوبة مسبقاً على كامل السلسلة."""
    __slots__ = ('symbol', 'open_time', 'close_time', 'open', 'high', 'low',
                 'close', 'volume', 'quote_volume', 'macd', 'rsi', 'obv',
                 'atr_ratio', 'adx', 'n')

    def __init__(self, symbol, df):
        self.symbol = symbol
        self.open_time = df['open_time'].to_numpy(dtype=np.int64)
        self.close_time = df['close_time'].to_numpy(dtype=np.int64)
        self.open = df['open'].to_numpy(dtype=np.float64)
        self.high = df['high'].to_numpy(dtype=np.float64)
        self.low = df['low'].to_numpy(dtype=np.float64)
        self.close = df['close'].to_numpy(dtype=np.float64)
        self.volume = df['volume'].to_numpy(dtype=np.float64)
        self.quote_volume = df['quote_volume'].to_numpy(dtype=np.float64)
        self.n = len(self.close)
        # مؤشرات على كامل السلسلة (مطابقة للدوال الأصلية)
        self.macd = S.calc_macd(self.close, S.MACD_FAST, S.MACD_SLOW)
        self.rsi = S.calc_rsi(self.close, S.RSI_PERIOD)
        self.obv = S.calc_obv(self.close, self.volume)
        # فلتر جودة الدخول (ATR/ADX) — معطّل افتراضياً في scan
        self.atr_ratio = _atr_ratio_series(self.high, self.low, self.close, 14)
        self.adx = _adx_series(self.high, self.low, self.close, 14)


class SymbolDaily:
    """مؤشرات يومية محسوبة على أيام مغلقة (EMA99/EMA50/MACD يومي/sigma)."""
    __slots__ = ('symbol', 'open_time', 'close_time', 'day_open', 'close',
                 'ema99', 'ema50', 'macd_line', 'macd_signal', 'sigma', 'valid', 'n')

    def __init__(self, symbol, df):
        self.symbol = symbol
        self.valid = len(df) >= (S.AD_EMA50_DAILY_PERIOD + S.AD_MACD_DAILY_SLOW +
                                 S.AD_MACD_DAILY_SIGNAL + 5)
        self.open_time = df['open_time'].to_numpy(dtype=np.int64)
        self.close_time = df['close_time'].to_numpy(dtype=np.int64)
        self.day_open = df['open'].to_numpy(dtype=np.float64)
        c = df['close']
        self.close = c.to_numpy(dtype=np.float64)
        self.n = len(self.close)
        self.ema99 = c.ewm(span=S.EMA99_PERIOD, adjust=False).mean().to_numpy()
        self.ema50 = c.ewm(span=S.AD_EMA50_DAILY_PERIOD, adjust=False).mean().to_numpy()
        ema_f = c.ewm(span=S.AD_MACD_DAILY_FAST, adjust=False).mean()
        ema_s = c.ewm(span=S.AD_MACD_DAILY_SLOW, adjust=False).mean()
        line = ema_f - ema_s
        self.macd_line = line.to_numpy()
        self.macd_signal = line.ewm(span=S.AD_MACD_DAILY_SIGNAL, adjust=False).mean().to_numpy()
        # sigma: std لـ 14 عائد (مطابق calc_sigma: آخر 15 إغلاق)
        self.sigma = c.pct_change().rolling(S.SIGMA_LOOKBACK_DAYS).std().to_numpy()


def fast_signal(s4: Symbol4H, end_idx: int):
    """
    يعيد إنتاج calc_signals باستخدام المؤشرات المحسوبة مسبقاً.
    end_idx = فهرس آخر شمعة 4h مغلقة (slice[-1]).
    النافذة = آخر 200 شمعة منتهية عند end_idx (مطابق limit=200 في البوت).
    """
    start = max(0, end_idx - 199)
    win_n = end_idx - start + 1
    if win_n < S.MACD_SLOW + 1:
        return {'has_signal': False, 'bullish_macd': False, 'bullish_rsi': False,
                'bullish_obv': False, 'price': float(s4.close[end_idx]), 'volume_24h': 0.0}
    highs = s4.high[start:end_idx + 1]
    lows = s4.low[start:end_idx + 1]
    pivots = S.zigzag_fixed(highs, lows, S.DIV_ZIGZAG_PCT)
    # المؤشرات على نفس النافذة (نأخذ من السلسلة الكاملة عند الفهارس العالمية)
    macd_win = s4.macd[start:end_idx + 1]
    rsi_win = s4.rsi[start:end_idx + 1]
    obv_win = s4.obv[start:end_idx + 1]
    n = win_n
    b_macd = b_rsi = b_obv = False
    for offset in range(S.SIGNAL_LOOKBACK_BARS):
        confirm_idx = n - 2 - offset
        if confirm_idx < 0:
            break
        if not b_macd:
            b_macd = S.detect_bullish_divergence(macd_win, pivots, confirm_idx)
        if not b_rsi:
            b_rsi = S.detect_bullish_divergence(rsi_win, pivots, confirm_idx)
        if not b_obv:
            b_obv = S.detect_bullish_divergence(obv_win, pivots, confirm_idx)
    # vol_24h = آخر 6 شموع مغلقة [-7:-1] داخل النافذة
    if win_n >= 7:
        qv = s4.quote_volume[end_idx - 6:end_idx]   # 6 عناصر = window[-7:-1]
        vol_24h = float(qv.sum())
    else:
        vol_24h = float(s4.quote_volume[start:end_idx].sum())
    return {
        'has_signal': bool(b_macd or b_rsi or b_obv),
        'bullish_macd': bool(b_macd), 'bullish_rsi': bool(b_rsi), 'bullish_obv': bool(b_obv),
        'price': float(s4.close[end_idx]), 'volume_24h': vol_24h,
    }


def breakout_signal(s4: Symbol4H, ei: int, lookback=20, mom_lookback=10):
    """
    إشارة زخم/اختراق (بديل الـ divergence — تشتري القوة لا الهبوط):
      • اختراق Donchian: إغلاق الشمعة ei ≥ أعلى high في آخر `lookback` شمعة (قبلها).
      • تأكيد زخم: العائد على آخر `mom_lookback` شمعة موجب.
    سببي تماماً (يستخدم شموع مغلقة فقط). يعيد dict بنفس مفاتيح fast_signal.
    """
    out = {'has_signal': False, 'bullish_macd': False, 'bullish_rsi': False,
           'bullish_obv': False, 'breakout': False, 'mom': 0.0,
           'price': float(s4.close[ei]), 'volume_24h': 0.0}
    if ei < lookback + 1 or ei < mom_lookback + 1:
        return out
    close_ei = float(s4.close[ei])
    prior_high = float(s4.high[ei - lookback:ei].max())  # أعلى قمة في الشموع السابقة
    mom = close_ei / float(s4.close[ei - mom_lookback]) - 1.0
    is_breakout = close_ei >= prior_high and mom > 0
    out['breakout'] = bool(is_breakout)
    out['has_signal'] = bool(is_breakout)
    out['mom'] = float(mom)
    return out


# ═══════════════════════════════════════════════════════════════════
# المحرك
# ═══════════════════════════════════════════════════════════════════

class Backtester:
    def __init__(self, start_capital=5000.0, year=2025,
                 universe=None, max_symbols=None, verbose=True):
        self.start_capital = start_capital
        self.year = year
        self.verbose = verbose
        self.sim_start = datetime(year, 1, 1, tzinfo=timezone.utc)
        self.sim_end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
        # نسمح بإدارة الصفقات حتى 31 يوم بعد نهاية السنة لإغلاقها طبيعياً
        self.manage_end = self.sim_end + timedelta(days=31)
        # ✅ تواريخ الإحماء نسبية للسنة (لا ثوابت مثبّتة):
        #   4h: نحتاج ~200 شمعة (≈33 يوم) → 70 يوم احتياط
        #   1d: نحتاج EMA99 + sigma (~110 يوم) + احتياط → 270 يوم
        self.warmup_4h_start = self.sim_start - timedelta(days=70)
        self.warmup_1d_start = self.sim_start - timedelta(days=270)

        if universe is None:
            universe = data.list_usdt_pairs()
        # استبعاد العملات المستقرة/الرافعة مسبقاً (سترفض في الـ pipeline على أي حال)
        uni = []
        for sym in universe:
            base = sym[:-4] if sym.endswith('USDT') else sym
            if base.upper() in S.NEVER_TOUCH_ASSETS:
                continue
            if S.is_leveraged_token(base):
                continue
            uni.append(sym)
        if max_symbols:
            uni = uni[:max_symbols]
        self.universe = uni

        self.data4h = {}      # symbol -> Symbol4H
        self.daily = {}       # symbol -> SymbolDaily (lazy)
        self._daily_missing = set()
        self.k15 = {}         # symbol -> DataFrame 15m (lazy) indexed by open_time
        self._k15_loaded = {} # symbol -> set(months) محمّلة

        # state محاكى (مطابق بنية البوت لإعادة استخدام التقارير)
        self.state = {
            PLATFORM: {
                'liquid_capital': start_capital,
                'open_positions': {},
                'history': [],
                'cooldowns': {},
                'monthly_reserve': 0.0,
                'transfer_reserve': 0.0,
            }
        }
        self.equity_log = []  # (ts_ms, equity)

        # فلتر جودة الدخول (ATR/ADX) — معطّل افتراضياً (لا يؤثر على التشغيل الأصلي)
        self.quality_filter = False
        self.atr_ratio_min = 0.05
        self.adx_min = 40.0
        # خروج ديناميكي مرتبط بالـ ATR — معطّل افتراضياً
        self.atr_exit = False
        self.atr_sl_mult = 1.5
        self.atr_trail_mult = 2.5
        self.atr_dead_h = None
        # إشارة الدخول: 'divergence' (الأصلية) | 'breakout' | 'breakout_rs'
        self.entry_mode = 'divergence'
        self.breakout_lookback = 20
        self.breakout_mom = 10
        self.rs_lookback = 30          # شموع 4h لحساب القوة النسبية مقابل BTC
        self.rs_margin = 0.0           # يجب أن يتفوّق على BTC بهذا الهامش (مثلاً 0.05 = +5%)

    # ─── log ───
    def _log(self, msg):
        if self.verbose:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

    # ─── تحميل 4h لكل العملات (متوازي) ───
    def preload_4h(self, workers=16):
        self._log(f"تحميل 4h لـ {len(self.universe)} عملة...")
        end = self.sim_end
        def _load(sym):
            df = data.get_klines_df(sym, '4h', self.warmup_4h_start, end)
            return sym, df
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_load, s): s for s in self.universe}
            for fut in as_completed(futs):
                sym = futs[fut]
                try:
                    s, df = fut.result()
                    if len(df) >= 50:
                        self.data4h[s] = Symbol4H(s, df)
                except Exception as e:
                    self._log(f"  تخطّي {sym}: {str(e)[:60]}")
                done += 1
                if done % 50 == 0:
                    self._log(f"  4h: {done}/{len(self.universe)}")
        self.universe = [s for s in self.universe if s in self.data4h]
        self._log(f"4h جاهز لـ {len(self.universe)} عملة.")

    # ─── daily preload (متوازي) ───
    def preload_daily(self, workers=16):
        self._log(f"تحميل 1d لـ {len(self.universe)} عملة...")
        def _load(sym):
            df = data.get_klines_df(sym, '1d', self.warmup_1d_start, self.sim_end)
            return sym, df
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_load, s): s for s in self.universe}
            for fut in as_completed(futs):
                sym = futs[fut]
                try:
                    s, df = fut.result()
                    if len(df) >= 60:
                        self.daily[s] = SymbolDaily(s, df)
                    else:
                        self._daily_missing.add(s)
                except Exception:
                    self._daily_missing.add(sym)
                done += 1
                if done % 100 == 0:
                    self._log(f"  1d: {done}/{len(self.universe)}")
        self._log(f"1d جاهز لـ {len(self.daily)} عملة.")

    # ─── daily lazy ───
    def get_daily(self, sym):
        if sym in self.daily:
            return self.daily[sym]
        if sym in self._daily_missing:
            return None
        try:
            df = data.get_klines_df(sym, '1d', self.warmup_1d_start, self.sim_end)
        except Exception:
            df = pd.DataFrame()
        if len(df) < 60:
            self._daily_missing.add(sym)
            return None
        sd = SymbolDaily(sym, df)
        self.daily[sym] = sd
        return sd

    # ─── 15m lazy (يحمّل الشهر عند الحاجة) ───
    def _ensure_15m_month(self, sym, year, month):
        loaded = self._k15_loaded.setdefault(sym, set())
        if (year, month) in loaded:
            return
        start = datetime(year, month, 1, tzinfo=timezone.utc)
        end = (datetime(year + 1, 1, 1, tzinfo=timezone.utc) if month == 12
               else datetime(year, month + 1, 1, tzinfo=timezone.utc))
        try:
            df = data.get_klines_df(sym, '15m', start, end)
        except Exception:
            df = pd.DataFrame()
        loaded.add((year, month))
        if df.empty:
            return
        if sym in self.k15:
            self.k15[sym] = pd.concat([self.k15[sym], df]).drop_duplicates('open_time').sort_values('open_time')
        else:
            self.k15[sym] = df.sort_values('open_time').reset_index(drop=True)

    def get_15m_candle(self, sym, open_time_ms):
        dt = datetime.fromtimestamp(open_time_ms / 1000, tz=timezone.utc)
        self._ensure_15m_month(sym, dt.year, dt.month)
        df = self.k15.get(sym)
        if df is None:
            return None
        row = df[df['open_time'] == open_time_ms]
        if row.empty:
            return None
        r = row.iloc[0]
        return {'open': float(r['open']), 'high': float(r['high']),
                'low': float(r['low']), 'close': float(r['close']),
                'close_time': int(r['close_time'])}

    # ─── as-of فهرس يومي ───
    @staticmethod
    def _daily_asof_idx(sd: SymbolDaily, t_ms):
        # آخر يوم مغلق close_time <= t_ms
        i = np.searchsorted(sd.close_time, t_ms, side='right') - 1
        return i

    @staticmethod
    def _daily_open_idx(sd: SymbolDaily, t_ms):
        # اليوم الذي يحوي t (open_time <= t < open_time+يوم)
        i = np.searchsorted(sd.open_time, t_ms, side='right') - 1
        return i

    @staticmethod
    def _4h_closed_idx(s4: Symbol4H, t_ms):
        # آخر شمعة 4h close_time <= t_ms
        i = np.searchsorted(s4.close_time, t_ms, side='right') - 1
        return i

    # ─── equity / available ───
    def _equity(self):
        ps = self.state[PLATFORM]
        eq = ps['liquid_capital']
        for p in ps['open_positions'].values():
            eq += p['entry_price'] * p['quantity']
        return eq

    def _available(self, equity):
        ps = self.state[PLATFORM]
        reserves = equity * S.FEE_RESERVE_PCT  # monthly/transfer = 0
        return max(0.0, ps['liquid_capital'] - reserves)

    # ═══════════════════════════════════════════════════════════════
    # المسح (مطابق search_signals)
    # ═══════════════════════════════════════════════════════════════
    def scan(self, t_ms):
        ps = self.state[PLATFORM]
        open_symbols = set(ps['open_positions'].keys())
        open_count = len(open_symbols)
        now_sec = t_ms / 1000.0
        candidates = []
        # القوة النسبية مقابل BTC: عائد BTC على آخر rs_lookback شمعة (سببي)
        btc_ret = None
        if self.entry_mode == 'breakout_rs':
            btc4 = self.data4h.get('BTCUSDT')
            if btc4 is not None:
                bei = self._4h_closed_idx(btc4, t_ms)
                if bei >= self.rs_lookback:
                    btc_ret = float(btc4.close[bei]) / float(btc4.close[bei - self.rs_lookback]) - 1.0
        for sym in self.universe:
            if sym in open_symbols:
                continue
            if S.is_in_cooldown(self.state, PLATFORM, sym, now_sec):
                continue
            s4 = self.data4h[sym]
            ei = self._4h_closed_idx(s4, t_ms)
            if ei < 0:
                continue
            win_n = min(ei + 1, 200)
            if win_n < 50:
                continue
            # vol_24h (آخر 6 شموع مغلقة قبل الإشارة)
            if ei >= 6:
                vol_24h = float(s4.quote_volume[ei - 6:ei].sum())
            else:
                vol_24h = float(s4.quote_volume[:ei].sum())
            if vol_24h < S.MIN_VOLUME_BINANCE:
                continue
            # ─── إشارة الدخول (حسب entry_mode) ───
            if self.entry_mode == 'divergence':
                sig = fast_signal(s4, ei)
                if not sig['has_signal']:
                    continue
                # OBV filter (#47) — خاص بالـ divergence
                _obv, _macd, _rsi = sig['bullish_obv'], sig['bullish_macd'], sig['bullish_rsi']
                if _obv and _macd:
                    continue
                if _obv and not _macd and not _rsi:
                    continue
            else:  # 'breakout' أو 'breakout_rs'
                sig = breakout_signal(s4, ei, self.breakout_lookback, self.breakout_mom)
                if not sig['has_signal']:
                    continue
                if self.entry_mode == 'breakout_rs':
                    # قوة نسبية: العملة تتفوّق على BTC على نفس الفترة
                    if ei < self.rs_lookback or btc_ret is None:
                        continue
                    coin_ret = float(s4.close[ei]) / float(s4.close[ei - self.rs_lookback]) - 1.0
                    if coin_ret <= btc_ret + self.rs_margin:
                        continue
            current_price = sig['price']
            # فلتر جودة الدخول (ATR/ADX) — معطّل افتراضياً
            if self.quality_filter:
                ar = s4.atr_ratio[ei]
                ad = s4.adx[ei]
                if not (np.isfinite(ar) and np.isfinite(ad)):
                    continue
                if ar < self.atr_ratio_min or ad < self.adx_min:
                    continue
            # A+D daily filter (#48)
            sd = self.get_daily(sym)
            di = self._daily_asof_idx(sd, t_ms) if sd else -1
            if sd is None or not sd.valid or di < 0:
                cond_A = cond_D = False
                ad_valid = False
            else:
                # البوت (calc_ad_daily): إغلاق آخر يوم مغلق مقابل EMA50 اليومي
                cond_A = bool(sd.close[di] >= sd.ema50[di])
                cond_D = bool(sd.macd_line[di] > sd.macd_signal[di])
                ad_valid = not (np.isnan(sd.ema50[di]) or np.isnan(sd.macd_line[di]))
            ad_ok, ad_reason = S.ad_priority_accept(cond_A, cond_D, ad_valid,
                                                    open_count, S.MAX_OPEN_POSITIONS)
            if not ad_ok:
                continue
            sig['ad_classification'] = ad_reason
            # F3 filter (#51): السعر فوق افتتاح اليوم * 0.99
            if S.F3_FILTER_ENABLED and sd is not None:
                oi = self._daily_open_idx(sd, t_ms)
                if 0 <= oi < sd.n:
                    open_today = float(sd.day_open[oi])
                    if open_today > 0 and current_price <= open_today * S.F3_OPEN_MULTIPLIER:
                        continue
            # EMA99 filter (#20)
            if S.EMA99_FILTER_ENABLED:
                ema99 = float(sd.ema99[di]) if (sd is not None and di >= 0) else 0.0
                if ema99 <= 0 or np.isnan(ema99) or current_price < ema99 * S.EMA99_TOLERANCE:
                    continue
            # Vol Ratio filter (#49) — Binance
            if S.VOL_RATIO_FILTER_ENABLED and ei >= S.VOL_RATIO_PERIOD:
                vol_at_signal = float(s4.volume[ei])
                vol_ma = float(s4.volume[ei - S.VOL_RATIO_PERIOD:ei].mean())
                if vol_ma > 0:
                    if (vol_at_signal / vol_ma) < S.VOL_RATIO_THRESHOLD:
                        continue
            # sigma
            sigma = float(sd.sigma[di]) if (sd is not None and di >= 0) else 0.0
            if sigma <= 0 or np.isnan(sigma):
                continue
            # volatility_96h (للترتيب)
            lookback = min(24, win_n - 1)
            rh = s4.high[ei - lookback:ei]
            rl = s4.low[ei - lookback:ei]
            rc = s4.close[ei - lookback:ei]
            avg_close = float(rc.mean()) if len(rc) else current_price
            volatility_96h = ((float(rh.max()) - float(rl.min())) / avg_close) if avg_close > 0 and len(rh) else 0.0
            candidates.append({
                'symbol': sym, 'price': current_price, 'daily_volume': vol_24h,
                'sigma': sigma, 'signals': sig, 'volatility_96h': volatility_96h,
                'strategy_used': ['v3'], 'v4_signal': None,
                'atr_ratio': float(s4.atr_ratio[ei]),
            })
        # ترتيب composite (مطابق نهاية search_signals)
        rank_candidates(candidates, self.verbose)
        return candidates

    # ═══════════════════════════════════════════════════════════════
    # فتح صفقات (مطابق open_new_positions + calc_position_sizes)
    # ═══════════════════════════════════════════════════════════════
    def open_positions(self, candidates, t_ms):
        ps = self.state[PLATFORM]
        open_count = len(ps['open_positions'])
        slots = S.MAX_OPEN_POSITIONS - open_count
        if slots <= 0 or not candidates:
            return
        equity = self._equity()
        available = self._available(equity)
        if available <= 0:
            return
        sized = S.calc_position_sizes(candidates[:slots], available, equity, PLATFORM)
        for cand in sized:
            entry_price = cand['price']
            size = cand['final_size']
            qty = size / entry_price
            cost_with_fee = size * (1 + FEE)
            if cost_with_fee > ps['liquid_capital']:
                continue
            ps['liquid_capital'] -= cost_with_fee
            pos = {
                'symbol': cand['symbol'], 'entry_time': t_ms / 1000.0,
                'entry_time_ms': t_ms, 'entry_price': entry_price,
                'quantity': qty, 'size_usd': size,
                'sigma_at_entry': cand['sigma'],
                'daily_volume_at_entry': cand['daily_volume'],
                'equity_at_entry': equity, 'signals': cand['signals'],
                'sl_price': entry_price * (1 + S.STOP_LOSS_PCT),
                'tp_threshold_price': entry_price * (1 + S.TP_THRESHOLD),
                'peak': entry_price, 'trail_active': False,
                'smart_tp_mode': False, 'smart_tp_peak': entry_price,
                'strategy_used': ['v3'],
            }
            # ATR بوحدة السعر وقت الدخول (للخروج الديناميكي)
            ar = cand.get('atr_ratio', np.nan)
            pos['atr_price'] = (ar * entry_price) if (ar and np.isfinite(ar) and ar > 0) \
                else abs(S.STOP_LOSS_PCT) / max(self.atr_sl_mult, 1e-9) * entry_price
            ps['open_positions'][cand['symbol']] = pos

    # ═══════════════════════════════════════════════════════════════
    # إدارة الخروج عند tick (15m)
    # ═══════════════════════════════════════════════════════════════
    def manage_exits(self, t_ms):
        ps = self.state[PLATFORM]
        if not ps['open_positions']:
            return
        # نقيّم شمعة 15m التي أُغلقت للتو [t-15m, t) (تفادي lookahead)
        bar_open = t_ms - MS_15M
        to_close = []
        for sym, pos in list(ps['open_positions'].items()):
            candle = self.get_15m_candle(sym, bar_open)
            if candle is None:
                continue
            now_sec = candle['close_time'] / 1000.0
            if self.atr_exit:
                res = check_exit_atr(pos, candle['close'], now_sec,
                                     pos.get('atr_price', abs(S.STOP_LOSS_PCT) * pos['entry_price']),
                                     self.atr_sl_mult, self.atr_trail_mult,
                                     candle=candle, dead_h=self.atr_dead_h)
            else:
                res = S.check_exit(pos, candle['close'], now_sec, candle=candle)
            if not res:
                continue
            exit_price = res['exit_price']
            qty = pos['quantity']
            pnl = S.calc_pnl(pos, exit_price, FEE)
            net_proceeds = (exit_price * qty) * (1 - FEE)
            ps['liquid_capital'] += net_proceeds
            trade = {
                **pos, **res, **pnl,
                'exit_price': exit_price,
                'duration_hours': (res['exit_time'] - pos['entry_time']) / 3600,
                'peak_price': pos['peak'],
                'peak_pct': ((pos['peak'] - pos['entry_price']) / pos['entry_price']) * 100,
                'smart_tp_activated': pos.get('smart_tp_mode', False),
                'smart_tp_peak': pos.get('smart_tp_peak', pos['entry_price']),
                'platform': PLATFORM,
            }
            to_close.append((sym, trade))
            S.update_cooldown(self.state, PLATFORM, sym, trade)
        for sym, trade in to_close:
            del ps['open_positions'][sym]
            ps['history'].append(trade)

    def force_close_all(self, t_ms, reason='EOY_FORCE'):
        ps = self.state[PLATFORM]
        for sym, pos in list(ps['open_positions'].items()):
            s4 = self.data4h.get(sym)
            price = pos['entry_price']
            if s4 is not None:
                ei = self._4h_closed_idx(s4, t_ms)
                if ei >= 0:
                    price = float(s4.close[ei])
            qty = pos['quantity']
            pnl = S.calc_pnl(pos, price, FEE)
            ps['liquid_capital'] += (price * qty) * (1 - FEE)
            trade = {
                **pos, 'exit_reason': reason, 'exit_price': price,
                'exit_time': t_ms / 1000.0, **pnl,
                'duration_hours': (t_ms / 1000.0 - pos['entry_time']) / 3600,
                'peak_price': pos['peak'],
                'peak_pct': ((pos['peak'] - pos['entry_price']) / pos['entry_price']) * 100,
                'smart_tp_activated': pos.get('smart_tp_mode', False),
                'smart_tp_peak': pos.get('smart_tp_peak', pos['entry_price']),
                'platform': PLATFORM,
            }
            ps['history'].append(trade)
            del ps['open_positions'][sym]

    # ═══════════════════════════════════════════════════════════════
    # الحلقة الرئيسية
    # ═══════════════════════════════════════════════════════════════
    def run(self):
        ps = self.state[PLATFORM]
        t = int(self.sim_start.timestamp() * 1000)
        scan_end = int(self.sim_end.timestamp() * 1000)
        manage_end = int(self.manage_end.timestamp() * 1000)
        next_equity_log = t
        n_scans = 0
        self._log("بدء المحاكاة...")
        while t < manage_end:
            # 1) إدارة الخروج كل 15m
            self.manage_exits(t)
            # 2) المسح كل 4h (فقط ضمن السنة)
            if t < scan_end and (t % MS_4H == 0):
                if len(ps['open_positions']) < S.MAX_OPEN_POSITIONS:
                    cands = self.scan(t)
                    if cands:
                        self.open_positions(cands, t)
                n_scans += 1
                if n_scans % 180 == 0:  # كل ~30 يوم
                    eq = self._equity()
                    dt = datetime.fromtimestamp(t / 1000, tz=timezone.utc)
                    self._log(f"  {dt.strftime('%Y-%m-%d')} | equity=${eq:,.2f} | "
                              f"open={len(ps['open_positions'])} | trades={len(ps['history'])}")
            # 3) تسجيل equity يومياً
            if t >= next_equity_log:
                self.equity_log.append((t, self._equity()))
                next_equity_log += MS_DAY
            t += MS_15M
        # إغلاق المتبقي قسراً في نهاية فترة الإدارة
        self.force_close_all(manage_end - MS_15M, reason='EOY_FORCE')
        self.equity_log.append((manage_end, self._equity()))
        self._log(f"انتهت المحاكاة: {len(ps['history'])} صفقة | "
                  f"equity نهائي=${self._equity():,.2f}")
        return self.state


# ═══════════════════════════════════════════════════════════════════
# مساعدات الترتيب (مطابقة نهاية search_signals)
# ═══════════════════════════════════════════════════════════════════

def current_price_geq(a, b):
    return a >= b


def rank_candidates(candidates, verbose=False):
    if len(candidates) >= 10:
        sorted_by_vol = sorted(candidates, key=lambda c: c['daily_volume'])
        n = len(sorted_by_vol)
        median_idx = n // 2
        small_idx = n // 4
        large_idx = (n * 3) // 4
        max_dist_in_medium = max((large_idx - small_idx) // 2, 1)
        for i, c in enumerate(sorted_by_vol):
            distance = abs(i - median_idx)
            is_above_median = i >= median_idx
            if small_idx <= i <= large_idx:
                proximity = max(1.0 - (distance / max_dist_in_medium), 0.0)
                proximity_bonus = 50 + (50 * proximity)
            elif i < small_idx:
                proximity_bonus = 20 + (20 * (i / small_idx)) if small_idx > 0 else 20
            else:
                position_in_large = (i - large_idx) / max(n - 1 - large_idx, 1)
                proximity_bonus = 40 - (20 * position_in_large)
            c['proximity_bonus'] = proximity_bonus
            c['upward_bias'] = 10 if is_above_median else 0
    else:
        for c in candidates:
            c['proximity_bonus'] = 50
            c['upward_bias'] = 0

    def _sigma_sweet_spot_bonus(sigma):
        if sigma < 0.01:
            return 10
        elif sigma < 0.02:
            return 30
        elif sigma <= 0.04:
            return 50
        elif sigma <= 0.06:
            return 30
        else:
            return 10

    def _volatility_penalty(vol_96h):
        if vol_96h > 0.30:
            return -20
        elif vol_96h < 0.05:
            return -10
        else:
            return 0

    def _composite_score(c):
        sig_count = (int(c['signals'].get('bullish_macd', False)) +
                     int(c['signals'].get('bullish_rsi', False)) +
                     int(c['signals'].get('bullish_obv', False)))
        signal_bonus = sig_count * 50
        proximity_bonus = c.get('proximity_bonus', 50)
        upward_bias = c.get('upward_bias', 0)
        sigma_bonus = _sigma_sweet_spot_bonus(c['sigma'])
        vol_penalty = _volatility_penalty(c.get('volatility_96h', 0.0))
        return signal_bonus + proximity_bonus + upward_bias + sigma_bonus + vol_penalty

    candidates.sort(key=_composite_score, reverse=True)
