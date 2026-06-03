#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
strategy_core.py — نسخة دقيقة من منطق الاستراتيجية في Real Bot v3.27
تُستخدم في المحاكاة التاريخية (backtest). كل الدوال هنا منسوخة حرفياً من
الكود الأصلي (real_bot v3.27/v3.28) باستثناء check_exit التي عُدّلت لتأخذ
الوقت المحاكى (now_ts) بدلاً من time.time().

المصدر الوحيد للحقيقة للمنطق التداولي — أي تعديل هنا يجب أن يطابق البوت.
"""

import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════════
# الثوابت (مطابقة للبوت الأصلي — مسار Binance النشط)
# ═══════════════════════════════════════════════════════════════════

TIMEFRAME_SIGNAL = '4h'
DIV_ZIGZAG_PCT = 5.0
EMA99_PERIOD = 99
EMA99_TOLERANCE = 0.90
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
RSI_PERIOD = 14
SIGNAL_LOOKBACK_BARS = 3

# Vol Ratio Filter (#49) — Binance only
VOL_RATIO_FILTER_ENABLED = True
VOL_RATIO_THRESHOLD = 0.7
VOL_RATIO_PERIOD = 20

# F3 Filter (#51)
F3_FILTER_ENABLED = True
F3_OPEN_MULTIPLIER = 0.99

# Strategy v4 — معطّلة
STRATEGY_V4_ENABLED = False

# Trade Management
STOP_LOSS_PCT = -0.04
TP_THRESHOLD = 0.15
TRAILING_ACTIVATE_PCT = 0.03
TRAILING_RATIO = 0.98
SMART_TP_TRAIL_RATIO = 0.97
TIME_LIMIT_HOURS = 24
MAX_HOLD_DAYS = 30

# Position Sizing
MAX_OPEN_POSITIONS = 5
SLIPPAGE_BUDGET = 0.0015
SIGMA_LOOKBACK_DAYS = 14
MAX_CAP_DOLLAR = 40_000
EQUITY_HALF_LIMIT = 0.5
MIN_TRADE_BINANCE_ABS = 12

# Volume Filter (Binance)
MIN_VOLUME_BINANCE = 1_500_000

# Cooldown System
COOLDOWN_HOURS_NORMAL = 72
COOLDOWN_HOURS_STRICT = 168
COOLDOWN_FAILURES_FOR_STRICT = 3
COOLDOWN_PEAK_THRESHOLD = 0.02
COOLDOWN_HOURS_SMART_TP = 12
COOLDOWN_HOURS_TRAIL_GOOD = 6
COOLDOWN_HOURS_TRAIL_WEAK = 24
COOLDOWN_HOURS_TIME_EXIT = 48
TRAIL_GOOD_PEAK_THRESHOLD = 0.03
COOLDOWN_HOURS_EARLY_EXIT = 24

# Early Exit — معطّلة
EARLY_EXIT_ENABLED = False
EARLY_EXIT_HOURS = 6
EARLY_EXIT_PEAK_THRESHOLD = 0.025

# EMA99 Daily Filter (#20)
EMA99_FILTER_ENABLED = True

# A+D Daily Filter (#48)
AD_FILTER_ENABLED = True
AD_EMA50_DAILY_PERIOD = 50
AD_MACD_DAILY_FAST = 12
AD_MACD_DAILY_SLOW = 26
AD_MACD_DAILY_SIGNAL = 9

# Fees & Reserves
EFFECTIVE_FEE_BINANCE = 0.00075
FEE_RESERVE_PCT = 0.00375

# Leveraged tokens detection
LEVERAGED_TOKEN_SUFFIXES = (
    'UP', 'DOWN', '3L', '3S', '5L', '5S',
    'BULL', 'BEAR', 'HALF', 'HEDGE',
)

NEVER_TOUCH_ASSETS = {
    'USDT', 'USDC', 'BUSD', 'FDUSD', 'TUSD', 'DAI', 'PAX',
    'USDe', 'USDP', 'GUSD', 'USDD', 'PYUSD', 'USDS',
    'EUR', 'EURC', 'EURT', 'EURI',
    'GBP', 'TRY', 'BRL', 'AUD', 'JPY', 'RUB', 'ARS', 'ZAR',
    'LUSD', 'SUSD', 'MIM', 'FRAX', 'USTC',
    'PAXG', 'XAUT', 'KAU', 'DGX', 'CACHE', 'GLDX', 'AURA', 'AWG', 'XAU',
    'KAG', 'WPMC', 'AWS', 'XAG',
    'XPT', 'XPD',
    'PETRO', 'OIL', 'WTI',
}
# طبيع الـ set ليكون كله uppercase للمقارنة المتسقة
NEVER_TOUCH_ASSETS = {a.upper() for a in NEVER_TOUCH_ASSETS}


def is_leveraged_token(asset: str) -> bool:
    if not asset or len(asset) < 4:
        return False
    asset_upper = asset.upper()
    for suffix in LEVERAGED_TOKEN_SUFFIXES:
        if asset_upper.endswith(suffix):
            base = asset_upper[:-len(suffix)]
            if len(base) >= 2 and base.isalpha():
                return True
    return False


# ═══════════════════════════════════════════════════════════════════
# ZigZag + Indicators + Signals (منسوخة حرفياً)
# ═══════════════════════════════════════════════════════════════════

def zigzag_fixed(highs, lows, threshold_pct=5.0):
    n = len(highs)
    if n < 3:
        return {'pivot_idx': [], 'pivot_type': [], 'pivot_price': [], 'pivot_confirm_idx': []}
    threshold = threshold_pct / 100.0
    pivot_idx, pivot_type, pivot_price, pivot_confirm_idx = [], [], [], []
    direction = 0
    last_pivot_idx = 0
    last_pivot_price = highs[0]
    confirm_break_at = 0
    for i in range(1, n):
        if highs[i] >= last_pivot_price * (1 + threshold):
            direction = 1
            pivot_idx.append(last_pivot_idx)
            pivot_type.append('low')
            pivot_price.append(lows[last_pivot_idx])
            pivot_confirm_idx.append(i)
            last_pivot_idx = i
            last_pivot_price = highs[i]
            confirm_break_at = i
            break
        elif lows[i] <= last_pivot_price * (1 - threshold):
            direction = -1
            pivot_idx.append(last_pivot_idx)
            pivot_type.append('high')
            pivot_price.append(highs[last_pivot_idx])
            pivot_confirm_idx.append(i)
            last_pivot_idx = i
            last_pivot_price = lows[i]
            confirm_break_at = i
            break
        else:
            if highs[i] > last_pivot_price:
                last_pivot_idx = i
                last_pivot_price = highs[i]
    if direction == 0:
        return {'pivot_idx': [], 'pivot_type': [], 'pivot_price': [], 'pivot_confirm_idx': []}
    i = confirm_break_at + 1
    while i < n:
        if direction == 1:
            if highs[i] > last_pivot_price:
                last_pivot_idx = i
                last_pivot_price = highs[i]
            if lows[i] <= last_pivot_price * (1 - threshold):
                pivot_idx.append(last_pivot_idx)
                pivot_type.append('high')
                pivot_price.append(last_pivot_price)
                pivot_confirm_idx.append(i)
                direction = -1
                last_pivot_idx = i
                last_pivot_price = lows[i]
        else:
            if lows[i] < last_pivot_price:
                last_pivot_idx = i
                last_pivot_price = lows[i]
            if highs[i] >= last_pivot_price * (1 + threshold):
                pivot_idx.append(last_pivot_idx)
                pivot_type.append('low')
                pivot_price.append(last_pivot_price)
                pivot_confirm_idx.append(i)
                direction = 1
                last_pivot_idx = i
                last_pivot_price = highs[i]
        i += 1
    return {'pivot_idx': pivot_idx, 'pivot_type': pivot_type,
            'pivot_price': pivot_price, 'pivot_confirm_idx': pivot_confirm_idx}


def calc_macd(close, fast=12, slow=26):
    if len(close) < slow:
        return np.zeros(len(close))
    s = pd.Series(close)
    return (s.ewm(span=fast, adjust=False).mean() -
            s.ewm(span=slow, adjust=False).mean()).values


def calc_rsi(close, period=14):
    if len(close) < period + 1:
        return np.full(len(close), 50.0)
    s = pd.Series(close)
    delta = s.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss.replace(0, 1e-10)
    return (100 - 100 / (1 + rs)).fillna(50.0).values


def calc_obv(close, volume):
    if len(close) < 2:
        return np.zeros(len(close))
    diff = np.diff(close, prepend=close[0])
    return (np.sign(diff) * volume).cumsum()


def detect_bullish_divergence(indicator, pivots, confirm_idx):
    types = pivots['pivot_type']
    idxs = pivots['pivot_idx']
    prices = pivots['pivot_price']
    confirms = pivots['pivot_confirm_idx']
    low_positions = [i for i, t in enumerate(types) if t == 'low']
    if len(low_positions) < 2:
        return False
    p2_pos = low_positions[-1]
    p1_pos = low_positions[-2]
    if confirms[p2_pos] > confirm_idx:
        return False
    idx1 = idxs[p1_pos]
    idx2 = idxs[p2_pos]
    if idx2 >= len(indicator) or idx1 >= len(indicator):
        return False
    return prices[p2_pos] < prices[p1_pos] and indicator[idx2] > indicator[idx1]


def calc_signals(candles, pivots):
    if len(candles) < MACD_SLOW + 1:
        return {'has_signal': False, 'bullish_macd': False, 'bullish_rsi': False,
                'bullish_obv': False, 'price': candles[-1]['close'] if candles else 0,
                'volume_24h': 0}
    closes = np.array([c['close'] for c in candles])
    volumes = np.array([c['volume'] for c in candles])
    macd = calc_macd(closes, MACD_FAST, MACD_SLOW)
    rsi = calc_rsi(closes, RSI_PERIOD)
    obv = calc_obv(closes, volumes)
    n = len(candles)
    b_macd = b_rsi = b_obv = False
    for offset in range(SIGNAL_LOOKBACK_BARS):
        confirm_idx = n - 2 - offset
        if confirm_idx < 0:
            break
        if not b_macd:
            b_macd = detect_bullish_divergence(macd, pivots, confirm_idx)
        if not b_rsi:
            b_rsi = detect_bullish_divergence(rsi, pivots, confirm_idx)
        if not b_obv:
            b_obv = detect_bullish_divergence(obv, pivots, confirm_idx)
    if len(candles) >= 7:
        vol_24h = sum(c.get('quote_volume', c['volume'] * c['close']) for c in candles[-7:-1])
    else:
        vol_24h = sum(c.get('quote_volume', c['volume'] * c['close']) for c in candles[:-1] if candles[:-1])
    return {
        'has_signal': b_macd or b_rsi or b_obv,
        'bullish_macd': b_macd, 'bullish_rsi': b_rsi, 'bullish_obv': b_obv,
        'price': float(closes[-1]), 'volume_24h': float(vol_24h),
    }


# ═══════════════════════════════════════════════════════════════════
# A+D tiered acceptance (منسوخة حرفياً)
# ═══════════════════════════════════════════════════════════════════

def ad_priority_accept(cond_A, cond_D, valid, open_count, max_positions):
    if not valid:
        return False, 'ad_no_daily_data'
    free_slots = max_positions - open_count
    has_AD = cond_A and cond_D
    if free_slots <= 0:
        return False, 'ad_no_slots'
    if free_slots == 1:
        if has_AD:
            return True, 'ad_full'
        return False, 'ad_need_full_AD'
    elif free_slots <= 3:
        if has_AD or cond_A:
            return True, 'ad_full' if has_AD else 'ad_A_only'
        return False, 'ad_need_A'
    else:
        if has_AD or cond_A or cond_D:
            if has_AD:    return True, 'ad_full'
            if cond_A:    return True, 'ad_A_only'
            return True, 'ad_D_only'
        return False, 'ad_none_matched'


# ═══════════════════════════════════════════════════════════════════
# Sizing (منسوخة حرفياً)
# ═══════════════════════════════════════════════════════════════════

def calc_optimal_size(daily_volume, sigma):
    if sigma <= 0 or daily_volume <= 0:
        return 0.0
    return daily_volume * (SLIPPAGE_BUDGET / sigma) ** 2


def get_min_trade_size(platform, equity):
    absolute = MIN_TRADE_BINANCE_ABS  # Binance only
    if equity < 500:
        return absolute
    elif equity < 5000:
        return max(absolute, equity * 0.01)
    elif equity < 50000:
        return max(absolute, equity * 0.005)
    else:
        return max(absolute, 50)


def calc_position_sizes(candidates, available, equity, platform):
    if available <= 0 or not candidates:
        return []
    min_trade = get_min_trade_size(platform, equity)
    e_half = equity * EQUITY_HALF_LIMIT
    for c in candidates:
        opt = calc_optimal_size(c['daily_volume'], c['sigma'])
        c['optimal'] = min(opt, e_half, MAX_CAP_DOLLAR)
    total = sum(c['optimal'] for c in candidates)
    if total <= available:
        for c in candidates:
            c['final_size'] = c['optimal']
            c['scale_factor'] = 1.0
    else:
        scale = available / total
        for c in candidates:
            c['final_size'] = c['optimal'] * scale
            c['scale_factor'] = scale
    return [c for c in candidates if c['final_size'] >= min_trade]


# ═══════════════════════════════════════════════════════════════════
# Exit logic — معدّلة لتأخذ now_ts (الوقت المحاكى)
# ═══════════════════════════════════════════════════════════════════

def check_exit(position, current_price, now_ts, candle=None):
    """نسخة من check_exit الأصلية مع حقن الوقت now_ts بدل time.time()."""
    now = now_ts
    check_high = candle['high'] if candle and 'high' in candle else current_price
    check_low = candle['low'] if candle and 'low' in candle else current_price
    if check_high > position['peak']:
        position['peak'] = check_high
    if position.get('smart_tp_mode'):
        if check_high > position['smart_tp_peak']:
            position['smart_tp_peak'] = check_high
    if not position.get('trail_active') and check_high >= position['entry_price'] * (1 + TRAILING_ACTIVATE_PCT):
        position['trail_active'] = True
    if not position.get('smart_tp_mode') and check_high >= position['tp_threshold_price']:
        position['smart_tp_mode'] = True
        if check_high > position['smart_tp_peak']:
            position['smart_tp_peak'] = check_high
    if position.get('smart_tp_mode'):
        eff_sl = position['smart_tp_peak'] * SMART_TP_TRAIL_RATIO
    elif position.get('trail_active'):
        eff_sl = position['peak'] * TRAILING_RATIO
    else:
        eff_sl = position['sl_price']
    if check_low <= eff_sl:
        if position.get('smart_tp_mode'):
            reason = 'SMART_TP_TRAIL'
        elif position.get('trail_active'):
            reason = 'TRAILING'
        else:
            reason = 'STOP_LOSS'
        return {'exit_reason': reason, 'exit_price': eff_sl, 'exit_time': now}
    duration_h = (now - position['entry_time']) / 3600
    if duration_h >= TIME_LIMIT_HOURS and not position.get('smart_tp_mode'):
        return {'exit_reason': 'TIME_24H', 'exit_price': current_price, 'exit_time': now}
    if duration_h >= MAX_HOLD_DAYS * 24:
        return {'exit_reason': 'TIME_LIMIT', 'exit_price': current_price, 'exit_time': now}
    if (EARLY_EXIT_ENABLED and
            duration_h >= EARLY_EXIT_HOURS and
            not position.get('smart_tp_mode')):
        peak_pct = (position['peak'] / position['entry_price'] - 1)
        if peak_pct < EARLY_EXIT_PEAK_THRESHOLD:
            return {'exit_reason': 'EARLY_EXIT', 'exit_price': current_price, 'exit_time': now}
    return None


def calc_pnl(position, exit_price, fee_rate):
    entry = position['entry_price']
    qty = position['quantity']
    size = position['size_usd']
    gross = (exit_price - entry) * qty
    fees = size * fee_rate + (exit_price * qty) * fee_rate
    net = gross - fees
    pct = (net / size) * 100 if size > 0 else 0
    return {'gross_pnl': gross, 'fees': fees, 'net_pnl': net, 'pnl_pct': pct}


# ═══════════════════════════════════════════════════════════════════
# Cooldown (منسوخة حرفياً، تعمل على state dict محاكى)
# ═══════════════════════════════════════════════════════════════════

def _is_failed_attempt(trade):
    return trade.get('net_pnl', 0) < 0


def _get_cooldown_hours(trade):
    pnl = trade.get('net_pnl', 0)
    peak_pct = trade.get('peak_pct', 0)
    reason = (trade.get('exit_reason') or '').upper()
    if pnl < 0:
        return COOLDOWN_HOURS_NORMAL
    if reason in ('SMART_TP', 'TAKE_PROFIT'):
        return COOLDOWN_HOURS_SMART_TP
    if 'TRAIL' in reason:
        if peak_pct >= TRAIL_GOOD_PEAK_THRESHOLD * 100:
            return COOLDOWN_HOURS_TRAIL_GOOD
        else:
            return COOLDOWN_HOURS_TRAIL_WEAK
    if reason in ('TIME_24H', 'TIME_LIMIT'):
        return COOLDOWN_HOURS_TIME_EXIT
    if reason == 'EARLY_EXIT':
        return COOLDOWN_HOURS_EARLY_EXIT
    return COOLDOWN_HOURS_TRAIL_GOOD


def update_cooldown(state, platform, symbol, trade):
    cooldowns = state[platform].setdefault('cooldowns', {})
    is_failure = _is_failed_attempt(trade)
    cd = cooldowns.setdefault(symbol, {
        'failed_attempts': 0,
        'last_failure_time': 0,
        'last_close_time': 0,
        'last_close_cooldown_hours': 0,
        'last_close_reason': '',
    })
    now = trade.get('exit_time', 0)
    cd['last_close_time'] = now
    cd['last_close_reason'] = trade.get('exit_reason', '')
    if is_failure:
        cd['failed_attempts'] += 1
        cd['last_failure_time'] = now
        if cd['failed_attempts'] >= COOLDOWN_FAILURES_FOR_STRICT:
            cd['last_close_cooldown_hours'] = COOLDOWN_HOURS_STRICT
        else:
            cd['last_close_cooldown_hours'] = COOLDOWN_HOURS_NORMAL
    else:
        cd['failed_attempts'] = 0
        cd['last_close_cooldown_hours'] = _get_cooldown_hours(trade)


def is_in_cooldown(state, platform, symbol, now):
    cd = state[platform].get('cooldowns', {}).get(symbol)
    if not cd:
        return False
    last_close = cd.get('last_close_time', 0)
    cd_hours = cd.get('last_close_cooldown_hours', 0)
    if last_close > 0 and cd_hours > 0:
        hours_since = (now - last_close) / 3600
        if hours_since < cd_hours:
            return True
    last_failure = cd.get('last_failure_time', 0)
    if last_failure > 0:
        hours_since = (now - last_failure) / 3600
        threshold = (COOLDOWN_HOURS_STRICT
                     if cd.get('failed_attempts', 0) >= COOLDOWN_FAILURES_FOR_STRICT
                     else COOLDOWN_HOURS_NORMAL)
        if hours_since < threshold:
            return True
    return False
