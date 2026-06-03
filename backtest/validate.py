#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate.py — يتحقق أن fast_signal (المؤشرات المحسوبة مسبقاً) تطابق
calc_signals الأصلية (تُعيد حساب MACD/RSI/OBV على نافذة 200). كذلك يفحص
عدم وجود lookahead في الفهرسة الزمنية.
"""
from datetime import datetime, timezone

import numpy as np

from . import data
from . import strategy_core as S
from .engine import Symbol4H, fast_signal, WARMUP_4H_START


def validate_signal_equivalence(symbols, year=2025, step=7):
    sim_end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
    total = 0
    mismatch = 0
    examples = []
    for sym in symbols:
        df = data.get_klines_df(sym, '4h', WARMUP_4H_START, sim_end)
        if len(df) < 250:
            continue
        s4 = Symbol4H(sym, df)
        candles = data.df_to_candles(df)
        # نفحص فهارس متعددة (كل step شمعة)، بدءاً من بعد warmup كافٍ
        for ei in range(220, s4.n, step):
            total += 1
            fs = fast_signal(s4, ei)
            # المرجع: آخر 200 شمعة مغلقة منتهية عند ei (مطابق limit=200)
            start = max(0, ei - 199)
            window = candles[start:ei + 1]
            highs = np.array([c['high'] for c in window])
            lows = np.array([c['low'] for c in window])
            pivots = S.zigzag_fixed(highs, lows, S.DIV_ZIGZAG_PCT)
            ref = S.calc_signals(window, pivots)
            same = (fs['has_signal'] == ref['has_signal'] and
                    fs['bullish_macd'] == ref['bullish_macd'] and
                    fs['bullish_rsi'] == ref['bullish_rsi'] and
                    fs['bullish_obv'] == ref['bullish_obv'])
            if not same:
                mismatch += 1
                if len(examples) < 10:
                    examples.append((sym, ei, fs, ref))
    print(f"عينات مفحوصة: {total} | اختلافات: {mismatch} "
          f"({(mismatch/total*100 if total else 0):.3f}%)")
    for sym, ei, fs, ref in examples:
        print(f"  MISMATCH {sym}@{ei}: fast={_b(fs)} ref={_b(ref)}")
    return mismatch, total


def _b(s):
    return f"sig={int(s['has_signal'])} M={int(s['bullish_macd'])} " \
           f"R={int(s['bullish_rsi'])} O={int(s['bullish_obv'])}"


if __name__ == '__main__':
    syms = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'ADAUSDT',
            'DOGEUSDT', 'LINKUSDT', 'AVAXUSDT']
    validate_signal_equivalence(syms, step=5)
