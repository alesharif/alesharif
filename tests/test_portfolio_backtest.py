"""Deterministic, offline tests for the portfolio backtest mechanics.

We monkeypatch the data layer so no network is needed, then assert that
entries, the ATR trailing stop, the hard stop and PnL accounting behave
exactly as the live bot specifies.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from binance_sim import paper_bot_strategy as S  # noqa: E402
from binance_sim import portfolio_backtest as PB  # noqa: E402


def _bars(prices, start_t=0, atr=1.0, vol=5_000_000.0):
    """Build a featurised per-symbol frame with a forced entry signal at t0.

    We bypass compute_features and inject the columns the simulator reads, so
    the test isolates the *portfolio mechanics* from indicator math.
    """
    n = len(prices)
    times = [start_t + i * PB.FOUR_H_MS for i in range(n)]
    df = pd.DataFrame({
        "close_time": times,
        "high": [p[1] for p in prices],
        "low": [p[2] for p in prices],
        "close": [p[0] for p in prices],
        "atr": [atr] * n,
        "vol_pit": [vol] * n,
        "entry_signal": [False] * n,
    })
    return df.set_index("close_time")


def test_trailing_stop_locks_profit(monkeypatch):
    # one symbol; signal on bar 0; price runs up then reverses
    # bars: (close, high, low)
    bars = [
        (100, 100, 100),     # t0 entry @100, atr=1, sl=100-1.5=98.5
        (102, 102, 101.9),   # peak 102 -> trailing on, sl=101.8; low 101.9 stays above
        (103, 103, 102.9),   # peak 103 -> sl=102.8; low 102.9 stays above
        (101, 103, 100.0),   # low 100 <= sl 102.8 -> exit @102.8 (TRAIL)
    ]
    df = _bars(bars, atr=1.0)
    df.iloc[0, df.columns.get_loc("entry_signal")] = True

    monkeypatch.setattr(PB, "build_stable_ratio", lambda *a, **k: {})
    sym_data = {"COINUSDT": df}

    def fake_loader(symbols, start_ms, end_ms, rank="vol", max_hold=0, log=lambda *a: None):
        return PB.run.__wrapped__ if False else None  # placeholder, unused

    # call run() with pre-built data by patching the loaders it uses
    monkeypatch.setattr(PB, "fetch_klines_range", lambda *a, **k: None)

    # Directly drive the inner walk by patching per_symbol construction:
    report = _run_with_data(sym_data, start_ms=0, end_ms=df.index.max())
    assert len(report.trades) == 1
    tr = report.trades[0]
    assert tr.outcome == "TRAIL"
    # exit at 102.8, entry 100 -> +2.8% minus 0.2% fee = +2.6%
    assert abs(tr.net_pct - 2.6) < 1e-6
    assert report.final_capital > report.initial_capital


def test_hard_stop_loss(monkeypatch):
    bars = [
        (100, 100, 100),     # entry @100, sl=98.5
        (99, 100, 97),       # low 97 <= 98.5 -> SL exit @98.5 (trailing never armed)
    ]
    df = _bars(bars, atr=1.0)
    df.iloc[0, df.columns.get_loc("entry_signal")] = True
    report = _run_with_data({"COINUSDT": df}, 0, df.index.max())
    assert len(report.trades) == 1
    tr = report.trades[0]
    assert tr.outcome == "SL"
    # exit 98.5 / 100 -1 = -1.5% minus 0.2% fee = -1.7%
    assert abs(tr.net_pct - (-1.7)) < 1e-6
    assert report.final_capital < report.initial_capital


def test_max_concurrent_cap(monkeypatch):
    # 10 symbols all signalling at t0; only 8 may open
    data = {}
    for i in range(10):
        df = _bars([(100, 100, 100), (100, 100, 100)], atr=1.0, vol=1e6 + i)
        df.iloc[0, df.columns.get_loc("entry_signal")] = True
        data[f"C{i}USDT"] = df
    report = _run_with_data(data, 0, 1 * PB.FOUR_H_MS)
    # no exits triggered; positions are force-closed EOD -> exactly 8 trades
    assert len(report.trades) == S.MAX_CONCURRENT


def test_stable_ratio_blocks_entry():
    df = _bars([(100, 100, 100), (100, 100, 100)], atr=1.0)
    df.iloc[0, df.columns.get_loc("entry_signal")] = True
    # stable ratio above threshold at t0 -> entry blocked
    ratio = {0: S.STABLE_RATIO_MAX + 0.5}
    report = _run_with_data({"COINUSDT": df}, 0, df.index.max(), stable_ratio=ratio)
    assert len(report.trades) == 0


# -- helper that injects pre-built per-symbol data into PB.run -----------
def _run_with_data(sym_data, start_ms, end_ms, stable_ratio=None, **kw):
    import binance_sim.portfolio_backtest as PBmod

    orig_fetch = PBmod.fetch_klines_range
    orig_stable = PBmod.build_stable_ratio
    orig_compute = PBmod.S.compute_features
    orig_attach = PBmod.S.attach_entry_signal

    # Make run() pick up our frames: it calls fetch -> compute -> attach -> set_index.
    # We short-circuit those to return the already-featurised frame (reset to a
    # column so set_index('close_time') works).
    def fake_fetch(symbol, a, b, use_cache=True):
        df = sym_data.get(symbol)
        return None if df is None else df.reset_index()

    orig_minhist = PBmod.S.MIN_HISTORY
    PBmod.fetch_klines_range = fake_fetch
    PBmod.S.compute_features = lambda df: df
    PBmod.S.attach_entry_signal = lambda df: df
    PBmod.S.MIN_HISTORY = 1  # tiny synthetic frames in these mechanics tests
    PBmod.build_stable_ratio = (lambda a, b: (stable_ratio or {}))
    try:
        return PBmod.run(list(sym_data.keys()), start_ms, end_ms, log=lambda *a: None, **kw)
    finally:
        PBmod.fetch_klines_range = orig_fetch
        PBmod.build_stable_ratio = orig_stable
        PBmod.S.compute_features = orig_compute
        PBmod.S.attach_entry_signal = orig_attach
        PBmod.S.MIN_HISTORY = orig_minhist


def _run_all():
    import types
    mp = types.SimpleNamespace(setattr=lambda *a, **k: None)
    funcs = [
        lambda: test_trailing_stop_locks_profit(mp),
        lambda: test_hard_stop_loss(mp),
        lambda: test_max_concurrent_cap(mp),
        test_stable_ratio_blocks_entry,
    ]
    names = ["trailing_stop", "hard_stop", "max_concurrent", "stable_ratio_block"]
    failed = 0
    for name, fn in zip(names, funcs):
        try:
            fn()
            print(f"PASS {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            import traceback
            print(f"FAIL {name}: {exc}")
            traceback.print_exc()
    print(f"\n{len(funcs) - failed}/{len(funcs)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
