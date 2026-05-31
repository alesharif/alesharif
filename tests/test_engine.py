"""Offline tests for indicators, portfolio and the backtest engine.

These run without any network access by synthesising candles.
Run with:  python -m pytest -q   (or)   python tests/test_engine.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from binance_sim.client import Candle  # noqa: E402
from binance_sim.indicators import crossover, ema, rsi, sma  # noqa: E402
from binance_sim.portfolio import Portfolio  # noqa: E402
from binance_sim.backtest import run_backtest  # noqa: E402
from binance_sim.strategies import get_strategy  # noqa: E402
from binance_sim.strategy import Signal, Strategy  # noqa: E402


def make_candles(prices):
    candles = []
    t = 0
    for p in prices:
        candles.append(Candle(t, p, p, p, p, 1.0, t + 60_000))
        t += 60_000
    return candles


def test_sma():
    assert sma([1, 2, 3, 4], 2) == [None, 1.5, 2.5, 3.5]


def test_ema_basic():
    out = ema([1, 2, 3, 4, 5], 3)
    assert out[0] is None and out[1] is None
    assert abs(out[2] - 2.0) < 1e-9  # seed = SMA(1,2,3)
    assert out[3] > out[2]  # rising series


def test_rsi_extremes():
    rising = list(range(1, 30))
    out = rsi([float(x) for x in rising], 14)
    assert out[-1] == 100.0  # only gains -> RSI 100


def test_crossover():
    a = [1, 2, 3]
    b = [3, 2, 1]
    # at i=1 fast(2) only reaches slow(2): not yet a strict cross
    assert crossover(a, b, 1) is False
    # at i=2 fast(3) > slow(1) after being below: cross completed
    assert crossover(a, b, 2) is True


def test_portfolio_roundtrip_pnl():
    pf = Portfolio(cash=1000.0, fee_rate=0.0)
    pf.buy(100.0, 0)
    assert pf.in_position
    assert abs(pf.base_qty - 10.0) < 1e-9
    pf.sell(110.0, 1)
    assert not pf.in_position
    assert abs(pf.cash - 1100.0) < 1e-6  # +10%
    sell = pf.trades[-1]
    assert sell.pnl is not None and sell.pnl > 0


def test_portfolio_fees_reduce_equity():
    pf = Portfolio(cash=1000.0, fee_rate=0.01)
    pf.buy(100.0, 0)
    pf.sell(100.0, 1)  # flat price, fees on both sides -> a loss
    assert pf.cash < 1000.0


def test_portfolio_serialization():
    pf = Portfolio(cash=500.0, fee_rate=0.001)
    pf.buy(50.0, 0)
    restored = Portfolio.from_dict(pf.to_dict())
    assert restored.cash == pf.cash
    assert restored.base_qty == pf.base_qty
    assert len(restored.trades) == 1


class _AlwaysBuy(Strategy):
    name = "always_buy"

    @property
    def warmup(self):
        return 1

    def generate_signal(self, candles):
        return Signal.BUY


def test_backtest_tracks_buy_hold():
    prices = [100, 110, 120, 130, 140]
    candles = make_candles(prices)
    res = run_backtest(candles, _AlwaysBuy(), initial_cash=1000.0, fee_rate=0.0)
    # buy at first candle, hold to the end -> matches buy & hold (no fees)
    assert abs(res.total_return - res.buy_hold_return) < 1e-6
    assert res.buy_hold_return > 0


def test_ma_crossover_produces_trades():
    # craft a series that dips then trends up to force a crossover
    prices = [100 - i for i in range(40)] + [60 + i * 2 for i in range(40)]
    candles = make_candles([float(p) for p in prices])
    strat = get_strategy("ma_crossover", {"fast": 5, "slow": 20})
    res = run_backtest(candles, strat, initial_cash=1000.0, fee_rate=0.001)
    assert res.num_trades >= 1
    assert len(res.equity_curve) == len(candles)


def test_registry_lookup():
    assert get_strategy("rsi").name == "rsi"
    try:
        get_strategy("does_not_exist")
    except KeyError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected KeyError for unknown strategy")


def _run_all():
    funcs = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in funcs:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
    print(f"\n{len(funcs) - failed}/{len(funcs)} tests passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
