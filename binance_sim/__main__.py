"""Command-line interface for the Binance Strategy Simulator.

Examples
--------
List strategies:
    python -m binance_sim list

Backtest:
    python -m binance_sim backtest --symbol BTCUSDT --interval 1h \
        --limit 1000 --strategy ma_crossover --params '{"fast":20,"slow":50}'

Paper trade (live, simulated money):
    python -m binance_sim paper --symbol BTCUSDT --interval 1m \
        --strategy rsi --state state.json
"""

from __future__ import annotations

import argparse
import json
import sys

from .backtest import run_backtest
from .client import BinanceClient, BinanceError
from .paper import PaperTrader
from .portfolio import Portfolio
from .strategies import REGISTRY, get_strategy


def _parse_params(raw: str | None) -> dict:
    if not raw:
        return {}
    try:
        params = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--params must be valid JSON: {exc}")
    if not isinstance(params, dict):
        raise SystemExit("--params must be a JSON object, e.g. '{\"fast\":20}'")
    return params


def cmd_list(_: argparse.Namespace) -> int:
    print("Available strategies:")
    for name, cls in sorted(REGISTRY.items()):
        try:
            doc = (cls.__doc__ or "").strip().splitlines()[0]
        except IndexError:
            doc = ""
        print(f"  - {name:<14} {doc}")
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    client = BinanceClient(base_url=args.base_url)
    strategy = get_strategy(args.strategy, _parse_params(args.params))
    try:
        candles = client.get_klines(args.symbol, args.interval, limit=args.limit)
    except BinanceError as exc:
        print(f"Failed to fetch data: {exc}", file=sys.stderr)
        return 2
    if len(candles) < strategy.warmup:
        print(
            f"Not enough candles ({len(candles)}) for warmup "
            f"({strategy.warmup}). Increase --limit.",
            file=sys.stderr,
        )
        return 2

    result = run_backtest(
        candles,
        strategy,
        initial_cash=args.cash,
        fee_rate=args.fee,
        position_fraction=args.position,
        symbol=args.symbol.upper(),
        interval=args.interval,
    )
    print(result.summary())
    if args.trades:
        print("\nTrades:")
        for t in result.trades:
            pnl = f" pnl={t.pnl:+.2f}" if t.pnl is not None else ""
            print(f"  {t.side:<4} qty={t.qty:.6f} @ {t.price:.2f} fee={t.fee:.2f}{pnl}")
    return 0


def cmd_paper(args: argparse.Namespace) -> int:
    client = BinanceClient(base_url=args.base_url)
    strategy = get_strategy(args.strategy, _parse_params(args.params))
    portfolio = Portfolio(
        cash=args.cash, fee_rate=args.fee, position_fraction=args.position
    )
    trader = PaperTrader(
        client=client,
        symbol=args.symbol,
        interval=args.interval,
        strategy=strategy,
        portfolio=portfolio,
        state_file=args.state,
        history=args.history,
    )
    try:
        trader.run(poll_seconds=args.poll, max_iterations=args.max_iterations)
    except BinanceError as exc:
        print(f"Binance error: {exc}", file=sys.stderr)
        return 2
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="binance_sim",
        description="Backtest and paper-trade strategies on Binance Spot.",
    )
    parser.add_argument(
        "--base-url",
        default="https://api.binance.com",
        help="Binance REST base URL (use https://testnet.binance.vision for testnet)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # list
    p_list = sub.add_parser("list", help="List available strategies")
    p_list.set_defaults(func=cmd_list)

    # shared trading args
    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--symbol", default="BTCUSDT", help="Trading pair, e.g. BTCUSDT")
        p.add_argument("--interval", default="1h", help="Candle interval, e.g. 1m,1h,1d")
        p.add_argument("--strategy", required=True, help="Strategy name (see `list`)")
        p.add_argument("--params", default=None, help="Strategy params as JSON")
        p.add_argument("--cash", type=float, default=10_000.0, help="Starting cash")
        p.add_argument("--fee", type=float, default=0.001, help="Fee rate (0.001 = 0.1%%)")
        p.add_argument(
            "--position", type=float, default=1.0,
            help="Fraction of cash to deploy per entry (0-1)",
        )

    # backtest
    p_bt = sub.add_parser("backtest", help="Run a historical backtest")
    add_common(p_bt)
    p_bt.add_argument("--limit", type=int, default=1000, help="Number of candles")
    p_bt.add_argument("--trades", action="store_true", help="Print every trade")
    p_bt.set_defaults(func=cmd_backtest)

    # paper
    p_pp = sub.add_parser("paper", help="Run live paper trading (simulated money)")
    add_common(p_pp)
    p_pp.add_argument("--state", default=None, help="JSON file to persist/resume state")
    p_pp.add_argument("--history", type=int, default=500, help="Candles of context to load")
    p_pp.add_argument("--poll", type=float, default=None, help="Seconds between polls")
    p_pp.add_argument(
        "--max-iterations", type=int, default=None,
        help="Stop after N polls (for testing)",
    )
    p_pp.set_defaults(func=cmd_paper)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
