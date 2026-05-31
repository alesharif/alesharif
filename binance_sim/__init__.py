"""Binance Strategy Simulator.

A lightweight framework to backtest and paper-trade strategies on the
Binance Spot market using only the public market-data API.
"""

__version__ = "0.1.0"

from .strategy import Strategy, Signal
from .client import BinanceClient, Candle

__all__ = ["Strategy", "Signal", "BinanceClient", "Candle", "__version__"]
