"""Strategy registry.

Register new strategies here so the CLI can discover them by name.
"""

from __future__ import annotations

from typing import Dict, Type

from ..strategy import Strategy
from .ma_crossover import MaCrossover
from .rsi import RsiStrategy

# name -> strategy class
REGISTRY: Dict[str, Type[Strategy]] = {
    MaCrossover.name: MaCrossover,
    RsiStrategy.name: RsiStrategy,
}


def get_strategy(name: str, params: dict | None = None) -> Strategy:
    """Instantiate a registered strategy by name with optional params."""
    if name not in REGISTRY:
        available = ", ".join(sorted(REGISTRY))
        raise KeyError(f"Unknown strategy {name!r}. Available: {available}")
    return REGISTRY[name](**(params or {}))


__all__ = ["REGISTRY", "get_strategy", "MaCrossover", "RsiStrategy"]
