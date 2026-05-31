"""A minimal long-only Spot portfolio simulator.

Models a single base/quote pair (e.g. BTC/USDT). Holds quote currency
(cash) and, when in a position, a quantity of the base asset. Trading fees
are charged as a fraction of the traded notional value, matching how
Binance applies the taker fee.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Trade:
    side: str        # "BUY" or "SELL"
    time: int        # ms epoch
    price: float
    qty: float       # base-asset quantity
    fee: float       # fee paid in quote currency
    pnl: Optional[float] = None  # realised PnL in quote currency (on SELL)


@dataclass
class Portfolio:
    """Tracks cash, an optional long position, fees and trade history."""

    cash: float
    fee_rate: float = 0.001          # 0.10% default Binance taker fee
    position_fraction: float = 1.0   # fraction of cash to deploy per entry

    base_qty: float = 0.0
    entry_price: float = 0.0
    entry_cost: float = 0.0          # quote spent on the open position (incl. fee)
    trades: List[Trade] = field(default_factory=list)

    @property
    def in_position(self) -> bool:
        return self.base_qty > 0

    def equity(self, price: float) -> float:
        """Total account value marked at ``price``."""
        return self.cash + self.base_qty * price

    def buy(self, price: float, time: int) -> Optional[Trade]:
        """Open a long position using ``position_fraction`` of available cash."""
        if self.in_position or self.cash <= 0:
            return None
        spend = self.cash * self.position_fraction
        fee = spend * self.fee_rate
        qty = (spend - fee) / price
        if qty <= 0:
            return None
        self.cash -= spend
        self.base_qty = qty
        self.entry_price = price
        self.entry_cost = spend
        trade = Trade("BUY", time, price, qty, fee)
        self.trades.append(trade)
        return trade

    def sell(self, price: float, time: int) -> Optional[Trade]:
        """Close the current long position entirely."""
        if not self.in_position:
            return None
        proceeds = self.base_qty * price
        fee = proceeds * self.fee_rate
        net = proceeds - fee
        pnl = net - self.entry_cost
        qty = self.base_qty
        self.cash += net
        self.base_qty = 0.0
        self.entry_price = 0.0
        self.entry_cost = 0.0
        trade = Trade("SELL", time, price, qty, fee, pnl=pnl)
        self.trades.append(trade)
        return trade

    # -- serialization (used by the paper trader to resume state) ------
    def to_dict(self) -> dict:
        return {
            "cash": self.cash,
            "fee_rate": self.fee_rate,
            "position_fraction": self.position_fraction,
            "base_qty": self.base_qty,
            "entry_price": self.entry_price,
            "entry_cost": self.entry_cost,
            "trades": [t.__dict__ for t in self.trades],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Portfolio":
        pf = cls(
            cash=data["cash"],
            fee_rate=data.get("fee_rate", 0.001),
            position_fraction=data.get("position_fraction", 1.0),
        )
        pf.base_qty = data.get("base_qty", 0.0)
        pf.entry_price = data.get("entry_price", 0.0)
        pf.entry_cost = data.get("entry_cost", 0.0)
        pf.trades = [Trade(**t) for t in data.get("trades", [])]
        return pf
