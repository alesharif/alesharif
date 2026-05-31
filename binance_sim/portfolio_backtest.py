#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Historical, multi-symbol portfolio backtest for the paper_bot_v2 strategy.

It replays the exact entry signal and ATR-trailing exit logic of the live
bot over historical 4h candles for the whole (current) USDT universe, with
realistic fees and the 8-concurrent-position / 12.5%-sizing portfolio model.

Run:
    python -m binance_sim.portfolio_backtest --start 2023-01-01 --end 2024-01-01
    python -m binance_sim.portfolio_backtest --start 2024-01-01 --max-symbols 80

Important caveats (read the README "Backtest fidelity" section):
  * Universe = symbols listed *today*  ->  survivorship bias (optimistic).
  * Exits are modelled on 4h OHLC, not the live 30s price feed (approximate).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import paper_bot_strategy as S

DATA_API = "https://data-api.binance.vision/api/v3"
FOUR_H_MS = 4 * 60 * 60 * 1000
WARMUP_BARS = 450               # candles fetched before --start so features are valid
STABLE_PAIRS = ["USDCUSDT", "FDUSDUSDT", "TUSDUSDT", "DAIUSDT"]
CACHE_DIR = os.environ.get("BINANCE_SIM_CACHE", "data/cache")


# ════════════════════════════ data fetching ═════════════════════════
def _http_get(path: str, params: dict, retries: int = 3):
    url = DATA_API + path + "?" + urllib.parse.urlencode(params)
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "binance-sim/0.1"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode())
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"GET {url} failed: {last}")


def _cache_path(symbol: str, start_ms: int, end_ms: int) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{symbol}_4h_{start_ms}_{end_ms}.csv")


def fetch_klines_range(symbol: str, start_ms: int, end_ms: int,
                       use_cache: bool = True) -> Optional[pd.DataFrame]:
    """Fetch all 4h candles in [start_ms, end_ms], paginating. Cached to disk."""
    path = _cache_path(symbol, start_ms, end_ms)
    if use_cache and os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:  # noqa: BLE001
            pass

    rows: List[list] = []
    cursor = start_ms
    while cursor < end_ms:
        data = _http_get("/klines", {
            "symbol": symbol, "interval": "4h",
            "startTime": cursor, "endTime": end_ms, "limit": 1000,
        })
        if not data:
            break
        rows.extend(data)
        if len(data) < 1000:
            break
        cursor = int(data[-1][6]) + 1
        time.sleep(0.05)

    if not rows:
        return None
    df = pd.DataFrame(rows, columns=[
        "time", "open", "high", "low", "close", "volume",
        "close_time", "quote_av", "trades", "tb_base", "tb_quote", "ignore"])
    df = df.drop_duplicates("time").reset_index(drop=True)
    for col in ["open", "high", "low", "close", "volume", "quote_av"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["time"] = df["time"].astype("int64")
    df["close_time"] = df["close_time"].astype("int64")
    df = df[["time", "open", "high", "low", "close", "volume", "close_time", "quote_av"]]
    if use_cache:
        df.to_csv(path, index=False)
    return df


def get_universe(max_symbols: Optional[int] = None,
                 explicit: Optional[List[str]] = None) -> List[str]:
    """Active USDT spot symbols minus the strategy's exclusions.

    If ``max_symbols`` is set, keep the most liquid ones by 24h quote volume.
    """
    if explicit:
        return [s.upper() for s in explicit]
    info = _http_get("/exchangeInfo", {})
    syms = []
    for s in info.get("symbols", []):
        if s.get("quoteAsset") != "USDT":
            continue
        if s.get("status") != "TRADING" or not s.get("isSpotTradingAllowed", False):
            continue
        if S.should_exclude(s.get("baseAsset", "")):
            continue
        syms.append(s["symbol"])

    if max_symbols and len(syms) > max_symbols:
        tickers = _http_get("/ticker/24hr", {})
        vol = {t["symbol"]: float(t.get("quoteVolume", 0)) for t in tickers}
        syms.sort(key=lambda x: vol.get(x, 0.0), reverse=True)
        syms = syms[:max_symbols]
    return syms


# ════════════════════════════ trade record ══════════════════════════
@dataclass
class ClosedTrade:
    symbol: str
    entry_time: int
    exit_time: int
    entry_price: float
    exit_price: float
    net_pct: float
    pnl: float
    outcome: str  # SL / TRAIL / MAXHOLD / EOD


@dataclass
class Position:
    symbol: str
    entry_time: int
    entry_price: float
    entry_atr: float
    pos_usd: float
    peak: float
    sl_price: float
    trailing_on: bool = False
    bars_held: int = 0


# ════════════════════════════ stable-ratio series ═══════════════════
def build_stable_ratio(start_ms: int, end_ms: int) -> Dict[int, float]:
    """close_time -> (current stable volume / trailing 30-bar avg)."""
    vol_by_time: Dict[int, float] = {}
    for sym in STABLE_PAIRS:
        df = fetch_klines_range(sym, start_ms, end_ms)
        if df is None:
            continue
        for ct, qv in zip(df["close_time"], df["quote_av"]):
            vol_by_time[int(ct)] = vol_by_time.get(int(ct), 0.0) + float(qv)
    if len(vol_by_time) < 32:
        return {}
    items = sorted(vol_by_time.items())
    times = [t for t, _ in items]
    vols = np.array([v for _, v in items], dtype=float)
    avg = pd.Series(vols).rolling(30).mean().shift(1).to_numpy()
    ratio = {}
    for i, t in enumerate(times):
        if np.isfinite(avg[i]) and avg[i] > 0:
            ratio[t] = vols[i] / avg[i]
    return ratio


# ════════════════════════════ the simulation ════════════════════════
@dataclass
class BacktestReport:
    start: str
    end: str
    n_symbols: int
    initial_capital: float
    final_capital: float
    equity_times: List[int] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)   # mark-to-market
    trades: List[ClosedTrade] = field(default_factory=list)
    btc_buy_hold: Optional[float] = None
    avg_exposure: float = 0.0
    threshold_cross: Optional[int] = None  # first time capital hit vol_threshold

    @property
    def total_return(self) -> float:
        return self.final_capital / self.initial_capital - 1.0

    @property
    def max_drawdown(self) -> float:
        peak = -1e18
        mdd = 0.0
        for v in self.equity_curve:
            peak = max(peak, v)
            if peak > 0:
                mdd = max(mdd, (peak - v) / peak)
        return mdd

    def summary(self) -> str:
        closed = self.trades
        n = len(closed)
        wins = [t for t in closed if t.net_pct > 0]
        losses = [t for t in closed if t.net_pct <= 0]
        gw = sum(t.net_pct for t in wins)
        gl = -sum(t.net_pct for t in losses)
        pf = (gw / gl) if gl > 0 else float("inf")
        wr = (len(wins) / n * 100) if n else 0.0
        avg_w = (sum(t.net_pct for t in wins) / len(wins)) if wins else 0.0
        avg_l = (sum(t.net_pct for t in losses) / len(losses)) if losses else 0.0
        by_outcome: Dict[str, int] = {}
        for t in closed:
            by_outcome[t.outcome] = by_outcome.get(t.outcome, 0) + 1

        lines = [
            "═" * 56,
            "  PAPER_BOT_V2 — Historical Backtest",
            "═" * 56,
            f"Period            : {self.start}  →  {self.end}",
            f"Universe          : {self.n_symbols} USDT symbols",
            f"Initial capital   : ${self.initial_capital:,.2f}",
            f"Final capital     : ${self.final_capital:,.2f}",
            f"Total return      : {self.total_return * 100:+.2f}%",
        ]
        if self.btc_buy_hold is not None:
            lines.append(f"BTC buy & hold    : {self.btc_buy_hold * 100:+.2f}%")
        lines += [
            f"Max drawdown      : {self.max_drawdown * 100:.2f}%",
            f"Avg open positions: {self.avg_exposure:.2f} / {S.MAX_CONCURRENT}",
        ]
        if self.threshold_cross is not None:
            lines.append(f"Vol-sizing kicked : {_fmt(self.threshold_cross)} (capital hit threshold)")
        lines += [
            "-" * 56,
            f"Closed trades     : {n}",
            f"Win rate          : {wr:.1f}%  ({len(wins)}W / {len(losses)}L)",
            f"Profit factor     : {pf:.2f}",
            f"Avg win / loss    : {avg_w:+.2f}% / {avg_l:+.2f}%",
            f"Exit breakdown    : {by_outcome}",
            "═" * 56,
        ]
        return "\n".join(lines)


def run(symbols: List[str], start_ms: int, end_ms: int,
        rank: str = "vol", max_hold: int = 0,
        slippage: float = 0.0, fixed_notional: bool = False,
        exit_model: str = "optimistic",
        vol_sizing: bool = False, vol_threshold: float = 30_000.0,
        vol_size_pct: float = 0.001, source: str = "api",
        capital_cap: Optional[float] = None,
        log=print) -> BacktestReport:
    """Replay the strategy over history.

    slippage       : extra round-trip cost (%) added on top of fees, to model
                     imperfect fills (tight ATR stops on alts slip).
    fixed_notional : size every position at INITIAL_CAPITAL*POSITION_PCT
                     instead of 12.5% of *current* capital. Removes the
                     compounding explosion and isolates the raw per-trade edge.
    exit_model     : 'optimistic' raises the trailing stop with the bar high
                     before testing the low (best case). 'pessimistic' tests
                     the low against the pre-bar stop first (worst case). The
                     truth lies between; reality needs the live 30s feed.
    vol_sizing     : once capital >= vol_threshold, size each entry at
                     vol_size_pct of the coin's daily quote volume, capped at
                     portfolio/8 (a liquidity-aware sizing that curbs the
                     compounding explosion on thin coins).
    """
    cost_rt = S.FEE_RT + slippage
    base_notional = S.INITIAL_CAPITAL * S.POSITION_PCT
    fetch_from = start_ms - WARMUP_BARS * FOUR_H_MS

    # 1) load + featurise every symbol -------------------------------
    per_symbol: Dict[str, pd.DataFrame] = {}
    if source == "archive":
        # survivorship-free: candles for delisted coins included for their life
        from . import pit_universe as PIT
        raw = PIT.prefetch_universe(symbols, fetch_from, end_ms, log=log)
        log(f"Featurising {len(raw)} symbols...")
        for sym, df in raw.items():
            df = S.compute_features(df)
            df = S.attach_entry_signal(df)
            per_symbol[sym] = df.set_index("close_time")
    else:
        log(f"Loading {len(symbols)} symbols...")
        for i, sym in enumerate(symbols, 1):
            df = fetch_klines_range(sym, fetch_from, end_ms)
            if df is None or len(df) < S.MIN_HISTORY:
                continue
            df = S.compute_features(df)
            df = S.attach_entry_signal(df)
            per_symbol[sym] = df.set_index("close_time")
            if i % 25 == 0:
                log(f"  loaded {i}/{len(symbols)}  (usable: {len(per_symbol)})")
    log(f"Usable symbols: {len(per_symbol)}")
    if not per_symbol:
        raise RuntimeError("No usable symbol data — check connectivity / date range.")

    # last candle time per symbol — used to force-close on delisting
    last_seen: Dict[str, int] = {s: int(df.index.max()) for s, df in per_symbol.items()}

    # 2) master 4h timeline within [start, end] ----------------------
    times = set()
    for df in per_symbol.values():
        times.update(int(t) for t in df.index if start_ms <= t <= end_ms)
    timeline = sorted(times)
    log(f"Timeline: {len(timeline)} 4h bars")

    stable_ratio = build_stable_ratio(fetch_from, end_ms)

    # 3) walk forward ------------------------------------------------
    capital = S.INITIAL_CAPITAL
    positions: Dict[str, Position] = {}
    report = BacktestReport(
        start=_fmt(start_ms), end=_fmt(end_ms),
        n_symbols=len(per_symbol), initial_capital=capital, final_capital=capital,
    )
    exposure_sum = 0

    for t in timeline:
        # --- EXIT pass (positions opened on prior bars) -------------
        for sym in list(positions.keys()):
            pos = positions[sym]
            if pos.entry_time >= t:
                continue  # opened this bar; no same-bar exit
            df = per_symbol[sym]
            if t not in df.index:
                # past this symbol's last candle => delisted while held: close it
                if t > last_seen[sym]:
                    last_px = float(df.iloc[-1]["close"])
                    net = (last_px / pos.entry_price - 1) * 100 - cost_rt
                    pnl = pos.pos_usd * net / 100.0
                    capital += pnl
                    report.trades.append(ClosedTrade(
                        symbol=sym, entry_time=pos.entry_time, exit_time=last_seen[sym],
                        entry_price=pos.entry_price, exit_price=last_px,
                        net_pct=net, pnl=pnl, outcome="DELIST"))
                    del positions[sym]
                continue  # otherwise a transient gap; skip this bar
            row = df.loc[t]
            high = float(row["high"]); low = float(row["low"]); close = float(row["close"])
            atr = pos.entry_atr
            pos.bars_held += 1

            exit_price = None
            outcome = None

            def _raise_trailing():
                if high > pos.peak:
                    pos.peak = high
                    if pos.peak - pos.entry_price >= S.ACTIVATE_ATR * atr:
                        pos.trailing_on = True
                        new_sl = pos.peak - S.TRAIL_ATR * atr
                        if new_sl > pos.sl_price:
                            pos.sl_price = new_sl

            if exit_model == "pessimistic":
                # worst case: the dip hits the pre-bar stop before any rally
                if low <= pos.sl_price:
                    exit_price = pos.sl_price
                    outcome = "TRAIL" if pos.trailing_on else "SL"
                else:
                    _raise_trailing()
            else:
                # optimistic: rally lifts the stop, then test the dip
                _raise_trailing()
                if low <= pos.sl_price:
                    exit_price = pos.sl_price
                    outcome = "TRAIL" if pos.trailing_on else "SL"

            if exit_price is None and max_hold and pos.bars_held >= max_hold:
                exit_price = close
                outcome = "MAXHOLD"

            if exit_price is not None:
                net = (exit_price / pos.entry_price - 1) * 100 - cost_rt
                pnl = pos.pos_usd * net / 100.0
                capital += pnl
                report.trades.append(ClosedTrade(
                    symbol=sym, entry_time=pos.entry_time, exit_time=int(t),
                    entry_price=pos.entry_price, exit_price=exit_price,
                    net_pct=net, pnl=pnl, outcome=outcome))
                del positions[sym]

        # --- ENTRY pass --------------------------------------------
        sr = stable_ratio.get(t, float("nan"))
        stable_block = np.isfinite(sr) and sr > S.STABLE_RATIO_MAX
        if not stable_block and len(positions) < S.MAX_CONCURRENT:
            cands = []
            for sym, df in per_symbol.items():
                if sym in positions or t not in df.index:
                    continue
                row = df.loc[t]
                if bool(row["entry_signal"]):
                    cands.append((sym, float(row["close"]), float(row["atr"]),
                                  float(row["vol_pit"])))
            if rank == "vol":
                cands.sort(key=lambda x: x[3], reverse=True)
            else:
                cands.sort(key=lambda x: x[0])
            slots = S.MAX_CONCURRENT - len(positions)
            for sym, price, atr, _vol in cands[:slots]:
                # cap the capital used for sizing (profits still accumulate,
                # but the largest position is frozen at capital_cap/8)
                sizing_cap = capital if capital_cap is None else min(capital, capital_cap)
                cap_eighth = base_notional if fixed_notional else sizing_cap * S.POSITION_PCT
                if vol_sizing and capital >= vol_threshold and np.isfinite(_vol):
                    # 0.1% of the coin's daily quote volume, capped at portfolio/8
                    pos_usd = min(vol_size_pct * _vol, cap_eighth)
                else:
                    pos_usd = cap_eighth
                positions[sym] = Position(
                    symbol=sym, entry_time=int(t), entry_price=price, entry_atr=atr,
                    pos_usd=pos_usd, peak=price, sl_price=price - S.SL_ATR * atr)

        # --- mark-to-market equity ---------------------------------
        mtm = capital
        for sym, pos in positions.items():
            df = per_symbol[sym]
            if t in df.index:
                px = float(df.loc[t]["close"])
                mtm += pos.pos_usd * ((px / pos.entry_price - 1) - cost_rt / 100.0)
        report.equity_times.append(int(t))
        report.equity_curve.append(mtm)
        exposure_sum += len(positions)
        if report.threshold_cross is None and capital >= vol_threshold:
            report.threshold_cross = int(t)

    # close anything still open at the final price (EOD)
    last_t = timeline[-1]
    for sym, pos in list(positions.items()):
        df = per_symbol[sym]
        if last_t in df.index:
            px = float(df.loc[last_t]["close"])
            net = (px / pos.entry_price - 1) * 100 - cost_rt
            pnl = pos.pos_usd * net / 100.0
            capital += pnl
            report.trades.append(ClosedTrade(
                symbol=sym, entry_time=pos.entry_time, exit_time=int(last_t),
                entry_price=pos.entry_price, exit_price=px,
                net_pct=net, pnl=pnl, outcome="EOD"))

    report.final_capital = capital
    report.avg_exposure = exposure_sum / max(1, len(timeline))

    # BTC buy & hold over the same window, for reference
    btc = per_symbol.get("BTCUSDT")
    if btc is not None:
        window = btc[(btc.index >= start_ms) & (btc.index <= end_ms)]
        if len(window) > 1:
            report.btc_buy_hold = float(window["close"].iloc[-1] / window["close"].iloc[0] - 1)

    return report


def _fmt(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def _parse_date(s: str) -> int:
    dt = datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


# ════════════════════════════ CLI ═══════════════════════════════════
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Historical backtest of the paper_bot_v2 strategy.")
    p.add_argument("--start", required=True, help="Backtest start date (YYYY-MM-DD)")
    p.add_argument("--end", default=None, help="Backtest end date (YYYY-MM-DD, default today)")
    p.add_argument("--max-symbols", type=int, default=None,
                   help="Limit universe to N most-liquid symbols (faster)")
    p.add_argument("--symbols", default=None,
                   help="Comma-separated explicit symbol list (overrides universe)")
    p.add_argument("--rank", choices=["vol", "symbol"], default="vol",
                   help="How to pick among more candidates than free slots")
    p.add_argument("--max-hold", type=int, default=0,
                   help="Close after N bars (0 = disabled, matches live bot)")
    p.add_argument("--slippage", type=float, default=0.0,
                   help="Extra round-trip cost %% on top of fees (realism)")
    p.add_argument("--fixed-notional", action="store_true",
                   help="Size each trade at a fixed $ (no compounding) to show the raw edge")
    p.add_argument("--exit-model", choices=["optimistic", "pessimistic"], default="optimistic",
                   help="Trailing-fill assumption within a 4h bar")
    p.add_argument("--vol-sizing", action="store_true",
                   help="Switch to volume-based sizing once capital >= --vol-threshold")
    p.add_argument("--vol-threshold", type=float, default=30_000.0,
                   help="Capital level ($) at which volume sizing activates")
    p.add_argument("--vol-size-pct", type=float, default=0.001,
                   help="Fraction of a coin's daily volume per trade (0.001 = 0.1%%)")
    p.add_argument("--source", choices=["api", "archive"], default="api",
                   help="'api' = live-listed symbols (survivorship-biased); "
                        "'archive' = point-in-time universe incl. delisted coins")
    p.add_argument("--capital-cap", type=float, default=None,
                   help="Cap the capital used for sizing ($); profits still accrue "
                        "but the largest position is frozen at cap/8 (e.g. 1000000 -> $125k)")
    p.add_argument("--out", default=None, help="Directory to write trades.csv / equity.csv")
    args = p.parse_args(argv)

    start_ms = _parse_date(args.start)
    end_ms = _parse_date(args.end) if args.end else int(time.time() * 1000)
    explicit = [s.strip() for s in args.symbols.split(",")] if args.symbols else None

    print("Resolving universe...")
    if explicit:
        symbols = explicit
    elif args.source == "archive":
        from . import pit_universe as PIT
        symbols = PIT.list_all_usdt_symbols()
        if args.max_symbols:
            symbols = symbols[:args.max_symbols]
    else:
        symbols = get_universe(max_symbols=args.max_symbols, explicit=explicit)
    print(f"Universe: {len(symbols)} symbols  (source={args.source})")

    report = run(symbols, start_ms, end_ms, rank=args.rank, max_hold=args.max_hold,
                 slippage=args.slippage, fixed_notional=args.fixed_notional,
                 exit_model=args.exit_model, vol_sizing=args.vol_sizing,
                 vol_threshold=args.vol_threshold, vol_size_pct=args.vol_size_pct,
                 source=args.source, capital_cap=args.capital_cap)
    print("\n" + report.summary())

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        pd.DataFrame([t.__dict__ for t in report.trades]).to_csv(
            os.path.join(args.out, "trades.csv"), index=False)
        pd.DataFrame({"close_time": report.equity_times,
                      "equity": report.equity_curve}).to_csv(
            os.path.join(args.out, "equity.csv"), index=False)
        print(f"\nWrote trades.csv and equity.csv to {args.out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
