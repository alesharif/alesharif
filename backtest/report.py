#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
report.py — يحسب إحصائيات الأداء من نتيجة المحاكاة ويكتب المخرجات:
  trades.csv ، equity_curve.csv ، equity.png ، وملخص نصّي.
"""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_DIR = Path(__file__).resolve().parent / 'output'


def _stats(trades):
    if not trades:
        return None
    pnls = np.array([t.get('net_pnl', 0.0) for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    gross_profit = float(wins.sum())
    gross_loss = float(-losses.sum())
    pf = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')
    return {
        'count': len(trades),
        'wins': int((pnls > 0).sum()),
        'losses': int((pnls <= 0).sum()),
        'wr': float((pnls > 0).mean() * 100),
        'net_pnl': float(pnls.sum()),
        'gross_profit': gross_profit,
        'gross_loss': gross_loss,
        'pf': pf,
        'avg_win': float(wins.mean()) if len(wins) else 0.0,
        'avg_loss': float(losses.mean()) if len(losses) else 0.0,
        'best': float(pnls.max()),
        'worst': float(pnls.min()),
        'avg_pnl_pct': float(np.mean([t.get('pnl_pct', 0) for t in trades])),
    }


def _max_drawdown(equity_log):
    if not equity_log:
        return 0.0, 0.0
    eq = np.array([e for _, e in equity_log], dtype=float)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak)
    dd_pct = dd / peak * 100
    return float(dd.min()), float(dd_pct.min())


def _equity_curve_df(equity_log):
    rows = [{'date': datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime('%Y-%m-%d'),
             'ts_ms': ts, 'equity': eq} for ts, eq in equity_log]
    return pd.DataFrame(rows)


def _trades_df(trades):
    rows = []
    for t in trades:
        sig = t.get('signals', {})
        inds = '+'.join([k for k, v in [('MACD', sig.get('bullish_macd')),
                                        ('RSI', sig.get('bullish_rsi')),
                                        ('OBV', sig.get('bullish_obv'))] if v]) or '-'
        rows.append({
            'symbol': t.get('symbol'),
            'entry_utc': datetime.fromtimestamp(t.get('entry_time', 0), tz=timezone.utc).strftime('%Y-%m-%d %H:%M'),
            'exit_utc': datetime.fromtimestamp(t.get('exit_time', 0), tz=timezone.utc).strftime('%Y-%m-%d %H:%M'),
            'entry_price': t.get('entry_price'),
            'exit_price': t.get('exit_price'),
            'size_usd': round(t.get('size_usd', 0), 2),
            'qty': t.get('quantity'),
            'duration_h': round(t.get('duration_hours', 0), 2),
            'exit_reason': t.get('exit_reason'),
            'signals': inds,
            'ad_class': sig.get('ad_classification', ''),
            'peak_pct': round(t.get('peak_pct', 0), 2),
            'net_pnl': round(t.get('net_pnl', 0), 4),
            'pnl_pct': round(t.get('pnl_pct', 0), 3),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values('entry_utc').reset_index(drop=True)
    return df


def _monthly_breakdown(trades):
    buckets = {}
    for t in trades:
        dt = datetime.fromtimestamp(t.get('exit_time', 0), tz=timezone.utc)
        key = dt.strftime('%Y-%m')
        b = buckets.setdefault(key, {'n': 0, 'pnl': 0.0, 'w': 0})
        b['n'] += 1
        b['pnl'] += t.get('net_pnl', 0)
        if t.get('net_pnl', 0) > 0:
            b['w'] += 1
    return dict(sorted(buckets.items()))


def _save_plot(equity_df, path, start_capital):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 5))
        x = pd.to_datetime(equity_df['date'])
        ax.plot(x, equity_df['equity'], color='#1565c0', lw=1.6)
        ax.axhline(start_capital, color='#999', ls='--', lw=1, label=f'Start ${start_capital:,.0f}')
        ax.fill_between(x, start_capital, equity_df['equity'],
                        where=equity_df['equity'] >= start_capital, color='#4caf50', alpha=0.15)
        ax.fill_between(x, start_capital, equity_df['equity'],
                        where=equity_df['equity'] < start_capital, color='#f44336', alpha=0.15)
        ax.set_title('Backtest Equity Curve — Real Bot v3.27 / Binance 2025')
        ax.set_ylabel('Equity (USDT)')
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
        return True
    except Exception as e:
        print(f"plot failed: {e}")
        return False


def build_report(state, equity_log, start_capital, year=2025, platform='binance',
                 out_dir=OUTPUT_DIR):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ps = state[platform]
    trades = ps['history']
    # نفصل صفقات الاسترجاع/القسري إن وُجدت
    real_trades = [t for t in trades if t.get('exit_reason') != 'ORPHAN_RECOVERED']
    st = _stats(real_trades)
    end_equity = equity_log[-1][1] if equity_log else start_capital
    ret_pct = (end_equity - start_capital) / start_capital * 100
    dd_abs, dd_pct = _max_drawdown(equity_log)

    # المخرجات الملفية
    tdf = _trades_df(real_trades)
    edf = _equity_curve_df(equity_log)
    tdf.to_csv(out_dir / f'trades_{year}.csv', index=False)
    edf.to_csv(out_dir / f'equity_curve_{year}.csv', index=False)
    _save_plot(edf, out_dir / f'equity_{year}.png', start_capital)

    # exit reasons
    reasons = Counter(t.get('exit_reason') for t in real_trades)
    reason_pnl = {}
    for t in real_trades:
        r = t.get('exit_reason')
        reason_pnl[r] = reason_pnl.get(r, 0) + t.get('net_pnl', 0)

    monthly = _monthly_breakdown(real_trades)

    # risk (per-trade Sharpe/Sortino مثل البوت)
    pnls = [t.get('net_pnl', 0) for t in real_trades]
    sharpe = sortino = 0.0
    if len(pnls) >= 2:
        avg = np.mean(pnls)
        std = np.std(pnls)
        sharpe = avg / std if std > 0 else 0
        neg = [x for x in pnls if x < 0]
        if neg:
            nstd = np.sqrt(np.mean(np.square(neg)))
            sortino = avg / nstd if nstd > 0 else 0

    lines = []
    A = lines.append
    A("═" * 60)
    A(f"  محاكاة تاريخية — Real Bot v3.27 — Binance {year}")
    A("═" * 60)
    A(f"رأس المال الابتدائي : ${start_capital:,.2f}")
    A(f"رأس المال النهائي   : ${end_equity:,.2f}")
    A(f"العائد الكلي        : {ret_pct:+.2f}%")
    A(f"السيولة النهائية    : ${ps['liquid_capital']:,.2f}")
    A(f"صفقات مفتوحة متبقية : {len(ps['open_positions'])}")
    A("-" * 60)
    if st:
        A(f"عدد الصفقات         : {st['count']}")
        A(f"رابحة / خاسرة       : {st['wins']} / {st['losses']}")
        A(f"Win Rate            : {st['wr']:.2f}%")
        A(f"Profit Factor       : {st['pf']:.2f}")
        A(f"صافي الربح          : ${st['net_pnl']:+,.2f}")
        A(f"إجمالي الربح/الخسارة: +${st['gross_profit']:,.2f} / -${st['gross_loss']:,.2f}")
        A(f"متوسط الرابحة/الخاسرة: ${st['avg_win']:+.2f} / ${st['avg_loss']:+.2f}")
        A(f"أفضل / أسوأ صفقة    : ${st['best']:+.2f} / ${st['worst']:+.2f}")
        A(f"متوسط % للصفقة      : {st['avg_pnl_pct']:+.2f}%")
    A(f"أكبر تراجع (DD)     : ${dd_abs:,.2f} ({dd_pct:.2f}%)")
    A(f"Sharpe / Sortino    : {sharpe:.2f} / {sortino:.2f}")
    A("-" * 60)
    A("أسباب الإغلاق:")
    for r, c in reasons.most_common():
        A(f"  {r:16s}: {c:4d} ({c/max(len(real_trades),1)*100:5.1f}%)  PnL ${reason_pnl[r]:+,.2f}")
    A("-" * 60)
    A("الأداء الشهري (حسب وقت الإغلاق):")
    for k, b in monthly.items():
        wr = b['w'] / b['n'] * 100 if b['n'] else 0
        A(f"  {k}: {b['n']:4d} صفقة | WR {wr:5.1f}% | PnL ${b['pnl']:+,.2f}")
    A("═" * 60)
    A(f"المخرجات: {out_dir}/trades_{year}.csv , equity_curve_{year}.csv , equity_{year}.png")

    summary = "\n".join(lines)
    (out_dir / f'summary_{year}.txt').write_text(summary, encoding='utf-8')
    return summary
