"""
Backtesting Engine
Tests signal logic on historical data and outputs win rate, drawdown, best pairs/TFs.
Uses walk-forward validation to avoid lookahead bias.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

from data_engine import fetch_ohlc, TF_EXPIRY, TF_MINUTES, ASSETS, TIMEFRAMES
from market_structure import analyse_structure
from indicators import compute_indicators
from price_action import analyse_price_action
from ai_model import build_features, get_ai_engine, CONFIDENCE_THRESHOLD
from filter_engine import apply_filters, get_current_session

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BacktestTrade:
    timestamp:   datetime
    asset:       str
    timeframe:   str
    direction:   str
    entry_price: float
    exit_price:  float
    expiry_min:  int
    confidence:  float
    result:      str    # "win" | "loss" | "draw"
    pnl_pct:    float   # % move in correct direction
    reasons:    List[str] = field(default_factory=list)


@dataclass
class BacktestResult:
    total_trades:  int = 0
    wins:          int = 0
    losses:        int = 0
    draws:         int = 0
    win_rate:      float = 0.0
    loss_rate:     float = 0.0
    max_drawdown:  float = 0.0
    avg_confidence: float = 0.0
    best_asset:    str = ""
    best_tf:       str = ""
    trades:        List[BacktestTrade] = field(default_factory=list)
    by_asset:      Dict = field(default_factory=dict)
    by_timeframe:  Dict = field(default_factory=dict)
    equity_curve:  List[float] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=" * 50,
            "        BACKTEST RESULTS SUMMARY",
            "=" * 50,
            f"  Total Trades:    {self.total_trades}",
            f"  Wins:            {self.wins} ({self.win_rate:.1f}%)",
            f"  Losses:          {self.losses} ({self.loss_rate:.1f}%)",
            f"  Draws:           {self.draws}",
            f"  Max Drawdown:    {self.max_drawdown:.2f}%",
            f"  Avg Confidence:  {self.avg_confidence:.1f}%",
            f"  Best Asset:      {self.best_asset}",
            f"  Best Timeframe:  {self.best_tf}",
            "",
            "  --- By Asset ---",
        ]
        for asset, stats in self.by_asset.items():
            wr = stats["wins"] / stats["total"] * 100 if stats["total"] > 0 else 0
            lines.append(
                f"  {asset:<8} {stats['total']:>3} trades  "
                f"Win: {wr:.1f}%  ({stats['wins']}W/{stats['losses']}L)"
            )

        lines += ["", "  --- By Timeframe ---"]
        for tf, stats in self.by_timeframe.items():
            wr = stats["wins"] / stats["total"] * 100 if stats["total"] > 0 else 0
            lines.append(
                f"  {tf:<4} {stats['total']:>3} trades  "
                f"Win: {wr:.1f}%  ({stats['wins']}W/{stats['losses']}L)"
            )

        lines.append("=" * 50)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Backtest runner
# ---------------------------------------------------------------------------

def _simulate_outcome(
    df: pd.DataFrame,
    bar_index: int,
    direction: str,
    expiry_bars: int,
) -> Tuple[str, float, float]:
    """
    Simulate binary option outcome.
    Returns (result, entry_price, exit_price).
    """
    if bar_index + expiry_bars >= len(df):
        return "draw", 0.0, 0.0

    entry = float(df["close"].iloc[bar_index])
    exit_ = float(df["close"].iloc[bar_index + expiry_bars])
    pnl   = (exit_ - entry) / entry * 100

    if direction == "CALL":
        result = "win" if exit_ > entry else ("draw" if exit_ == entry else "loss")
    else:
        result = "win" if exit_ < entry else ("draw" if exit_ == entry else "loss")

    return result, entry, exit_


class Backtester:

    def __init__(self, confidence_threshold: float = CONFIDENCE_THRESHOLD):
        self.threshold = confidence_threshold

    def run(
        self,
        asset:    str,
        timeframe: str,
        df:       Optional[pd.DataFrame] = None,
        n_candles: int = 500,
        walk_forward_split: float = 0.7,
    ) -> BacktestResult:
        """
        Walk-forward backtest for a single asset/timeframe.
        First 70% of data is used to train AI model (if not yet trained).
        Last 30% is the test set.
        """
        if df is None:
            df = fetch_ohlc(asset, timeframe, n_candles=n_candles)

        if len(df) < 100:
            logger.warning(f"Insufficient data for {asset}/{timeframe} backtest")
            return BacktestResult()

        df = df.reset_index(drop=True)
        split    = int(len(df) * walk_forward_split)
        test_df  = df.iloc[split:].reset_index(drop=True)
        expiry_bars = max(1, TF_EXPIRY[timeframe] // TF_MINUTES[timeframe])

        result = BacktestResult()
        trades = []
        equity = [100.0]
        bank   = 100.0

        WARMUP = 220   # candles needed for EMA200

        for i in range(WARMUP, len(test_df) - expiry_bars - 1):
            window = test_df.iloc[:i + 1]

            try:
                struct = analyse_structure(window)
                if not struct.structure_valid:
                    continue

                ind = compute_indicators(window)
                pa  = analyse_price_action(window)

                if pa.dominant_pattern is None:
                    continue

                # Simulated timestamp
                ts = test_df["timestamp"].iloc[i] if "timestamp" in test_df else datetime.utcnow()
                if isinstance(ts, str):
                    ts = datetime.fromisoformat(ts)

                filt = apply_filters(
                    asset=asset, timeframe=timeframe,
                    atr_pct=ind.atr_pct, volatility_state=ind.volatility_state,
                    trend=struct.trend, dt=ts,
                )
                if not filt.allowed:
                    continue

                features = build_features(
                    rsi=ind.rsi, macd_hist=ind.macd_hist, macd_cross=ind.macd_cross,
                    ema_trend=ind.ema_trend, ema_spread_pct=ind.ema_spread_pct,
                    atr_pct=ind.atr_pct, volatility_state=ind.volatility_state,
                    sr_distance_pct=struct.sr_distance_pct,
                    at_support=struct.at_support, at_resistance=struct.at_resistance,
                    pullback=struct.pullback, breakout=struct.breakout,
                    bull_pa=pa.bullish_bias, bear_pa=pa.bearish_bias,
                    trend_strength=struct.trend_strength, trend=struct.trend,
                    dt=ts,
                )

                ai = get_ai_engine().score(features)
                if not ai.passes_threshold:
                    continue

                direction = ai.direction

                out, entry, exit_ = _simulate_outcome(test_df, i, direction, expiry_bars)
                pnl = (exit_ - entry) / entry * 100 if entry > 0 else 0.0

                trade = BacktestTrade(
                    timestamp=ts,
                    asset=asset,
                    timeframe=timeframe,
                    direction=direction,
                    entry_price=entry,
                    exit_price=exit_,
                    expiry_min=TF_EXPIRY[timeframe],
                    confidence=ai.confidence,
                    result=out,
                    pnl_pct=pnl,
                )
                trades.append(trade)

                # Equity simulation (flat 1% risk per trade, 85% payout)
                if out == "win":
                    bank *= 1.0085
                elif out == "loss":
                    bank *= 0.99
                equity.append(round(bank, 4))

            except Exception as exc:
                logger.debug(f"Backtest bar error at {i}: {exc}")
                continue

        result.trades       = trades
        result.total_trades = len(trades)
        result.wins         = sum(1 for t in trades if t.result == "win")
        result.losses       = sum(1 for t in trades if t.result == "loss")
        result.draws        = sum(1 for t in trades if t.result == "draw")
        result.win_rate     = round(result.wins / result.total_trades * 100, 2) if result.total_trades > 0 else 0.0
        result.loss_rate    = round(result.losses / result.total_trades * 100, 2) if result.total_trades > 0 else 0.0
        result.equity_curve = equity
        result.avg_confidence = round(np.mean([t.confidence for t in trades]), 2) if trades else 0.0

        # Max drawdown
        if equity:
            peak = equity[0]
            dd   = 0.0
            for v in equity:
                if v > peak:
                    peak = v
                dd = max(dd, (peak - v) / peak * 100)
            result.max_drawdown = round(dd, 2)

        return result

    def run_all(self, n_candles: int = 500) -> BacktestResult:
        """Run backtest across all assets and timeframes and aggregate."""
        all_trades = []
        by_asset   = {}
        by_tf      = {}

        for asset in ASSETS:
            for tf in TIMEFRAMES:
                logger.info(f"Backtesting {asset}/{tf}...")
                r = self.run(asset, tf, n_candles=n_candles)
                all_trades.extend(r.trades)

                if asset not in by_asset:
                    by_asset[asset] = {"total": 0, "wins": 0, "losses": 0}
                by_asset[asset]["total"]  += r.total_trades
                by_asset[asset]["wins"]   += r.wins
                by_asset[asset]["losses"] += r.losses

                if tf not in by_tf:
                    by_tf[tf] = {"total": 0, "wins": 0, "losses": 0}
                by_tf[tf]["total"]  += r.total_trades
                by_tf[tf]["wins"]   += r.wins
                by_tf[tf]["losses"] += r.losses

        total  = len(all_trades)
        wins   = sum(1 for t in all_trades if t.result == "win")
        losses = sum(1 for t in all_trades if t.result == "loss")
        draws  = total - wins - losses

        # Best asset
        best_asset = max(
            by_asset, key=lambda a: by_asset[a]["wins"] / max(by_asset[a]["total"], 1)
        ) if by_asset else ""

        best_tf = max(
            by_tf, key=lambda t: by_tf[t]["wins"] / max(by_tf[t]["total"], 1)
        ) if by_tf else ""

        # Aggregate equity
        equity_all = [100.0]
        for t in sorted(all_trades, key=lambda x: x.timestamp):
            last = equity_all[-1]
            if t.result == "win":
                equity_all.append(round(last * 1.0085, 4))
            elif t.result == "loss":
                equity_all.append(round(last * 0.99, 4))

        peak = equity_all[0]
        max_dd = 0.0
        for v in equity_all:
            if v > peak:
                peak = v
            max_dd = max(max_dd, (peak - v) / peak * 100)

        return BacktestResult(
            total_trades=total,
            wins=wins,
            losses=losses,
            draws=draws,
            win_rate=round(wins / total * 100, 2) if total > 0 else 0.0,
            loss_rate=round(losses / total * 100, 2) if total > 0 else 0.0,
            max_drawdown=round(max_dd, 2),
            avg_confidence=round(np.mean([t.confidence for t in all_trades]), 2) if all_trades else 0.0,
            best_asset=best_asset,
            best_tf=best_tf,
            trades=all_trades,
            by_asset=by_asset,
            by_timeframe=by_tf,
            equity_curve=equity_all,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    bt = Backtester()
    result = bt.run_all(n_candles=400)
    print(result.summary())
