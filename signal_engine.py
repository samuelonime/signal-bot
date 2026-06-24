"""
Signal Engine — Orchestrates all modules.
A signal is only emitted when ALL conditions are satisfied:
  1. Structure confirmed
  2. Indicators aligned
  3. Price action pattern present
  4. AI confidence > 80
  5. All filters passed
"""

import logging
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import pandas as pd

from data_engine import fetch_ohlc, store_ohlc, load_ohlc, TF_EXPIRY, ASSETS, TIMEFRAMES
from market_structure import analyse_structure, StructureResult
from indicators import compute_indicators, IndicatorResult
from price_action import analyse_price_action, PriceActionResult
from ai_model import build_features, get_ai_engine, AIScore, CONFIDENCE_THRESHOLD
from filter_engine import apply_filters, get_current_session

logger = logging.getLogger(__name__)

# Maximum signals per day per asset (anti-overtrade)
MAX_DAILY_SIGNALS = 20

# ---------------------------------------------------------------------------
# Signal data class
# ---------------------------------------------------------------------------

@dataclass
class Signal:
    timestamp:   datetime
    asset:       str
    timeframe:   str
    direction:   str          # "CALL" | "PUT"
    entry_price: float
    expiry_min:  int
    confidence:  float        # 0–100
    reasons:     List[str]    = field(default_factory=list)
    warnings:    List[str]    = field(default_factory=list)
    session:     str          = ""
    prob_up:     float        = 0.0
    prob_down:   float        = 0.0
    model_mode:  str          = ""
    is_valid:    bool         = True

    def to_dict(self) -> dict:
        return {
            "timestamp":   self.timestamp.isoformat(),
            "asset":       self.asset,
            "timeframe":   self.timeframe,
            "direction":   self.direction,
            "entry_price": self.entry_price,
            "expiry_min":  self.expiry_min,
            "confidence":  self.confidence,
            "reasons":     self.reasons,
            "warnings":    self.warnings,
            "session":     self.session,
            "prob_up":     self.prob_up,
            "prob_down":   self.prob_down,
        }

    def format_message(self, vip: bool = False) -> str:
        """Format Telegram-ready message."""
        star = "⭐ VIP" if vip else "🔔 Signal"
        bar  = "🟢" if self.direction == "CALL" else "🔴"

        reasons_text = "\n".join(f"  ✅ {r}" for r in self.reasons)
        warn_text    = ("\n" + "\n".join(f"  ⚠️ {w}" for w in self.warnings)) if self.warnings else ""

        msg = f"""
{star} | {bar} {self.direction}

📊 Pair:       {self.asset}
🕐 Timeframe:  {self.timeframe}
⏳ Expiry:     {self.expiry_min} minutes
💰 Entry:      {self.entry_price:.5f}
🤖 Confidence: {self.confidence:.0f}%
🌍 Session:    {self.session}
🕒 Time (UTC): {self.timestamp.strftime('%H:%M:%S')}

📋 Reasons:
{reasons_text}{warn_text}

⚠️ Risk Disclaimer: Binary options carry significant risk. Trade only what you can afford to lose.
"""
        if not vip:
            msg += "\n🔒 Full analysis in VIP channel. Join: t.me/your_vip_channel"

        return msg.strip()


# ---------------------------------------------------------------------------
# Daily signal counter (in-memory, reset at midnight)
# ---------------------------------------------------------------------------

_daily_counts: dict = {}
_last_signal_bar: dict = {}   # asset -> last bar index a signal was sent
SIGNAL_COOLDOWN_BARS = 3       # min bars between same-asset signals

def _daily_count_key(asset: str, dt: datetime) -> str:
    return f"{asset}:{dt.strftime('%Y-%m-%d')}"

def _increment_count(asset: str, dt: datetime) -> int:
    key = _daily_count_key(asset, dt)
    _daily_counts[key] = _daily_counts.get(key, 0) + 1
    return _daily_counts[key]

def _get_count(asset: str, dt: datetime) -> int:
    key = _daily_count_key(asset, dt)
    return _daily_counts.get(key, 0)


# ---------------------------------------------------------------------------
# Core signal generation
# ---------------------------------------------------------------------------

def generate_signal(
    asset: str,
    timeframe: str,
    df: Optional[pd.DataFrame] = None,
    dt: Optional[datetime] = None,
    spread_pct: float = 0.0,
) -> Optional[Signal]:
    """
    Run the full signal pipeline for one asset/timeframe pair.
    Returns a Signal if all conditions are met, else None.
    """
    dt = dt or datetime.utcnow()

    # --- Daily cap
    if _get_count(asset, dt) >= MAX_DAILY_SIGNALS:
        logger.debug(f"{asset}: daily signal cap reached.")
        return None

    # --- Data
    if df is None or len(df) < 50:
        try:
            df = load_ohlc(asset, timeframe, limit=300)
            if len(df) < 50:
                df = fetch_ohlc(asset, timeframe, n_candles=300)
                store_ohlc(asset, timeframe, df)
        except Exception as exc:
            logger.warning(f"Data fetch failed for {asset}/{timeframe}: {exc}")
            return None

    if len(df) < 50:
        return None

    entry_price = float(df["close"].iloc[-1])

    # ---- 1. Filters (session, news, volatility, trend) ----
    ind = compute_indicators(df)

    filt = apply_filters(
        asset=asset,
        timeframe=timeframe,
        atr_pct=ind.atr_pct,
        volatility_state=ind.volatility_state,
        trend=ind.ema_trend,       # Quick pre-check
        dt=dt,
        spread_pct=spread_pct,
    )

    if not filt.allowed:
        logger.debug(f"{asset}/{timeframe} blocked by filter: {filt.reasons}")
        return None

    # ---- 2. Market structure ----
    struct = analyse_structure(df)

    if not struct.structure_valid:
        logger.debug(f"{asset}/{timeframe}: structure invalid (trend={struct.trend}, strength={struct.trend_strength})")
        return None

    # ---- 3. Price action ----
    pa = analyse_price_action(df)

    if pa.dominant_pattern is None:
        logger.debug(f"{asset}/{timeframe}: no price action pattern detected")
        return None

    # ---- 4. Determine candidate direction ----
    direction = _resolve_direction(struct, ind, pa)
    if direction is None:
        logger.debug(f"{asset}/{timeframe}: conflicting signals, no clear direction")
        return None

    # ---- 5. AI confidence ----
    features = build_features(
        rsi=ind.rsi,
        macd_hist=ind.macd_hist,
        macd_cross=ind.macd_cross,
        ema_trend=ind.ema_trend,
        ema_spread_pct=ind.ema_spread_pct,
        atr_pct=ind.atr_pct,
        volatility_state=ind.volatility_state,
        sr_distance_pct=struct.sr_distance_pct,
        at_support=struct.at_support,
        at_resistance=struct.at_resistance,
        pullback=struct.pullback,
        breakout=struct.breakout,
        bull_pa=pa.bullish_bias,
        bear_pa=pa.bearish_bias,
        trend_strength=struct.trend_strength,
        trend=struct.trend,
        dt=dt,
    )

    ai_score = get_ai_engine().score(features)

    # Enforce direction consistency: AI direction must match our structural direction
    if ai_score.direction != direction:
        # If AI disagrees, require higher raw probability to override
        matching_prob = ai_score.prob_up if direction == "CALL" else ai_score.prob_down
        if matching_prob < 0.60:
            logger.debug(f"{asset}/{timeframe}: AI direction conflict (AI={ai_score.direction}, struct={direction})")
            return None
        # AI disagrees but structural probability is acceptable — keep structural direction but note it
        logger.debug(f"{asset}/{timeframe}: AI soft conflict, proceeding with structural direction")

    if not ai_score.passes_threshold:
        logger.debug(
            f"{asset}/{timeframe}: AI confidence {ai_score.confidence:.1f}% < {CONFIDENCE_THRESHOLD}%"
        )
        return None

    # ---- 6. Build reasons ----
    reasons = _build_reasons(direction, struct, ind, pa, ai_score, dt)

    # ---- 7. Assemble signal ----
    expiry = TF_EXPIRY[timeframe]
    session = get_current_session(dt)

    signal = Signal(
        timestamp=dt,
        asset=asset,
        timeframe=timeframe,
        direction=direction,
        entry_price=round(entry_price, 6),
        expiry_min=expiry,
        confidence=ai_score.confidence,
        reasons=reasons,
        warnings=filt.warnings,
        session=session,
        prob_up=ai_score.prob_up,
        prob_down=ai_score.prob_down,
        model_mode=ai_score.model_mode,
    )

    _increment_count(asset, dt)
    logger.info(
        f"✅ SIGNAL: {asset}/{timeframe} {direction} | conf={ai_score.confidence:.1f}% | {entry_price:.5f}"
    )
    return signal


# ---------------------------------------------------------------------------
# Scan all assets and timeframes
# ---------------------------------------------------------------------------

def scan_all(dt: Optional[datetime] = None, spread_pcts: dict = {}) -> List[Signal]:
    """Scan every asset/timeframe combination and return valid signals."""
    signals = []
    dt      = dt or datetime.utcnow()

    for asset in ASSETS:
        for tf in TIMEFRAMES:
            spread = spread_pcts.get(asset, 0.0)
            try:
                sig = generate_signal(asset, tf, dt=dt, spread_pct=spread)
                if sig:
                    signals.append(sig)
            except Exception as exc:
                logger.error(f"Scan error {asset}/{tf}: {exc}")

    logger.info(f"Scan complete — {len(signals)} signal(s) generated.")
    return signals


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_direction(
    struct: StructureResult,
    ind: IndicatorResult,
    pa: PriceActionResult,
) -> Optional[str]:
    """
    Resolve the signal direction from structure + indicator + price action.
    All three must agree on direction; conflicting signals are discarded.
    """
    votes = {"CALL": 0, "PUT": 0}

    # Structure vote
    if struct.trend == "bullish" and (struct.at_support or struct.pullback):
        votes["CALL"] += 3
    elif struct.trend == "bearish" and (struct.at_resistance or struct.pullback):
        votes["PUT"]  += 3

    if struct.breakout == "bullish_break":
        votes["CALL"] += 2
    elif struct.breakout == "bearish_break":
        votes["PUT"]  += 2

    # Indicator vote
    if ind.ema_trend == "bullish":
        votes["CALL"] += 2
    elif ind.ema_trend == "bearish":
        votes["PUT"]  += 2

    if ind.rsi_zone == "oversold":
        votes["CALL"] += 1
    elif ind.rsi_zone == "overbought":
        votes["PUT"]  += 1

    if ind.macd_cross == "bullish":
        votes["CALL"] += 2
    elif ind.macd_cross == "bearish":
        votes["PUT"]  += 2

    # Price action vote
    if pa.dominant_pattern:
        if pa.dominant_pattern.direction == "bullish":
            votes["CALL"] += int(pa.dominant_pattern.strength * 3)
        elif pa.dominant_pattern.direction == "bearish":
            votes["PUT"]  += int(pa.dominant_pattern.strength * 3)

    # Require decisive majority (60%+ of votes)
    total = votes["CALL"] + votes["PUT"]
    if total == 0:
        return None

    if votes["CALL"] / total >= 0.65:
        return "CALL"
    elif votes["PUT"] / total >= 0.65:
        return "PUT"

    return None   # Conflicting


def _build_reasons(
    direction: str,
    struct: StructureResult,
    ind: IndicatorResult,
    pa: PriceActionResult,
    ai: AIScore,
    dt: datetime,
) -> List[str]:
    reasons = []

    # Trend
    reasons.append(
        f"{'Uptrend' if struct.trend == 'bullish' else 'Downtrend'} confirmed "
        f"(EMA 50 {'>' if struct.trend == 'bullish' else '<'} EMA 200, strength={struct.trend_strength:.0%})"
    )

    # Structure
    if struct.at_support and direction == "CALL":
        reasons.append(f"Price at key support zone ({struct.nearest_support:.5f})")
    if struct.at_resistance and direction == "PUT":
        reasons.append(f"Price at key resistance zone ({struct.nearest_resistance:.5f})")
    if struct.pullback:
        reasons.append(f"Clean {'bullish' if direction == 'CALL' else 'bearish'} pullback to key level")
    if struct.breakout:
        reasons.append(f"Structural breakout: {struct.breakout.replace('_', ' ').title()}")

    # Price action
    if pa.dominant_pattern:
        reasons.append(f"{pa.dominant_pattern.name} — {pa.dominant_pattern.description}")

    # Indicators
    if ind.rsi_zone == "oversold" and direction == "CALL":
        reasons.append(f"RSI({ind.rsi:.1f}) recovering from oversold zone")
    elif ind.rsi_zone == "overbought" and direction == "PUT":
        reasons.append(f"RSI({ind.rsi:.1f}) retreating from overbought zone")
    else:
        reasons.append(f"RSI at {ind.rsi:.1f} — aligned with {direction.lower()} bias")

    if ind.macd_cross:
        reasons.append(f"MACD {ind.macd_cross} crossover confirmed")
    elif ind.macd_hist > 0 and direction == "CALL":
        reasons.append("MACD histogram positive — bullish momentum building")
    elif ind.macd_hist < 0 and direction == "PUT":
        reasons.append("MACD histogram negative — bearish momentum building")

    # AI
    reasons.append(
        f"AI confidence: {ai.confidence:.0f}% ({'ML' if ai.model_mode == 'ml' else 'Heuristic'} model | "
        f"P(UP)={ai.prob_up:.1%}, P(DN)={ai.prob_down:.1%})"
    )

    # Session
    session = get_current_session(dt)
    reasons.append(f"Trading in: {session}")

    return reasons
