# Signal Bot Pro
### Binary Options Signal Bot — AI-Powered · Multi-Asset · High-Confidence

> ⚠️ **Risk Disclaimer**: Binary options trading carries significant financial risk.  
> This bot does **not** guarantee profits. It filters for high-probability setups —  
> no system wins 100% of the time. Trade only what you can afford to lose.

---

## Architecture

```
signal_engine.py          ← Orchestrates all modules, applies ALL gates
├── market_structure.py   ← Trend, S/R zones, breakouts, pullbacks (PRIMARY GATE)
├── indicators.py         ← EMA50/200, RSI, MACD, ATR (CONFIRMATION ONLY)
├── price_action.py       ← Engulfing, pin bars, doji, rejection candles
├── ai_model.py           ← LightGBM/XGBoost confidence engine (>80% threshold)
├── filter_engine.py      ← News, session, volatility, spread filters
└── data_engine.py        ← OHLC fetch (Alpha Vantage / synthetic) + PostgreSQL

main.py                   ← Live bot loop / backtest / single scan
telegram_bot.py           ← Free + VIP channel delivery
backtester.py             ← Walk-forward backtest engine
performance_tracker.py    ← Signal logging + daily/weekly reports
dashboard.py              ← FastAPI REST API + HTML dashboard
```

---

## Signal Generation Logic

A signal is emitted **ONLY** when all of the following are true:

```
1. ✅ Filters pass       → Not weekend/dead session/news/low vol/ranging
2. ✅ Structure valid    → Clear trend + price at key S/R level
3. ✅ Indicators align  → EMA, RSI, MACD all confirm direction
4. ✅ PA pattern found  → Engulfing/pin bar/rejection at key level
5. ✅ AI confidence >80 → LightGBM model (or heuristic) agrees
6. ✅ Votes decisive    → 65%+ consensus across all modules
```

---

## Assets & Timeframes

| Asset   | Description | Notes                         |
|---------|-------------|-------------------------------|
| EURUSD  | Euro/Dollar | Most liquid, tightest spread  |
| GBPUSD  | Cable       | Volatile during London hours  |
| XAUUSD  | Gold        | Best during London/NY overlap |

| Timeframe | Expiry  | Best For         |
|-----------|---------|------------------|
| M1        | 3 min   | Scalp momentum   |
| M5        | 5 min   | Intraday setups  |
| M15       | 15 min  | Swing setups     |

---

## Installation

```bash
# 1. Clone / copy project
cd signal_bot

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
# venv\Scripts\activate    # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 5. Set up PostgreSQL
createdb signal_bot
# Or update DATABASE_URL in .env

# 6. Initialise database
python data_engine.py
```

---

## Usage

```bash
# Run live bot (loops forever)
python main.py

# Single scan — useful for testing
python main.py --scan-once

# Full backtest across all assets/timeframes
python main.py --backtest

# Start API dashboard (http://localhost:8000)
uvicorn dashboard:app --host 0.0.0.0 --port 8000
```

---

## AI Model Training

The model starts in **heuristic mode** (rule-based).  
After accumulating ~200+ resolved signals, train the ML model:

```python
from ai_model import AIConfidenceEngine, build_training_dataset
from performance_tracker import _fetch_signals_for_period
from datetime import datetime, timedelta

# Fetch resolved signals
df = _fetch_signals_for_period(
    datetime.utcnow() - timedelta(days=90),
    datetime.utcnow()
)

# Build feature matrix (requires feature columns from signal log)
X, y = build_training_dataset(df)

engine = AIConfidenceEngine()
engine.train(X, y)
# Model saved to models/signal_model.pkl automatically
```

---

## Telegram Setup

1. Open @BotFather on Telegram
2. Create a new bot → copy token to `TELEGRAM_BOT_TOKEN`
3. Create a Free channel → add bot as admin → set `TELEGRAM_FREE_CHANNEL`
4. Create a VIP channel → add bot as admin → set `TELEGRAM_VIP_CHANNEL`
5. Find your personal chat ID via @userinfobot → set `TELEGRAM_ADMIN_CHAT_ID`

### Free vs VIP

| Feature              | Free Channel | VIP Channel |
|----------------------|:------------:|:-----------:|
| Signal direction     | ✅           | ✅          |
| Confidence score     | ✅           | ✅          |
| Entry price          | ❌           | ✅          |
| Full analysis        | ❌           | ✅          |
| Performance reports  | ❌           | ✅          |

---

## API Endpoints

| Method | Endpoint                     | Description              |
|--------|------------------------------|--------------------------|
| GET    | `/`                          | HTML Dashboard           |
| GET    | `/api/status`                | Bot health check         |
| POST   | `/api/scan`                  | Trigger full scan        |
| GET    | `/api/signals`               | Latest signals           |
| GET    | `/api/signals/history?days=7`| Historical signals       |
| GET    | `/api/performance/daily`     | Daily report             |
| GET    | `/api/performance/weekly`    | Weekly report            |
| GET    | `/api/scan/{asset}/{tf}`     | Scan specific pair       |

---

## Session Guide

| Session          | Hours (UTC) | Quality    |
|------------------|-------------|------------|
| 🌏 Asian         | 00–07       | ❌ Avoid   |
| 🇬🇧 London        | 07–16       | ✅ Good    |
| 🇺🇸 New York      | 13–22       | ✅ Good    |
| ⭐ Overlap       | 13–16       | 🌟 Best    |
| 🌙 Late NY/Dead  | 22–00       | ❌ Avoid   |

---

## Filter Rules (Why Trades Are Blocked)

- **Dead session**: Asian zone 22:00–07:00 UTC — low liquidity, random moves
- **High-impact news**: NFP, CPI, FOMC ±30 min — unpredictable volatility
- **Low volatility**: ATR% < 0.05% — market sleeping, no real moves
- **Ranging market**: No clear trend — S/R levels unreliable
- **High spread**: Spread erodes edge on binary options instantly

---

## Project Files

```
signal_bot/
├── main.py                  ← Entry point
├── data_engine.py           ← OHLC data + PostgreSQL
├── market_structure.py      ← Structure analysis (primary gate)
├── indicators.py            ← Technical indicators (confirmation)
├── price_action.py          ← Candlestick pattern detection
├── ai_model.py              ← LightGBM/XGBoost confidence engine
├── signal_engine.py         ← Signal orchestration
├── filter_engine.py         ← Trade filters
├── telegram_bot.py          ← Telegram channel delivery
├── backtester.py            ← Walk-forward backtesting
├── performance_tracker.py   ← Signal logging + reports
├── dashboard.py             ← FastAPI REST + HTML UI
├── requirements.txt
├── .env.example
└── models/                  ← ML model artifacts (auto-created)
    └── signal_model.pkl
```

---

## Performance Expectations

> Realistic expectations based on multi-confluence filter logic:

- **Expected signals/day**: 5–15 (strict filtering, not overtrading)
- **Target win rate**: 65–75% (with trained ML model)
- **Confidence threshold**: 80% minimum (hard cutoff)
- **Best conditions**: London/NY overlap, clear trend, confirmed S/R

---

*Built for Samuel @ De Ebumars Innovation Ltd / MeritLives*
