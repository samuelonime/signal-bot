"""
FastAPI Dashboard
REST API + HTML dashboard for monitoring signals, performance, and bot health.
Run: uvicorn dashboard:app --host 0.0.0.0 --port 8000
"""

import os
import json
import logging
from datetime import datetime, date, timedelta
from typing import List, Optional

from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from signal_engine import scan_all, Signal, ASSETS, TIMEFRAMES
from data_engine import init_db, refresh_all
from performance_tracker import generate_daily_report, generate_weekly_report, _fetch_signals_for_period

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Signal Bot Pro",
    description="Binary Options Signal Bot Dashboard",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

_latest_signals: List[Signal] = []
_last_scan_ts: Optional[datetime] = None


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup():
    try:
        init_db()
        refresh_all()
        logger.info("Dashboard startup: DB initialised, data refreshed.")
    except Exception as exc:
        logger.error(f"Startup error: {exc}")


@app.get("/api/status")
def get_status():
    return {
        "status": "online",
        "timestamp": datetime.utcnow().isoformat(),
        "last_scan": _last_scan_ts.isoformat() if _last_scan_ts else None,
        "assets": ASSETS,
        "timeframes": TIMEFRAMES,
    }


@app.post("/api/scan")
def trigger_scan():
    global _latest_signals, _last_scan_ts
    signals = scan_all()
    _latest_signals = signals
    _last_scan_ts   = datetime.utcnow()
    return {
        "scanned_at": _last_scan_ts.isoformat(),
        "signal_count": len(signals),
        "signals": [s.to_dict() for s in signals],
    }


@app.get("/api/signals")
def get_latest_signals():
    return {
        "count": len(_latest_signals),
        "signals": [s.to_dict() for s in _latest_signals],
        "last_scan": _last_scan_ts.isoformat() if _last_scan_ts else None,
    }


@app.get("/api/signals/history")
def get_signal_history(days: int = Query(7, ge=1, le=90)):
    start = datetime.utcnow() - timedelta(days=days)
    end   = datetime.utcnow()
    df    = _fetch_signals_for_period(start, end)
    if df.empty:
        return {"signals": [], "total": 0}
    records = df.to_dict(orient="records")
    return {"signals": records, "total": len(records)}


@app.get("/api/performance/daily")
def daily_performance(target_date: Optional[str] = None):
    d = date.fromisoformat(target_date) if target_date else None
    return {"report": generate_daily_report(d)}


@app.get("/api/performance/weekly")
def weekly_performance(week_offset: int = Query(0, ge=0, le=52)):
    return {"report": generate_weekly_report(week_offset)}


@app.get("/api/scan/{asset}/{timeframe}")
def scan_pair(asset: str, timeframe: str):
    if asset not in ASSETS:
        raise HTTPException(404, f"Asset {asset} not supported. Choose: {ASSETS}")
    if timeframe not in TIMEFRAMES:
        raise HTTPException(404, f"Timeframe {timeframe} not supported. Choose: {TIMEFRAMES}")

    from signal_engine import generate_signal
    sig = generate_signal(asset, timeframe)
    if sig:
        return {"signal": sig.to_dict()}
    return {"signal": None, "message": "No valid signal at this time."}


# ---------------------------------------------------------------------------
# HTML Dashboard
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def dashboard():
    return HTMLResponse(content=_build_dashboard_html())


def _build_dashboard_html() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Signal Bot Pro — Dashboard</title>
<style>
  :root {
    --bg:      #0B0F1A;
    --surface: #141B2D;
    --card:    #1A2340;
    --accent:  #00D4FF;
    --green:   #00E5A0;
    --red:     #FF4D6D;
    --yellow:  #F5A623;
    --text:    #E2E8F0;
    --muted:   #64748B;
    --border:  rgba(255,255,255,0.08);
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'SF Mono', 'JetBrains Mono', 'Fira Code', monospace;
    min-height: 100vh;
  }

  header {
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 18px 32px;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .logo {
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--accent);
    letter-spacing: 0.05em;
  }

  .logo span { color: var(--green); }

  #status-dot {
    width: 10px; height: 10px;
    background: var(--green);
    border-radius: 50%;
    display: inline-block;
    margin-right: 8px;
    animation: pulse 2s infinite;
  }

  @keyframes pulse {
    0%,100% { opacity: 1; }
    50%      { opacity: 0.4; }
  }

  main { padding: 32px; max-width: 1400px; margin: 0 auto; }

  .grid-4 {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 20px;
    margin-bottom: 32px;
  }

  .stat-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 24px;
  }

  .stat-label { font-size: 0.75rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.1em; }
  .stat-value { font-size: 2rem; font-weight: 700; color: var(--accent); margin-top: 8px; }

  .section-title {
    font-size: 0.85rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 16px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
  }

  .signals-container {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(380px, 1fr));
    gap: 20px;
    margin-bottom: 40px;
  }

  .signal-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 24px;
    position: relative;
    overflow: hidden;
  }

  .signal-card.call { border-left: 4px solid var(--green); }
  .signal-card.put  { border-left: 4px solid var(--red); }

  .signal-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 16px;
  }

  .signal-pair { font-size: 1.3rem; font-weight: 700; color: var(--text); }
  .signal-tf   { font-size: 0.75rem; color: var(--muted); margin-top: 2px; }

  .direction-badge {
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 0.08em;
  }
  .direction-badge.call { background: rgba(0,229,160,0.15); color: var(--green); }
  .direction-badge.put  { background: rgba(255,77,109,0.15); color: var(--red); }

  .signal-meta {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    margin-bottom: 16px;
  }

  .meta-item .label { font-size: 0.7rem; color: var(--muted); text-transform: uppercase; }
  .meta-item .value { font-size: 0.95rem; color: var(--text); margin-top: 2px; }

  .confidence-bar {
    background: rgba(255,255,255,0.08);
    border-radius: 4px;
    height: 6px;
    margin-top: 4px;
    overflow: hidden;
  }
  .confidence-fill {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, var(--yellow), var(--green));
    transition: width 0.5s ease;
  }

  .reasons-list {
    list-style: none;
    font-size: 0.78rem;
    color: var(--muted);
    margin-top: 12px;
    padding-top: 12px;
    border-top: 1px solid var(--border);
  }
  .reasons-list li { padding: 2px 0; }
  .reasons-list li::before { content: "✓ "; color: var(--green); }

  .empty-state {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 60px;
    text-align: center;
    color: var(--muted);
    grid-column: 1/-1;
  }

  .scan-btn {
    background: linear-gradient(135deg, var(--accent), #0099BB);
    border: none;
    color: #000;
    font-family: inherit;
    font-size: 0.9rem;
    font-weight: 700;
    padding: 12px 28px;
    border-radius: 8px;
    cursor: pointer;
    letter-spacing: 0.05em;
    transition: opacity 0.2s;
  }
  .scan-btn:hover { opacity: 0.85; }
  .scan-btn:disabled { opacity: 0.4; cursor: not-allowed; }

  .last-scan { font-size: 0.75rem; color: var(--muted); margin-top: 8px; }

  footer {
    border-top: 1px solid var(--border);
    padding: 20px 32px;
    text-align: center;
    font-size: 0.72rem;
    color: var(--muted);
  }
</style>
</head>
<body>

<header>
  <div class="logo">Signal Bot <span>Pro</span></div>
  <div style="display:flex;align-items:center;gap:12px;">
    <span><span id="status-dot"></span><span id="status-text" style="font-size:0.8rem;color:var(--green)">ONLINE</span></span>
    <span id="clock" style="font-size:0.8rem;color:var(--muted)"></span>
  </div>
</header>

<main>
  <div class="grid-4">
    <div class="stat-card">
      <div class="stat-label">Signals Today</div>
      <div class="stat-value" id="stat-total">—</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Latest Confidence</div>
      <div class="stat-value" id="stat-conf">—</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Active Assets</div>
      <div class="stat-value" id="stat-assets">3</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Timeframes</div>
      <div class="stat-value" id="stat-tfs">3</div>
    </div>
  </div>

  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:24px;">
    <div class="section-title" style="margin:0;border:none;">Live Signals</div>
    <div>
      <button class="scan-btn" id="scan-btn" onclick="runScan()">⟳ Scan Market</button>
      <div class="last-scan" id="last-scan-text">Never scanned</div>
    </div>
  </div>

  <div class="signals-container" id="signals-container">
    <div class="empty-state">
      <div style="font-size:2rem;margin-bottom:12px;">📡</div>
      <div style="font-size:1rem;color:var(--text);margin-bottom:8px;">No signals yet</div>
      <div>Click "Scan Market" to analyse all assets and timeframes.</div>
    </div>
  </div>

</main>

<footer>
  ⚠️ Binary options carry significant financial risk. This bot does not guarantee profits.
  Trade responsibly. Past performance ≠ future results. | Signal Bot Pro v1.0
</footer>

<script>
  // Clock
  function updateClock() {
    const now = new Date();
    document.getElementById('clock').textContent =
      now.toUTCString().slice(17, 25) + ' UTC';
  }
  setInterval(updateClock, 1000);
  updateClock();

  // Scan
  async function runScan() {
    const btn = document.getElementById('scan-btn');
    btn.disabled = true;
    btn.textContent = '⟳ Scanning...';

    try {
      const resp = await fetch('/api/scan', { method: 'POST' });
      const data = await resp.json();
      renderSignals(data.signals || []);
      document.getElementById('stat-total').textContent = data.signal_count || 0;
      document.getElementById('last-scan-text').textContent =
        'Last scan: ' + new Date(data.scanned_at).toLocaleTimeString();
      if (data.signals && data.signals.length > 0) {
        const confs = data.signals.map(s => s.confidence);
        document.getElementById('stat-conf').textContent =
          Math.max(...confs).toFixed(0) + '%';
      }
    } catch (e) {
      console.error('Scan failed:', e);
      alert('Scan failed — check API connection.');
    }

    btn.disabled = false;
    btn.textContent = '⟳ Scan Market';
  }

  function renderSignals(signals) {
    const container = document.getElementById('signals-container');
    if (!signals || signals.length === 0) {
      container.innerHTML = `<div class="empty-state">
        <div style="font-size:2rem;margin-bottom:12px;">🔍</div>
        <div style="font-size:1rem;color:var(--text);margin-bottom:8px;">No signals found</div>
        <div>All filters passed but no high-confidence setups detected. Market conditions may be unfavourable.</div>
      </div>`;
      return;
    }

    container.innerHTML = signals.map(s => {
      const dir    = s.direction.toLowerCase();
      const conf   = parseFloat(s.confidence).toFixed(0);
      const reasons = (s.reasons || []).slice(0, 4)
        .map(r => `<li>${r}</li>`).join('');

      return `
      <div class="signal-card ${dir}">
        <div class="signal-header">
          <div>
            <div class="signal-pair">${s.asset}</div>
            <div class="signal-tf">${s.timeframe} · ${s.expiry_min}min expiry</div>
          </div>
          <span class="direction-badge ${dir}">${s.direction}</span>
        </div>

        <div class="signal-meta">
          <div class="meta-item">
            <div class="label">Entry Price</div>
            <div class="value">${parseFloat(s.entry_price).toFixed(5)}</div>
          </div>
          <div class="meta-item">
            <div class="label">Session</div>
            <div class="value">${s.session || '—'}</div>
          </div>
          <div class="meta-item">
            <div class="label">Confidence</div>
            <div class="value" style="color:${conf >= 85 ? 'var(--green)' : 'var(--yellow)'}">
              ${conf}%
            </div>
            <div class="confidence-bar">
              <div class="confidence-fill" style="width:${conf}%"></div>
            </div>
          </div>
          <div class="meta-item">
            <div class="label">Time (UTC)</div>
            <div class="value">${new Date(s.timestamp).toLocaleTimeString()}</div>
          </div>
        </div>

        <ul class="reasons-list">${reasons}</ul>
      </div>`;
    }).join('');
  }

  // Auto scan on load (optional)
  // window.addEventListener('load', runScan);
</script>
</body>
</html>"""
