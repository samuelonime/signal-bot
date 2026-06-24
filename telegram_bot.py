"""
Telegram Bot — Sends signals to Free and VIP channels.
Free channel: direction + confidence only.
VIP channel: full signal with all reasons.
"""

import os
import asyncio
import logging
import requests
from datetime import datetime
from typing import Optional, List
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

BOT_TOKEN      = os.getenv("TELEGRAM_BOT_TOKEN", "")
FREE_CHANNEL   = os.getenv("TELEGRAM_FREE_CHANNEL",  "")  # e.g. "@your_free_channel"
VIP_CHANNEL    = os.getenv("TELEGRAM_VIP_CHANNEL",   "")  # e.g. "@your_vip_channel"
ADMIN_CHAT_ID  = os.getenv("TELEGRAM_ADMIN_CHAT_ID", "")

TELEGRAM_API = "https://api.telegram.org/bot{token}/{method}"


# ---------------------------------------------------------------------------
# Low-level sender
# ---------------------------------------------------------------------------

def _send_message(chat_id: str, text: str, parse_mode: str = "HTML") -> bool:
    if not BOT_TOKEN or not chat_id:
        logger.warning("Telegram not configured — message not sent.")
        return False

    url  = TELEGRAM_API.format(token=BOT_TOKEN, method="sendMessage")
    data = {
        "chat_id":    chat_id,
        "text":       text,
        "parse_mode": parse_mode,
    }
    try:
        resp = requests.post(url, json=data, timeout=10)
        if resp.ok:
            logger.info(f"Telegram message sent to {chat_id}")
            return True
        else:
            logger.error(f"Telegram error: {resp.status_code} {resp.text[:200]}")
            return False
    except Exception as exc:
        logger.error(f"Telegram send failed: {exc}")
        return False


# ---------------------------------------------------------------------------
# Signal formatters
# ---------------------------------------------------------------------------

def _format_free_message(signal) -> str:
    icon = "🟢" if signal.direction == "CALL" else "🔴"
    return (
        f"🔔 <b>New Signal</b>\n\n"
        f"{icon} <b>{signal.direction}</b> — {signal.asset}\n"
        f"⏳ Expiry: {signal.expiry_min} min\n"
        f"🤖 Confidence: <b>{signal.confidence:.0f}%</b>\n"
        f"🕒 Time: {signal.timestamp.strftime('%H:%M UTC')}\n\n"
        f"📊 Full analysis available in VIP 👇\n"
        f"🔒 Join: {os.getenv('VIP_INVITE_LINK', 't.me/your_vip_link')}"
    )


def _format_vip_message(signal) -> str:
    icon = "🟢" if signal.direction == "CALL" else "🔴"
    reasons_html = "\n".join(f"  ✅ {r}" for r in signal.reasons)
    warn_html    = ""
    if signal.warnings:
        warn_html = "\n⚠️ <b>Warnings:</b>\n" + "\n".join(f"  ⚠️ {w}" for w in signal.warnings)

    return (
        f"⭐ <b>VIP Signal</b> — {icon} {signal.direction}\n\n"
        f"📊 <b>Pair:</b>      {signal.asset}\n"
        f"📈 <b>Timeframe:</b> {signal.timeframe}\n"
        f"⏳ <b>Expiry:</b>    {signal.expiry_min} minutes\n"
        f"💰 <b>Entry:</b>     <code>{signal.entry_price:.5f}</code>\n"
        f"🤖 <b>Confidence:</b> <b>{signal.confidence:.0f}%</b>\n"
        f"🌍 <b>Session:</b>   {signal.session}\n"
        f"🕒 <b>UTC Time:</b>  {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        f"📋 <b>Analysis:</b>\n{reasons_html}{warn_html}\n\n"
        f"⚠️ <i>Risk disclaimer: Binary options carry significant financial risk. "
        f"Never trade with money you cannot afford to lose. Past performance does not guarantee future results.</i>"
    )


# ---------------------------------------------------------------------------
# Public send functions
# ---------------------------------------------------------------------------

def send_signal(signal) -> dict:
    """
    Send signal to both Free and VIP channels.
    Returns dict with send status per channel.
    """
    results = {}

    if FREE_CHANNEL:
        results["free"] = _send_message(FREE_CHANNEL, _format_free_message(signal))

    if VIP_CHANNEL:
        results["vip"] = _send_message(VIP_CHANNEL, _format_vip_message(signal))

    return results


def send_admin_alert(text: str):
    """Send a system alert to admin chat."""
    if ADMIN_CHAT_ID:
        _send_message(ADMIN_CHAT_ID, f"🤖 <b>Bot Alert</b>\n\n{text}")


def send_performance_report(report_text: str, report_type: str = "Daily"):
    """Send daily/weekly performance report."""
    msg = (
        f"📊 <b>{report_type} Performance Report</b>\n"
        f"🕒 {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n\n"
        f"{report_text}"
    )
    if VIP_CHANNEL:
        _send_message(VIP_CHANNEL, msg)
    if ADMIN_CHAT_ID:
        _send_message(ADMIN_CHAT_ID, msg)


# ---------------------------------------------------------------------------
# Telegram Bot webhook handler (for receiving commands)
# ---------------------------------------------------------------------------

class BotCommandHandler:
    """
    Handles /start, /status, /stats commands from Telegram users.
    For production: use python-telegram-bot library with webhook.
    This is a simplified polling-based implementation.
    """

    def __init__(self):
        self.offset = 0

    def poll_once(self):
        if not BOT_TOKEN:
            return []

        url  = TELEGRAM_API.format(token=BOT_TOKEN, method="getUpdates")
        data = {"offset": self.offset, "timeout": 10, "limit": 10}
        try:
            resp = requests.get(url, params=data, timeout=15)
            if not resp.ok:
                return []

            updates = resp.json().get("result", [])
            for update in updates:
                self.offset = update["update_id"] + 1
                self._handle_update(update)
            return updates
        except Exception as exc:
            logger.error(f"Poll error: {exc}")
            return []

    def _handle_update(self, update: dict):
        msg = update.get("message", {})
        if not msg:
            return

        chat_id = str(msg["chat"]["id"])
        text    = msg.get("text", "").strip()

        if text == "/start":
            _send_message(chat_id,
                "👋 Welcome to <b>Signal Bot Pro</b>!\n\n"
                "I generate high-confidence binary options signals using AI + market structure analysis.\n\n"
                "Commands:\n"
                "/status — Bot status\n"
                "/stats — Today's performance\n\n"
                "⚠️ <i>Trading involves significant risk.</i>"
            )

        elif text == "/status":
            _send_message(chat_id,
                "✅ <b>Bot Status: ONLINE</b>\n"
                f"🕒 {datetime.utcnow().strftime('%H:%M UTC')}\n"
                "📡 Scanning: EURUSD, GBPUSD, XAUUSD\n"
                "⏱️ Timeframes: M1, M5, M15"
            )

        elif text == "/stats":
            # Fetch from DB in production
            _send_message(chat_id,
                "📊 <b>Today's Stats</b>\n\n"
                "Available after first signals are resolved.\n"
                "Join VIP for detailed performance tracking."
            )
