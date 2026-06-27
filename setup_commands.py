"""
Run this ONCE to register bot commands as buttons in Telegram.
After running, users will see all commands in the menu button (/) at the bottom of chat.

Usage: python setup_commands.py
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")

def set_commands():
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/setMyCommands"

    commands = [
        {"command": "start",       "description": "👋 Welcome message"},
        {"command": "help",        "description": "📖 Show all commands"},
        {"command": "status",      "description": "✅ Bot status & uptime"},
        {"command": "pairs",       "description": "📊 Monitored pairs & timeframes"},
        {"command": "stats",       "description": "📈 Today's win/loss report"},
        {"command": "connect",     "description": "🔗 Link your Deriv account"},
        {"command": "disconnect",  "description": "❌ Unlink Deriv account"},
        {"command": "balance",     "description": "💰 Check Deriv balance"},
        {"command": "setamount",   "description": "💵 Set default trade amount"},
        {"command": "myaccount",   "description": "👤 View your account settings"},
    ]

    resp = requests.post(url, json={"commands": commands})

    if resp.ok and resp.json().get("result"):
        print("✅ Bot commands registered successfully!")
        print("Users will now see all commands in the / menu button.")
    else:
        print(f"❌ Failed: {resp.text}")


if __name__ == "__main__":
    if not BOT_TOKEN:
        print("❌ TELEGRAM_BOT_TOKEN not found in .env")
    else:
        set_commands()
