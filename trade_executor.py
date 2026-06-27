"""
Trade Executor — Places trades on Deriv via WebSocket API.
Used when subscribers tap CALL/PUT buttons in Telegram.
"""

import json
import logging
import asyncio
import websockets
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

DERIV_WS_URL = "wss://ws.binaryws.com/websockets/v3?app_id=1089"

# Deriv symbol mapping
DERIV_SYMBOLS = {
    "EURUSD": "frxEURUSD",
    "GBPUSD": "frxGBPUSD",
    "XAUUSD": "frxXAUUSD",
    "USDJPY": "frxUSDJPY",
    "BTCUSD": "cryBTCUSD",
}

# Deriv contract type mapping
DIRECTION_MAP = {
    "CALL": "CALL",
    "PUT":  "PUT",
}


async def _place_trade_async(
    token: str,
    asset: str,
    direction: str,
    amount: float,
    expiry_min: int,
) -> dict:
    """
    Place a binary trade on Deriv.
    Returns result dict with success/error info.
    """
    symbol = DERIV_SYMBOLS.get(asset)
    if not symbol:
        return {"success": False, "error": f"Unsupported asset: {asset}"}

    contract_type = DIRECTION_MAP.get(direction)
    if not contract_type:
        return {"success": False, "error": f"Invalid direction: {direction}"}

    duration     = expiry_min
    duration_unit = "m"  # minutes

    try:
        async with websockets.connect(DERIV_WS_URL, ping_interval=20) as ws:

            # 1. Authenticate
            await ws.send(json.dumps({"authorize": token}))
            auth_resp = json.loads(await ws.recv())

            if auth_resp.get("error"):
                return {
                    "success": False,
                    "error": f"Auth failed: {auth_resp['error']['message']}"
                }

            balance = auth_resp.get("authorize", {}).get("balance", 0)
            currency = auth_resp.get("authorize", {}).get("currency", "USD")

            if float(balance) < amount:
                return {
                    "success": False,
                    "error": f"Insufficient balance. Available: {balance} {currency}"
                }

            # 2. Buy contract
            buy_request = {
                "buy": 1,
                "price": amount,
                "parameters": {
                    "amount":        amount,
                    "basis":         "stake",
                    "contract_type": contract_type,
                    "currency":      currency,
                    "duration":      duration,
                    "duration_unit": duration_unit,
                    "symbol":        symbol,
                }
            }

            await ws.send(json.dumps(buy_request))
            buy_resp = json.loads(await ws.recv())

            if buy_resp.get("error"):
                return {
                    "success": False,
                    "error": buy_resp["error"]["message"]
                }

            contract = buy_resp.get("buy", {})
            return {
                "success":       True,
                "contract_id":   contract.get("contract_id"),
                "buy_price":     contract.get("buy_price"),
                "payout":        contract.get("payout"),
                "balance_after": contract.get("balance_after"),
                "currency":      currency,
                "longcode":      contract.get("longcode", ""),
            }

    except Exception as exc:
        logger.error(f"Trade execution error: {exc}")
        return {"success": False, "error": str(exc)}


def place_trade(
    token: str,
    asset: str,
    direction: str,
    amount: float,
    expiry_min: int,
) -> dict:
    """Sync wrapper for async trade placement."""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(
            _place_trade_async(token, asset, direction, amount, expiry_min)
        )
    finally:
        loop.close()


async def _get_balance_async(token: str) -> dict:
    """Get account balance."""
    try:
        async with websockets.connect(DERIV_WS_URL, ping_interval=20) as ws:
            await ws.send(json.dumps({"authorize": token}))
            resp = json.loads(await ws.recv())

            if resp.get("error"):
                return {"success": False, "error": resp["error"]["message"]}

            auth = resp.get("authorize", {})
            return {
                "success":  True,
                "balance":  auth.get("balance", 0),
                "currency": auth.get("currency", "USD"),
                "name":     auth.get("fullname", ""),
                "loginid":  auth.get("loginid", ""),
            }
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def get_balance(token: str) -> dict:
    """Sync wrapper for balance check."""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(_get_balance_async(token))
    finally:
        loop.close()
