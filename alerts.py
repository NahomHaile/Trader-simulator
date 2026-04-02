#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    TRADE ALERT SYSTEM                                      ║
║                    ─────────────────                                       ║
║  Sends notifications when the Veteran Trader v2 generates actionable       ║
║  signals. Supports: Windows desktop toasts, Discord webhooks, Telegram.    ║
║                                                                            ║
║  SETUP:                                                                    ║
║    1. Copy alert_config_TEMPLATE.json to alert_config.json                 ║
║    2. Fill in your webhook URLs / bot tokens                               ║
║    3. Run: python alerts.py QQQ_data.csv                                   ║
║       Or let auto_run.py call it automatically                             ║
║                                                                            ║
║  DISCORD SETUP:                                                            ║
║    1. Server Settings -> Integrations -> Webhooks -> New Webhook            ║
║    2. Copy the webhook URL into alert_config.json                          ║
║                                                                            ║
║  TELEGRAM SETUP:                                                           ║
║    1. Message @BotFather on Telegram, create a bot, get the token          ║
║    2. Start a chat with your bot, then visit:                              ║
║       https://api.telegram.org/bot<TOKEN>/getUpdates                       ║
║    3. Find your chat_id in the response                                    ║
║    4. Put both in alert_config.json                                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import json
import os
import sys
import urllib.request
import urllib.error
from datetime import datetime

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
from veteran_trader_v2 import (
    OHLCV, TraderConfig, RiskProfile, VeteranTrader, Signal, SignalContext,
    load_data
)


CONFIG_FILE = os.path.join(script_dir, "alert_config.json")

DEFAULT_CONFIG = {
    "desktop_notifications": True,
    "discord_webhook_url": "",
    "telegram_bot_token": "",
    "telegram_chat_id": "",
    "alert_on": {
        "strong_buy": True,
        "buy": True,
        "strong_sell": True,
        "sell": True,
        "lean_buy": False,
        "lean_sell": False,
        "high_confidence_low_risk": True,
    },
    "min_conviction_for_alert": 0.5,
    "tickers": ["QQQ"],
    "risk_profile": "moderate",
    "capital": 100000,
    "smt_compare": "",
}


def load_alert_config() -> dict:
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    else:
        # Create template
        template_path = os.path.join(script_dir, "alert_config.json")
        with open(template_path, "w") as f:
            json.dump(DEFAULT_CONFIG, f, indent=2)
        print(f"  Created alert_config.json — edit it with your webhook URLs")
        return DEFAULT_CONFIG


def should_alert(signal, config: dict) -> bool:
    """Check if this signal should trigger an alert."""
    alert_on = config.get("alert_on", {})
    min_conv = config.get("min_conviction_for_alert", 0.5)

    if signal.conviction < min_conv:
        return False

    sig_map = {
        Signal.STRONG_BUY: "strong_buy",
        Signal.BUY: "buy",
        Signal.STRONG_SELL: "strong_sell",
        Signal.SELL: "sell",
        Signal.LEAN_BUY: "lean_buy",
        Signal.LEAN_SELL: "lean_sell",
    }

    key = sig_map.get(signal.signal)
    if key and alert_on.get(key, False):
        return True

    # Always alert on HCLR
    if signal.context == SignalContext.HIGH_CONFIDENCE_LOW_RISK and alert_on.get("high_confidence_low_risk", True):
        return True

    return False


def format_alert(signal, ticker: str) -> dict:
    """Format signal into alert message with clear action instructions."""
    is_buy = "BUY" in signal.signal.value
    direction = "LONG" if is_buy else "SHORT"
    emoji = "🟢" if is_buy else "🔴"
    if signal.context == SignalContext.HIGH_CONFIDENCE_LOW_RISK:
        emoji = "💎"

    title = f"{emoji} {signal.signal.value} — {ticker} ${signal.price:,.2f}"

    # Calculate dollar amounts for a concrete position
    capital = 100_000  # default; overridden by config
    position_value = capital * signal.position_size_pct
    risk_dollars = capital * 0.02  # 2% risk

    # Risk as price distance
    risk_per_share = abs(signal.price - signal.stop_loss)
    reward_per_share = abs(signal.take_profit - signal.price)

    # ── Build the action plan ──────────────────────────────────────────
    body_lines = []

    # Big clear action at the top
    body_lines.append("=" * 40)
    if is_buy:
        strength = "STRONG " if "STRONG" in signal.signal.value else ""
        body_lines.append(f"ACTION: {strength}BUY {ticker} (go LONG)")
        body_lines.append(f"")
        body_lines.append(f"WHAT TO DO:")
        body_lines.append(f"  1. BUY {ticker} near ${signal.price:,.2f}")
        body_lines.append(f"  2. Set STOP LOSS at ${signal.stop_loss:,.2f}")
        body_lines.append(f"     (${risk_per_share:,.2f} below entry)")
        body_lines.append(f"  3. Set TAKE PROFIT at ${signal.take_profit:,.2f}")
        body_lines.append(f"     (${reward_per_share:,.2f} above entry)")
        body_lines.append(f"  4. Position size: {signal.position_size_pct*100:.1f}% of capital")
        body_lines.append(f"")
        body_lines.append(f"  If price drops to ${signal.stop_loss:,.2f} -> GET OUT")
        body_lines.append(f"  If price hits ${signal.take_profit:,.2f} -> TAKE PROFIT")
    else:
        strength = "STRONG " if "STRONG" in signal.signal.value else ""
        body_lines.append(f"ACTION: {strength}SHORT {ticker} (bet on drop)")
        body_lines.append(f"")
        body_lines.append(f"WHAT TO DO:")
        body_lines.append(f"  1. SHORT/SELL {ticker} near ${signal.price:,.2f}")
        body_lines.append(f"     (or buy inverse ETF like SQQQ)")
        body_lines.append(f"  2. Set STOP LOSS at ${signal.stop_loss:,.2f}")
        body_lines.append(f"     (${risk_per_share:,.2f} above entry)")
        body_lines.append(f"  3. Set TAKE PROFIT at ${signal.take_profit:,.2f}")
        body_lines.append(f"     (${reward_per_share:,.2f} below entry)")
        body_lines.append(f"  4. Position size: {signal.position_size_pct*100:.1f}% of capital")
        body_lines.append(f"")
        body_lines.append(f"  If price rises to ${signal.stop_loss:,.2f} -> GET OUT")
        body_lines.append(f"  If price drops to ${signal.take_profit:,.2f} -> TAKE PROFIT")

    body_lines.append("=" * 40)

    # Confidence and context
    body_lines.append("")
    conv_label = "HIGH" if signal.conviction >= 0.8 else "MODERATE" if signal.conviction >= 0.6 else "LEAN"
    body_lines.append(f"Confidence: {conv_label} ({signal.conviction:.0%})")
    body_lines.append(f"Setup Type: {signal.context.value}")
    body_lines.append(f"Risk/Reward: {signal.risk_reward:.1f}x  (risking $1 to make ${signal.risk_reward:.1f})")
    body_lines.append(f"Market:     {signal.regime.value}")
    body_lines.append(f"Zone:       {signal.premium_discount_zone or 'N/A'}")

    # Why
    if signal.smart_money_reasons:
        body_lines.append("")
        body_lines.append("WHY (Smart Money):")
        for r in signal.smart_money_reasons[:3]:
            body_lines.append(f"  {r}")

    if signal.reasons:
        body_lines.append("")
        body_lines.append("WHY (Technical):")
        for r in signal.reasons[:3]:
            body_lines.append(f"  {r}")

    # Warnings
    if signal.warnings:
        body_lines.append("")
        body_lines.append("CAUTION:")
        for w in signal.warnings[:2]:
            body_lines.append(f"  {w}")

    body_lines.append("")
    body_lines.append("Not financial advice. Always do your own research.")

    body = "\n".join(body_lines)

    return {"title": title, "body": body, "direction": direction, "emoji": emoji}


# ── Notification Channels ──────────────────────────────────────────────────

def send_desktop_notification(title: str, body: str):
    """Windows toast notification."""
    try:
        # Try Windows-native approach
        if sys.platform == "win32":
            try:
                from win11toast import toast
                toast(title, body[:200])
                return True
            except ImportError:
                pass

            # Fallback: PowerShell toast
            import subprocess
            ps_cmd = f'''
            [Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
            $template = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]::ToastText02)
            $text = $template.GetElementsByTagName("text")
            $text[0].AppendChild($template.CreateTextNode("{title[:60]}")) | Out-Null
            $text[1].AppendChild($template.CreateTextNode("{body[:120]}")) | Out-Null
            $toast = [Windows.UI.Notifications.ToastNotification]::new($template)
            [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("Veteran Trader").Show($toast)
            '''
            subprocess.run(["powershell", "-Command", ps_cmd],
                          capture_output=True, timeout=10)
            return True

    except Exception as e:
        print(f"    Desktop notification failed: {e}")
        # Silent fallback — just print to console
    return False


def send_discord_alert(webhook_url: str, alert: dict):
    """Send alert to Discord via webhook."""
    if not webhook_url:
        return False

    # Normalize old discordapp.com URLs to discord.com
    webhook_url = webhook_url.replace("discordapp.com", "discord.com")

    is_buy = "BUY" in alert.get("direction", "")
    color = 0x22c55e if is_buy else 0xef4444
    if alert["emoji"] == "💎":
        color = 0xa78bfa

    payload = {
        "embeds": [{
            "title": alert["title"],
            "description": f"```\n{alert['body']}\n```",
            "color": color,
            "footer": {"text": "Veteran Trader v2 | Not financial advice"},
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }]
    }

    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            webhook_url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "VeteranTrader/2.0",
            },
            method="POST",
        )
        urllib.request.urlopen(req, timeout=10)
        print(f"    Discord alert sent")
        return True
    except Exception as e:
        print(f"    Discord alert failed: {e}")
        return False


def send_telegram_alert(bot_token: str, chat_id: str, alert: dict):
    """Send alert to Telegram."""
    if not bot_token or not chat_id:
        return False

    text = f"*{alert['title']}*\n\n```\n{alert['body']}\n```"

    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown",
    }

    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=10)
        print(f"    Telegram alert sent")
        return True
    except Exception as e:
        print(f"    Telegram alert failed: {e}")
        return False


# ── Main Alert Runner ──────────────────────────────────────────────────────

def run_alerts(csv_path: str, alert_config: dict, smt_path: str = None):
    """Run the trader and send alerts for the latest signal."""

    profile = RiskProfile(alert_config.get("risk_profile", "moderate"))
    capital = alert_config.get("capital", 100000)

    config = TraderConfig(risk_profile=profile, starting_capital=capital)
    config.adjust_for_profile()

    data = load_data(csv_path)
    smt_data = None
    if smt_path and os.path.exists(smt_path):
        smt_data = load_data(smt_path)

    trader = VeteranTrader(data, config, smt_data)
    signals = trader.analyze()

    if not signals:
        print("  No signals generated.")
        return

    latest = signals[-1]
    ticker = os.path.basename(csv_path).replace("_data.csv", "").upper()

    print()
    print(f"  Latest signal for {ticker}:")
    print(f"    {latest.signal.value} | {latest.context.value} | conv: {latest.conviction:.2f}")

    if not should_alert(latest, alert_config):
        print(f"    Signal does not meet alert criteria — no notification sent.")
        return

    alert = format_alert(latest, ticker)

    # Console alert (always)
    print()
    print("  " + "!" * 60)
    print(f"  ALERT: {alert['title']}")
    print("  " + "!" * 60)
    for line in alert["body"].split("\n"):
        print(f"    {line}")
    print()

    # Desktop notification
    if alert_config.get("desktop_notifications", True):
        send_desktop_notification(alert["title"], alert["body"])

    # Discord — env var takes priority over config file
    discord_url = os.environ.get("DISCORD_WEBHOOK_URL") or alert_config.get("discord_webhook_url", "")
    if discord_url:
        send_discord_alert(discord_url, alert)

    # Telegram
    tg_token = alert_config.get("telegram_bot_token", "")
    tg_chat = alert_config.get("telegram_chat_id", "")
    if tg_token and tg_chat:
        send_telegram_alert(tg_token, tg_chat, alert)


def main():
    """Run alerts on one or more CSV files."""
    import argparse

    parser = argparse.ArgumentParser(description="Send trade alerts from Veteran Trader v2")
    parser.add_argument("datafiles", nargs="*", help="OHLCV CSV files to analyze")
    parser.add_argument("--smt-compare", default=None)
    parser.add_argument("--test", action="store_true", help="Send a test notification")
    args = parser.parse_args()

    config = load_alert_config()

    if args.test:
        print("  Sending test notification...")
        send_desktop_notification(
            "Veteran Trader — Test",
            "If you see this, desktop notifications are working!"
        )
        discord_url = os.environ.get("DISCORD_WEBHOOK_URL") or config.get("discord_webhook_url", "")
        if discord_url:
            send_discord_alert(discord_url, {
                "title": "Test Alert",
                "body": "Veteran Trader notifications are connected!",
                "direction": "BUY", "emoji": "🟢",
            })
        tg_token = config.get("telegram_bot_token", "")
        tg_chat = config.get("telegram_chat_id", "")
        if tg_token and tg_chat:
            send_telegram_alert(tg_token, tg_chat, {
                "title": "Test Alert",
                "body": "Veteran Trader notifications are connected!",
                "direction": "BUY", "emoji": "🟢",
            })
        print("  Test complete.")
        return

    if not args.datafiles:
        # Use config tickers
        print("  No CSV files provided — use: python alerts.py QQQ_data.csv")
        return

    for csv_path in args.datafiles:
        run_alerts(csv_path, config, args.smt_compare)


if __name__ == "__main__":
    main()