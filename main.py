"""
Crypto Trading Signal Bot
- Scans 20 coins every 15 minutes via CoinGecko free API
- Calculates RSI(14) from 3-day hourly OHLCV
- Sends RSI triggers to Gemini AI with news for BUY/SELL/NO TRADE decisions
- Sends signals + TP/SL alerts to Telegram
- Tracks active trades with +3% TP / -1.5% SL
"""

import os
import time
import json
import logging
import threading
from datetime import datetime, timezone

import requests
import google.generativeai as genai

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
GEMINI_API_KEY      = os.environ["GEMINI_API_KEY"]
TELEGRAM_BOT_TOKEN  = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID    = os.environ["TELEGRAM_CHAT_ID"]
NEWS_API_KEY        = os.environ["NEWS_API_KEY"]

COINGECKO_BASE      = "https://api.coingecko.com/api/v3"
COINGECKO_API_KEY   = os.environ.get("COINGECKO_API_KEY", "")  # optional demo key

SCAN_INTERVAL_SEC   = 15 * 60          # 15 minutes
TRADE_CHECK_SEC     = 60               # 1 minute
COIN_DELAY_SEC      = 3               # delay between each coin to respect rate limits
RETRY_AFTER_429_SEC = 60

TP_PCT              = 3.0              # take-profit %
SL_PCT              = 1.5              # stop-loss %
RSI_OVERSOLD        = 40
RSI_OVERBOUGHT      = 60
RSI_PERIOD          = 14
MIN_CONFIDENCE      = 65              # Gemini must be >= this to act

COINS = [
    "bitcoin", "ethereum", "binancecoin", "solana", "ripple",
    "cardano", "avalanche-2", "polkadot", "chainlink", "dogecoin",
    "shiba-inu", "matic-network", "litecoin", "uniswap", "cosmos",
    "stellar", "monero", "tron", "ethereum-classic", "filecoin",
]

# symbol lookup (used in messages)
COIN_SYMBOLS = {
    "bitcoin": "BTC", "ethereum": "ETH", "binancecoin": "BNB",
    "solana": "SOL", "ripple": "XRP", "cardano": "ADA",
    "avalanche-2": "AVAX", "polkadot": "DOT", "chainlink": "LINK",
    "dogecoin": "DOGE", "shiba-inu": "SHIB", "matic-network": "MATIC",
    "litecoin": "LTC", "uniswap": "UNI", "cosmos": "ATOM",
    "stellar": "XLM", "monero": "XMR", "tron": "TRX",
    "ethereum-classic": "ETC", "filecoin": "FIL",
}

# ─────────────────────────────────────────────
# Gemini setup
# ─────────────────────────────────────────────
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

# ─────────────────────────────────────────────
# Active trades store  {coin_id: trade_dict}
# ─────────────────────────────────────────────
active_trades: dict = {}
trades_lock = threading.Lock()

# ─────────────────────────────────────────────
# Helpers – HTTP with retry on 429
# ─────────────────────────────────────────────

def _get(url: str, params: dict = None, headers: dict = None, retries: int = 3) -> dict | None:
    """GET with automatic 429 back-off."""
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=20)
            if resp.status_code == 429:
                log.warning("[RATE_LIMIT] 429 on %s – waiting %ss (attempt %d/%d)",
                            url, RETRY_AFTER_429_SEC, attempt, retries)
                time.sleep(RETRY_AFTER_429_SEC)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            log.error("[HTTP_ERROR] %s attempt %d/%d – %s", url, attempt, retries, exc)
            if attempt < retries:
                time.sleep(5)
    return None

# ─────────────────────────────────────────────
# CoinGecko
# ─────────────────────────────────────────────

def cg_headers() -> dict:
    h = {"accept": "application/json"}
    if COINGECKO_API_KEY:
        h["x-cg-demo-api-key"] = COINGECKO_API_KEY
    return h


def fetch_current_price(coin_id: str) -> float | None:
    url = f"{COINGECKO_BASE}/simple/price"
    data = _get(url, params={"ids": coin_id, "vs_currencies": "usd"}, headers=cg_headers())
    if data and coin_id in data:
        return data[coin_id]["usd"]
    return None


def fetch_ohlcv_hourly(coin_id: str) -> list[float] | None:
    """
    Returns list of hourly CLOSE prices for the last ~3 days (72 hours).
    CoinGecko /coins/{id}/ohlc with days=3 returns 4-hour candles for free tier.
    We fall back to /market_chart for hourly closes.
    """
    url = f"{COINGECKO_BASE}/coins/{coin_id}/market_chart"
    data = _get(
        url,
        params={"vs_currency": "usd", "days": 3, "interval": "hourly"},
        headers=cg_headers(),
    )
    if not data or "prices" not in data:
        return None
    # prices = [[timestamp_ms, price], ...]
    closes = [p[1] for p in data["prices"]]
    return closes

# ─────────────────────────────────────────────
# RSI calculation
# ─────────────────────────────────────────────

def compute_rsi(closes: list[float], period: int = RSI_PERIOD) -> float | None:
    if len(closes) < period + 1:
        return None
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [d if d > 0 else 0.0 for d in deltas]
    losses = [-d if d < 0 else 0.0 for d in deltas]

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1 + rs))

# ─────────────────────────────────────────────
# NewsAPI
# ─────────────────────────────────────────────

def fetch_news_headlines(query: str, max_articles: int = 5) -> list[str]:
    url = "https://newsapi.org/v2/everything"
    try:
        resp = requests.get(
            url,
            params={
                "q": query,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": max_articles,
                "apiKey": NEWS_API_KEY,
            },
            timeout=15,
        )
        resp.raise_for_status()
        articles = resp.json().get("articles", [])
        return [a["title"] for a in articles if a.get("title")]
    except Exception as exc:
        log.error("[NEWS_ERROR] %s", exc)
        return []

# ─────────────────────────────────────────────
# Gemini AI analysis
# ─────────────────────────────────────────────

def ask_gemini(coin_id: str, symbol: str, price: float, rsi: float, headlines: list[str]) -> dict | None:
    """
    Returns dict: {"action": "BUY"|"SELL"|"NO TRADE", "confidence": int, "reason": str}
    or None on failure.
    """
    news_block = "\n".join(f"- {h}" for h in headlines) if headlines else "No recent news found."
    prompt = f"""You are a professional crypto trading analyst. Analyze the following data and provide a trading signal.

Coin: {symbol} ({coin_id})
Current Price: ${price:,.4f}
RSI(14): {rsi:.2f}

Recent News Headlines:
{news_block}

Based on this data, provide:
1. ACTION: Must be exactly one of: BUY, SELL, or NO TRADE
2. CONFIDENCE: Integer from 0 to 100 representing your confidence percentage
3. REASON: One concise sentence explaining your decision

Respond ONLY in this exact JSON format (no markdown, no extra text):
{{"action": "BUY", "confidence": 75, "reason": "RSI oversold with positive sentiment"}}"""

    try:
        response = gemini_model.generate_content(prompt)
        text = response.text.strip()
        # Strip any accidental markdown fences
        text = text.replace("```json", "").replace("```", "").strip()
        result = json.loads(text)
        # Validate
        action = result.get("action", "").upper()
        if action not in ("BUY", "SELL", "NO TRADE"):
            log.warning("[GEMINI] Invalid action '%s' for %s", action, symbol)
            return None
        result["action"] = action
        result["confidence"] = int(result.get("confidence", 0))
        return result
    except json.JSONDecodeError as exc:
        log.error("[GEMINI_PARSE] %s – raw: %s", exc, response.text[:200] if 'response' in dir() else "n/a")
        return None
    except Exception as exc:
        log.error("[GEMINI_ERROR] %s", exc)
        return None

# ─────────────────────────────────────────────
# Telegram
# ─────────────────────────────────────────────

def send_telegram(message: str) -> bool:
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        resp = requests.post(
            url,
            json={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"},
            timeout=15,
        )
        resp.raise_for_status()
        return True
    except Exception as exc:
        log.error("[TELEGRAM_ERROR] %s", exc)
        return False


def fmt_signal_msg(action: str, symbol: str, coin_id: str, price: float,
                   rsi: float, confidence: int, reason: str) -> str:
    emoji = "🟢" if action == "BUY" else "🔴"
    tp = price * (1 + TP_PCT / 100)
    sl = price * (1 - SL_PCT / 100)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return (
        f"{emoji} <b>{action} SIGNAL – {symbol}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"💰 Entry Price: <b>${price:,.4f}</b>\n"
        f"🎯 Take Profit: <b>${tp:,.4f}</b> (+{TP_PCT}%)\n"
        f"🛑 Stop Loss:   <b>${sl:,.4f}</b> (-{SL_PCT}%)\n"
        f"📊 RSI(14): {rsi:.2f}\n"
        f"🤖 Confidence: {confidence}%\n"
        f"💡 {reason}\n"
        f"🕐 {now}"
    )


def fmt_tp_msg(symbol: str, entry: float, exit_price: float, pct: float) -> str:
    return (
        f"✅ <b>TAKE PROFIT HIT – {symbol}</b>\n"
        f"Entry: ${entry:,.4f} → Exit: ${exit_price:,.4f}\n"
        f"Gain: +{pct:.2f}% 🎉"
    )


def fmt_sl_msg(symbol: str, entry: float, exit_price: float, pct: float) -> str:
    return (
        f"❌ <b>STOP LOSS HIT – {symbol}</b>\n"
        f"Entry: ${entry:,.4f} → Exit: ${exit_price:,.4f}\n"
        f"Loss: -{abs(pct):.2f}%"
    )

# ─────────────────────────────────────────────
# Trade management
# ─────────────────────────────────────────────

def open_trade(coin_id: str, symbol: str, action: str, entry_price: float):
    tp = entry_price * (1 + TP_PCT / 100)
    sl = entry_price * (1 - SL_PCT / 100)
    with trades_lock:
        active_trades[coin_id] = {
            "symbol": symbol,
            "action": action,
            "entry": entry_price,
            "tp": tp,
            "sl": sl,
            "opened_at": datetime.now(timezone.utc).isoformat(),
        }
    log.info("[TRADE_OPEN] %s %s @ $%.4f | TP=$%.4f SL=$%.4f",
             action, symbol, entry_price, tp, sl)


def check_trades():
    """Called every minute. Checks TP/SL for all active trades."""
    with trades_lock:
        coin_ids = list(active_trades.keys())

    for coin_id in coin_ids:
        try:
            price = fetch_current_price(coin_id)
            if price is None:
                continue

            with trades_lock:
                trade = active_trades.get(coin_id)
            if not trade:
                continue

            symbol = trade["symbol"]
            entry  = trade["entry"]
            tp     = trade["tp"]
            sl     = trade["sl"]
            action = trade["action"]

            hit_tp = (action == "BUY" and price >= tp) or (action == "SELL" and price <= tp)
            hit_sl = (action == "BUY" and price <= sl) or (action == "SELL" and price >= sl)

            if hit_tp:
                pct = ((price - entry) / entry) * 100 if action == "BUY" else ((entry - price) / entry) * 100
                msg = fmt_tp_msg(symbol, entry, price, pct)
                send_telegram(msg)
                log.info("[TP] %s | entry=$%.4f exit=$%.4f gain=+%.2f%%", symbol, entry, price, pct)
                with trades_lock:
                    active_trades.pop(coin_id, None)

            elif hit_sl:
                pct = ((entry - price) / entry) * 100 if action == "BUY" else ((price - entry) / entry) * 100
                msg = fmt_sl_msg(symbol, entry, price, pct)
                send_telegram(msg)
                log.info("[SL] %s | entry=$%.4f exit=$%.4f loss=-%.2f%%", symbol, entry, price, pct)
                with trades_lock:
                    active_trades.pop(coin_id, None)

        except Exception as exc:
            log.error("[CHECK_TRADE_ERROR] %s – %s", coin_id, exc)


# ─────────────────────────────────────────────
# Trade monitor thread (runs independently)
# ─────────────────────────────────────────────

def trade_monitor_loop():
    log.info("[MONITOR] Trade monitor thread started (checking every %ds)", TRADE_CHECK_SEC)
    while True:
        try:
            check_trades()
        except Exception as exc:
            log.error("[MONITOR_ERROR] %s", exc)
        time.sleep(TRADE_CHECK_SEC)

# ─────────────────────────────────────────────
# Main scan loop
# ─────────────────────────────────────────────

def scan_coin(coin_id: str):
    symbol = COIN_SYMBOLS.get(coin_id, coin_id.upper())

    # 1. Fetch OHLCV
    closes = fetch_ohlcv_hourly(coin_id)
    if not closes or len(closes) < RSI_PERIOD + 2:
        log.info("[SKIP] %s – insufficient price data (%s candles)",
                 symbol, len(closes) if closes else 0)
        return

    # 2. RSI
    rsi = compute_rsi(closes)
    if rsi is None:
        log.info("[SKIP] %s – RSI calculation failed", symbol)
        return

    # 3. Check trigger
    if RSI_OVERSOLD < rsi < RSI_OVERBOUGHT:
        log.info("[SKIP] %s RSI=%.2f – no trigger (%.0f–%.0f range)",
                 symbol, rsi, RSI_OVERSOLD, RSI_OVERBOUGHT)
        return

    # 4. Already in an active trade for this coin?
    with trades_lock:
        already_trading = coin_id in active_trades
    if already_trading:
        log.info("[SKIP] %s – already in active trade", symbol)
        return

    log.info("[SCAN] %s RSI=%.2f – TRIGGERED (< %d or > %d)",
             symbol, rsi, RSI_OVERSOLD, RSI_OVERBOUGHT)

    # 5. Current price
    price = fetch_current_price(coin_id)
    if price is None:
        log.warning("[SKIP] %s – could not fetch current price", symbol)
        return

    # 6. News
    headlines = fetch_news_headlines(f"{symbol} crypto")
    log.info("[SCAN] %s – fetched %d news headlines", symbol, len(headlines))

    # 7. Gemini
    ai_result = ask_gemini(coin_id, symbol, price, rsi, headlines)
    if ai_result is None:
        log.warning("[SKIP] %s – Gemini returned no valid result", symbol)
        return

    action     = ai_result["action"]
    confidence = ai_result["confidence"]
    reason     = ai_result.get("reason", "N/A")

    log.info("[SIGNAL] %s → %s (confidence=%d%%) | %s", symbol, action, confidence, reason)

    # 8. Filter by confidence
    if action == "NO TRADE" or confidence < MIN_CONFIDENCE:
        log.info("[SKIP] %s – action=%s confidence=%d%% (min=%d%%)",
                 symbol, action, confidence, MIN_CONFIDENCE)
        return

    # 9. Send Telegram
    msg = fmt_signal_msg(action, symbol, coin_id, price, rsi, confidence, reason)
    sent = send_telegram(msg)
    if sent:
        log.info("[SIGNAL] %s – Telegram message sent", symbol)

    # 10. Open trade tracker
    open_trade(coin_id, symbol, action, price)


def scan_all_coins():
    log.info("[SCAN] ── Starting full scan of %d coins ──", len(COINS))
    for coin_id in COINS:
        try:
            scan_coin(coin_id)
        except Exception as exc:
            log.error("[SCAN_ERROR] %s – %s", coin_id, exc)
        time.sleep(COIN_DELAY_SEC)
    log.info("[SCAN] ── Scan complete ──")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("  Crypto Signal Bot starting up")
    log.info("  Coins: %d | Interval: %dm | TP: +%.1f%% SL: -%.1f%%",
             len(COINS), SCAN_INTERVAL_SEC // 60, TP_PCT, SL_PCT)
    log.info("=" * 60)

    # Verify env vars
    for var in ("GEMINI_API_KEY", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "NEWS_API_KEY"):
        if not os.environ.get(var):
            raise EnvironmentError(f"Missing required environment variable: {var}")

    # Send startup notification
    send_telegram(
        "🤖 <b>Crypto Signal Bot Started</b>\n"
        f"Scanning {len(COINS)} coins every {SCAN_INTERVAL_SEC // 60} minutes.\n"
        f"TP: +{TP_PCT}% | SL: -{SL_PCT}% | Min confidence: {MIN_CONFIDENCE}%"
    )

    # Start trade monitor in background
    monitor_thread = threading.Thread(target=trade_monitor_loop, daemon=True)
    monitor_thread.start()

    # Main scan loop – must never crash
    while True:
        try:
            scan_all_coins()
        except Exception as exc:
            log.error("[MAIN_LOOP_ERROR] Unexpected error: %s", exc)

        log.info("[MAIN] Sleeping %d seconds until next scan…", SCAN_INTERVAL_SEC)
        time.sleep(SCAN_INTERVAL_SEC)


if __name__ == "__main__":
    main()
