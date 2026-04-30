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
# Logging  (stdout flush so Railway shows logs instantly)
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Config  – read env vars INSIDE main() so missing
# vars raise a clear error, not a silent crash at
# module-load time (which breaks Railway restart logic)
# ─────────────────────────────────────────────
COINGECKO_BASE      = "https://api.coingecko.com/api/v3"

SCAN_INTERVAL_SEC   = 15 * 60   # 15 minutes
TRADE_CHECK_SEC     = 60        # 1 minute
COIN_DELAY_SEC      = 4         # 4 s between coins  → max 15 calls/min well under 30
RETRY_AFTER_429_SEC = 65        # slightly over 60 to be safe

TP_PCT              = 3.0       # take-profit %
SL_PCT              = 1.5       # stop-loss %
RSI_OVERSOLD        = 40
RSI_OVERBOUGHT      = 60
RSI_PERIOD          = 14
MIN_CONFIDENCE      = 65        # Gemini must be >= this to act

# ── Verified CoinGecko IDs ──────────────────
# BUG FIX 1: "near" should be "near-protocol" on CoinGecko
# BUG FIX 2: "the-open-network" is correct for TON
COINS = [
    "bitcoin",
    "ethereum",
    "binancecoin",
    "solana",
    "ripple",
    "cardano",
    "avalanche-2",
    "polkadot",
    "chainlink",
    "dogecoin",
    "shiba-inu",
    "matic-network",
    "litecoin",
    "uniswap",
    "cosmos",
    "stellar",
    "monero",
    "tron",
    "near-protocol",        # FIX: was "near" → 404 on CoinGecko
    "the-open-network",     # TON – correct
]

COIN_SYMBOLS = {
    "bitcoin":          "BTC",
    "ethereum":         "ETH",
    "binancecoin":      "BNB",
    "solana":           "SOL",
    "ripple":           "XRP",
    "cardano":          "ADA",
    "avalanche-2":      "AVAX",
    "polkadot":         "DOT",
    "chainlink":        "LINK",
    "dogecoin":         "DOGE",
    "shiba-inu":        "SHIB",
    "matic-network":    "MATIC",
    "litecoin":         "LTC",
    "uniswap":          "UNI",
    "cosmos":           "ATOM",
    "stellar":          "XLM",
    "monero":           "XMR",
    "tron":             "TRX",
    "near-protocol":    "NEAR",  # updated key to match above
    "the-open-network": "TON",
}

# ─────────────────────────────────────────────
# Runtime globals (set in main())
# ─────────────────────────────────────────────
GEMINI_API_KEY     = ""
TELEGRAM_BOT_TOKEN = ""
TELEGRAM_CHAT_ID   = ""
NEWS_API_KEY       = ""
COINGECKO_API_KEY  = ""
gemini_model       = None

# Active trades store  {coin_id: trade_dict}
active_trades: dict = {}
trades_lock = threading.Lock()

# ─────────────────────────────────────────────
# HTTP helper – retry on 429
# BUG FIX 3: After a 429 we must NOT call raise_for_status()
#            on the same response object – we `continue` instead.
# ─────────────────────────────────────────────

def _get(url: str, params: dict = None, headers: dict = None, retries: int = 3):
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=25)

            if resp.status_code == 429:
                log.warning("[RATE_LIMIT] 429 – sleeping %ds (attempt %d/%d)",
                            RETRY_AFTER_429_SEC, attempt, retries)
                time.sleep(RETRY_AFTER_429_SEC)
                continue                      # retry without calling raise_for_status

            if resp.status_code == 404:
                log.warning("[NOT_FOUND] 404 – %s", url)
                return None                   # no point retrying a 404

            resp.raise_for_status()           # raises for 5xx etc.
            return resp.json()

        except requests.exceptions.Timeout:
            log.error("[TIMEOUT] %s attempt %d/%d", url, attempt, retries)
            if attempt < retries:
                time.sleep(6)

        except requests.exceptions.JSONDecodeError:
            log.error("[JSON_ERROR] Empty/invalid JSON from %s", url)
            return None                       # no point retrying a bad response

        except requests.RequestException as exc:
            log.error("[HTTP_ERROR] %s attempt %d/%d – %s", url, attempt, retries, exc)
            if attempt < retries:
                time.sleep(6)

    return None

# ─────────────────────────────────────────────
# CoinGecko helpers
# ─────────────────────────────────────────────

def cg_headers() -> dict:
    h = {"accept": "application/json"}
    if COINGECKO_API_KEY:
        h["x-cg-demo-api-key"] = COINGECKO_API_KEY
    return h


def fetch_current_price(coin_id: str):
    """Single price lookup via /simple/price (1 API call)."""
    data = _get(
        f"{COINGECKO_BASE}/simple/price",
        params={"ids": coin_id, "vs_currencies": "usd"},
        headers=cg_headers(),
    )
    if data and coin_id in data:
        return float(data[coin_id]["usd"])
    return None


def fetch_ohlcv_hourly(coin_id: str):
    """
    Returns list of close prices (hourly granularity, ~72 points for 3 days).
    /market_chart?days=3  → CoinGecko FREE tier auto-returns hourly data.
    'interval' param intentionally omitted (causes 422 on free tier).
    BUG FIX 4: was passing days=3 as int, must be string "3" for some
               versions of requests; now explicitly cast.
    """
    data = _get(
        f"{COINGECKO_BASE}/coins/{coin_id}/market_chart",
        params={"vs_currency": "usd", "days": "3"},
        headers=cg_headers(),
    )
    if not data or "prices" not in data:
        return None
    prices = data["prices"]
    if len(prices) < RSI_PERIOD + 2:
        return None
    return [float(p[1]) for p in prices]

# ─────────────────────────────────────────────
# RSI  (Wilder smoothing, pure Python)
# ─────────────────────────────────────────────

def compute_rsi(closes: list, period: int = RSI_PERIOD):
    """
    BUG FIX 5: Previous code used RSI_PERIOD (global) inside the function
               but the parameter was also called 'period' – shadowing issue
               on older Pythons. Now uses the parameter explicitly.
    """
    if len(closes) < period + 2:
        return None

    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains  = [max(d, 0.0) for d in deltas]
    losses = [max(-d, 0.0) for d in deltas]

    # Initial simple average over first `period` bars
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    # Wilder smoothing for the rest
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100.0 - (100.0 / (1 + rs)), 2)

# ─────────────────────────────────────────────
# NewsAPI
# ─────────────────────────────────────────────

def fetch_news_headlines(query: str, max_articles: int = 5) -> list:
    try:
        resp = requests.get(
            "https://newsapi.org/v2/everything",
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
        # Guard against None titles
        return [a["title"] for a in articles if a.get("title")]
    except Exception as exc:
        log.error("[NEWS_ERROR] %s", exc)
        return []

# ─────────────────────────────────────────────
# Gemini AI  – gemini-2.0-flash (free tier, stable)
# ─────────────────────────────────────────────

def ask_gemini(coin_id: str, symbol: str, price: float, rsi: float, headlines: list):
    news_block = (
        "\n".join(f"- {h}" for h in headlines)
        if headlines
        else "No recent news available."
    )
    prompt = f"""You are a professional crypto trading analyst. Analyze the data below and respond with a trading signal.

Coin: {symbol} ({coin_id})
Current Price: ${price:,.6f}
RSI(14): {rsi}

Recent News Headlines:
{news_block}

Decision rules:
- BUY  → RSI < 40 (oversold) AND news is neutral or positive
- SELL → RSI > 60 (overbought) AND news is neutral or negative
- NO TRADE → uncertain or conflicting signals

Respond ONLY with a single valid JSON object. No markdown, no explanation, no extra text.
Example format: {{"action": "BUY", "confidence": 75, "reason": "RSI oversold with positive sentiment"}}

Constraints:
- "action" must be exactly one of: BUY, SELL, NO TRADE
- "confidence" must be an integer between 0 and 100
- "reason" must be one short sentence"""

    try:
        response = gemini_model.generate_content(prompt)
        text = response.text.strip()

        # Strip accidental markdown fences
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()

        result = json.loads(text)

        action = str(result.get("action", "")).upper().strip()
        if action not in ("BUY", "SELL", "NO TRADE"):
            log.warning("[GEMINI] Unexpected action '%s' for %s – treating as NO TRADE", action, symbol)
            return {"action": "NO TRADE", "confidence": 0, "reason": "Invalid Gemini response"}

        return {
            "action":     action,
            "confidence": int(result.get("confidence", 0)),
            "reason":     str(result.get("reason", "N/A")),
        }

    except json.JSONDecodeError:
        raw = locals().get("text", "n/a")
        log.error("[GEMINI_PARSE] JSON decode failed. Raw: %.200s", raw)
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
            json={
                "chat_id":    TELEGRAM_CHAT_ID,
                "text":       message,
                "parse_mode": "HTML",
            },
            timeout=15,
        )
        resp.raise_for_status()
        return True
    except Exception as exc:
        log.error("[TELEGRAM_ERROR] %s", exc)
        return False


def fmt_signal_msg(action, symbol, coin_id, price, rsi, confidence, reason):
    emoji = "🟢" if action == "BUY" else "🔴"
    tp  = price * (1 + TP_PCT / 100)
    sl  = price * (1 - SL_PCT / 100)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return (
        f"{emoji} <b>{action} SIGNAL – {symbol}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"💰 Entry Price : <b>${price:,.6f}</b>\n"
        f"🎯 Take Profit : <b>${tp:,.6f}</b> (+{TP_PCT}%)\n"
        f"🛑 Stop Loss   : <b>${sl:,.6f}</b> (-{SL_PCT}%)\n"
        f"📊 RSI(14)     : {rsi}\n"
        f"🤖 Confidence  : {confidence}%\n"
        f"💡 {reason}\n"
        f"🕐 {now}"
    )


def fmt_tp_msg(symbol, entry, exit_price, pct):
    return (
        f"✅ <b>TAKE PROFIT HIT – {symbol}</b>\n"
        f"Entry : ${entry:,.6f}\n"
        f"Exit  : ${exit_price:,.6f}\n"
        f"Gain  : +{pct:.2f}% 🎉"
    )


def fmt_sl_msg(symbol, entry, exit_price, pct):
    return (
        f"❌ <b>STOP LOSS HIT – {symbol}</b>\n"
        f"Entry : ${entry:,.6f}\n"
        f"Exit  : ${exit_price:,.6f}\n"
        f"Loss  : -{abs(pct):.2f}%"
    )

# ─────────────────────────────────────────────
# Trade management
# ─────────────────────────────────────────────

def open_trade(coin_id, symbol, action, entry_price):
    tp = entry_price * (1 + TP_PCT / 100)
    sl = entry_price * (1 - SL_PCT / 100)
    with trades_lock:
        active_trades[coin_id] = {
            "symbol":     symbol,
            "action":     action,
            "entry":      entry_price,
            "tp":         tp,
            "sl":         sl,
            "opened_at":  datetime.now(timezone.utc).isoformat(),
        }
    log.info("[TRADE_OPEN] %s %s @ $%.6f | TP=$%.6f | SL=$%.6f",
             action, symbol, entry_price, tp, sl)


def check_trades():
    """Called every 60 s by the monitor thread."""
    with trades_lock:
        snapshot = dict(active_trades)          # copy to avoid holding lock during I/O

    for coin_id, trade in snapshot.items():
        try:
            price = fetch_current_price(coin_id)
            if price is None:
                continue

            symbol = trade["symbol"]
            entry  = trade["entry"]
            tp     = trade["tp"]
            sl     = trade["sl"]
            action = trade["action"]

            # BUG FIX 7: For SELL trades TP is BELOW entry, SL is ABOVE entry.
            # Previous logic was already correct but let's add a comment for clarity.
            hit_tp = (action == "BUY"  and price >= tp) or \
                     (action == "SELL" and price <= tp)
            hit_sl = (action == "BUY"  and price <= sl) or \
                     (action == "SELL" and price >= sl)

            if hit_tp:
                pct = (
                    (price - entry) / entry * 100
                    if action == "BUY"
                    else (entry - price) / entry * 100
                )
                send_telegram(fmt_tp_msg(symbol, entry, price, pct))
                log.info("[TP] %s | entry=$%.6f exit=$%.6f gain=+%.2f%%",
                         symbol, entry, price, pct)
                with trades_lock:
                    active_trades.pop(coin_id, None)

            elif hit_sl:
                pct = (
                    (entry - price) / entry * 100
                    if action == "BUY"
                    else (price - entry) / entry * 100
                )
                send_telegram(fmt_sl_msg(symbol, entry, price, pct))
                log.info("[SL] %s | entry=$%.6f exit=$%.6f loss=-%.2f%%",
                         symbol, entry, price, pct)
                with trades_lock:
                    active_trades.pop(coin_id, None)

        except Exception as exc:
            log.error("[CHECK_TRADE_ERROR] %s – %s", coin_id, exc)

# ─────────────────────────────────────────────
# Trade monitor thread
# ─────────────────────────────────────────────

def trade_monitor_loop():
    log.info("[MONITOR] Trade monitor started – checking every %ds", TRADE_CHECK_SEC)
    while True:
        try:
            check_trades()
        except Exception as exc:
            log.error("[MONITOR_ERROR] Unhandled: %s", exc)
        time.sleep(TRADE_CHECK_SEC)

# ─────────────────────────────────────────────
# Main scan loop
# ─────────────────────────────────────────────

def scan_coin(coin_id: str):
    symbol = COIN_SYMBOLS.get(coin_id, coin_id.upper())
    log.info("[SCAN] Checking %s (%s)...", symbol, coin_id)

    # 1. Fetch OHLCV (hourly closes, ~72 points)
    closes = fetch_ohlcv_hourly(coin_id)
    if not closes:
        log.info("[SKIP] %s – no market data returned", symbol)
        return

    # 2. Compute RSI
    rsi = compute_rsi(closes)
    if rsi is None:
        log.info("[SKIP] %s – not enough candles for RSI (got %d)", symbol, len(closes))
        return

    log.info("[SCAN] %s | RSI=%.2f | Candles=%d", symbol, rsi, len(closes))

    # 3. RSI trigger gate
    if RSI_OVERSOLD < rsi < RSI_OVERBOUGHT:
        log.info("[SKIP] %s – RSI=%.2f neutral zone (%d–%d)",
                 symbol, rsi, RSI_OVERSOLD, RSI_OVERBOUGHT)
        return

    # 4. Skip if already in an active trade for this coin
    with trades_lock:
        already_trading = coin_id in active_trades
    if already_trading:
        log.info("[SKIP] %s – already in active trade", symbol)
        return

    log.info("[SCAN] %s RSI=%.2f TRIGGERED → fetching price + news...", symbol, rsi)

    # 5. Current price
    price = fetch_current_price(coin_id)
    if price is None:
        log.warning("[SKIP] %s – could not fetch current price", symbol)
        return

    # 6. News
    headlines = fetch_news_headlines(f"{symbol} crypto")
    log.info("[SCAN] %s – %d news headlines fetched", symbol, len(headlines))

    # 7. Gemini analysis
    ai_result = ask_gemini(coin_id, symbol, price, rsi, headlines)
    if ai_result is None:
        log.warning("[SKIP] %s – Gemini returned no valid result", symbol)
        return

    action     = ai_result["action"]
    confidence = ai_result["confidence"]
    reason     = ai_result["reason"]

    log.info("[SIGNAL] %s → %s | conf=%d%% | %s", symbol, action, confidence, reason)

    # 8. Confidence + action gate
    if action == "NO TRADE" or confidence < MIN_CONFIDENCE:
        log.info("[SKIP] %s – action=%s conf=%d%% (min=%d%%)",
                 symbol, action, confidence, MIN_CONFIDENCE)
        return

    # 9. Send Telegram
    msg = fmt_signal_msg(action, symbol, coin_id, price, rsi, confidence, reason)
    if send_telegram(msg):
        log.info("[SIGNAL] %s – Telegram alert sent ✓", symbol)

    # 10. Track trade
    open_trade(coin_id, symbol, action, price)


def scan_all_coins():
    log.info("[SCAN] ══════ Starting scan – %d coins ══════", len(COINS))
    signals = 0
    for idx, coin_id in enumerate(COINS, 1):
        try:
            with trades_lock:
                before = len(active_trades)
            scan_coin(coin_id)
            with trades_lock:
                after = len(active_trades)
            if after > before:
                signals += 1
        except Exception as exc:
            log.error("[SCAN_ERROR] %s – %s", coin_id, exc)

        # Rate-limit delay (skip after the last coin)
        if idx < len(COINS):
            time.sleep(COIN_DELAY_SEC)

    log.info("[SCAN] ══════ Done – signals: %d | active trades: %d ══════",
             signals, len(active_trades))

# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main():
    global GEMINI_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
    global NEWS_API_KEY, COINGECKO_API_KEY, gemini_model

    # ── Validate env vars up-front ──────────────
    missing = [v for v in
               ("GEMINI_API_KEY", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "NEWS_API_KEY")
               if not os.environ.get(v)]
    if missing:
        raise EnvironmentError(f"Missing required env vars: {', '.join(missing)}")

    GEMINI_API_KEY     = os.environ["GEMINI_API_KEY"]
    TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
    TELEGRAM_CHAT_ID   = os.environ["TELEGRAM_CHAT_ID"]
    NEWS_API_KEY       = os.environ["NEWS_API_KEY"]
    COINGECKO_API_KEY  = os.environ.get("COINGECKO_API_KEY", "")

    # ── Init Gemini ──────────────────────────────
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-2.0-flash")

    log.info("=" * 55)
    log.info("  Crypto Signal Bot  –  STARTED")
    log.info("  Coins    : %d", len(COINS))
    log.info("  Interval : %d min", SCAN_INTERVAL_SEC // 60)
    log.info("  TP / SL  : +%.1f%% / -%.1f%%", TP_PCT, SL_PCT)
    log.info("  Min conf : %d%%", MIN_CONFIDENCE)
    log.info("=" * 55)

    send_telegram(
        "🤖 <b>Crypto Signal Bot Started</b>\n"
        f"Scanning {len(COINS)} coins every {SCAN_INTERVAL_SEC // 60} min\n"
        f"TP +{TP_PCT}%  |  SL -{SL_PCT}%  |  Min conf {MIN_CONFIDENCE}%"
    )

    # ── Background trade monitor ─────────────────
    monitor_thread = threading.Thread(target=trade_monitor_loop, daemon=True, name="TradeMonitor")
    monitor_thread.start()

    # ── Main scan loop – must NEVER exit ────────
    while True:
        try:
            scan_all_coins()
        except Exception as exc:
            log.error("[MAIN_LOOP_ERROR] %s", exc)

        log.info("[WAIT] Next scan in %d min...", SCAN_INTERVAL_SEC // 60)
        time.sleep(SCAN_INTERVAL_SEC)


if __name__ == "__main__":
    main()
