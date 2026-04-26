"""
Crypto Trading Signal Bot
=========================
- Fetches top 20 coins from CoinGecko
- Calculates RSI from hourly OHLCV data
- Fetches news from NewsAPI
- Uses Gemini AI (google-genai) for BUY/SELL/NO TRADE decisions
- Sends signals to Telegram
- Tracks active trades for TP/SL alerts
- Runs every 15 min (signals) and every 1 min (trade tracking)
- Designed for Railway deployment with env vars
"""

import os
import time
import logging
import traceback
from datetime import datetime, timezone
from typing import Optional

import requests
from google import genai
from google.genai import types

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Environment Variables
# ─────────────────────────────────────────────
GEMINI_API_KEY    = os.environ.get("GEMINI_API_KEY", "")
NEWS_API_KEY      = os.environ.get("NEWS_API_KEY", "")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID  = os.environ.get("TELEGRAM_CHAT_ID", "")

SIGNAL_INTERVAL_SEC = int(os.environ.get("SIGNAL_INTERVAL_SEC", "900"))   # 15 min
TRACK_INTERVAL_SEC  = int(os.environ.get("TRACK_INTERVAL_SEC", "60"))      # 1 min

TP_PERCENT = float(os.environ.get("TP_PERCENT", "3.0"))   # Take-profit %
SL_PERCENT = float(os.environ.get("SL_PERCENT", "1.5"))   # Stop-loss %

COINGECKO_BASE = "https://api.coingecko.com/api/v3"
NEWS_API_BASE  = "https://newsapi.org/v2"

# ─────────────────────────────────────────────
# Gemini client (google-genai)
# ─────────────────────────────────────────────
gemini_client: Optional[genai.Client] = None

def init_gemini() -> bool:
    global gemini_client
    if not GEMINI_API_KEY:
        log.error("GEMINI_API_KEY not set.")
        return False
    try:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        log.info("Gemini client initialised.")
        return True
    except Exception as e:
        log.error(f"Gemini init failed: {e}")
        return False

# ─────────────────────────────────────────────
# Active Trade Tracker  {coin_id: trade_dict}
# ─────────────────────────────────────────────
active_trades: dict = {}

# ─────────────────────────────────────────────
# Telegram
# ─────────────────────────────────────────────
def send_telegram(message: str) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log.warning("Telegram credentials missing — skipping message.")
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
    }
    try:
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        log.info("Telegram message sent.")
        return True
    except requests.exceptions.RequestException as e:
        log.error(f"Telegram send failed: {e}")
        return False

# ─────────────────────────────────────────────
# CoinGecko — Top 20 Coins
# ─────────────────────────────────────────────
def fetch_top_coins(n: int = 20) -> list[dict]:
    url = f"{COINGECKO_BASE}/coins/markets"
    params = {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": n,
        "page": 1,
        "sparkline": False,
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        coins = resp.json()
        log.info(f"Fetched {len(coins)} coins from CoinGecko.")
        return coins
    except requests.exceptions.HTTPError as e:
        log.error(f"CoinGecko HTTP error (markets): {e} — {resp.text[:200]}")
    except requests.exceptions.RequestException as e:
        log.error(f"CoinGecko request error (markets): {e}")
    except Exception as e:
        log.error(f"Unexpected error fetching coins: {e}")
    return []

# ─────────────────────────────────────────────
# CoinGecko — Hourly OHLCV  (last ~2 days)
# ─────────────────────────────────────────────
def fetch_ohlcv(coin_id: str, days: int = 2) -> list[list]:
    """Returns list of [timestamp, open, high, low, close, volume]."""
    url = f"{COINGECKO_BASE}/coins/{coin_id}/ohlc"
    params = {"vs_currency": "usd", "days": days}
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        log.debug(f"OHLCV for {coin_id}: {len(data)} candles.")
        return data  # [timestamp_ms, open, high, low, close]
    except requests.exceptions.HTTPError as e:
        log.error(f"CoinGecko OHLCV error ({coin_id}): {e}")
    except requests.exceptions.RequestException as e:
        log.error(f"CoinGecko request error ({coin_id}): {e}")
    except Exception as e:
        log.error(f"Unexpected OHLCV error ({coin_id}): {e}")
    return []

# ─────────────────────────────────────────────
# RSI Calculation
# ─────────────────────────────────────────────
def calculate_rsi(ohlcv: list[list], period: int = 14) -> Optional[float]:
    """Calculates RSI from OHLCV close prices. Returns None if data insufficient."""
    if not ohlcv or len(ohlcv) < period + 1:
        return None
    closes = [candle[4] for candle in ohlcv]  # index 4 = close
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [max(d, 0) for d in deltas]
    losses = [abs(min(d, 0)) for d in deltas]

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)

# ─────────────────────────────────────────────
# NewsAPI — fetch coin news
# ─────────────────────────────────────────────
def fetch_news(coin_name: str, max_articles: int = 5) -> list[str]:
    if not NEWS_API_KEY:
        log.warning("NEWS_API_KEY not set — skipping news fetch.")
        return []
    url = f"{NEWS_API_BASE}/everything"
    params = {
        "q": coin_name,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": max_articles,
        "apiKey": NEWS_API_KEY,
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        articles = resp.json().get("articles", [])
        headlines = [
            a.get("title", "").strip()
            for a in articles
            if a.get("title")
        ]
        log.debug(f"News for {coin_name}: {len(headlines)} headlines.")
        return headlines[:max_articles]
    except requests.exceptions.HTTPError as e:
        log.error(f"NewsAPI HTTP error ({coin_name}): {e}")
    except requests.exceptions.RequestException as e:
        log.error(f"NewsAPI request error ({coin_name}): {e}")
    except Exception as e:
        log.error(f"Unexpected news error ({coin_name}): {e}")
    return []

# ─────────────────────────────────────────────
# Gemini AI — Signal Decision
# ─────────────────────────────────────────────
def get_ai_signal(
    coin_name: str,
    symbol: str,
    price: float,
    rsi: float,
    headlines: list[str],
) -> dict:
    """Returns dict with keys: signal, confidence, reasoning."""
    default = {"signal": "NO TRADE", "confidence": "N/A", "reasoning": "AI unavailable"}
    if not gemini_client:
        return default

    news_block = "\n".join(f"- {h}" for h in headlines) if headlines else "No recent news available."
    prompt = f"""You are a professional crypto trading analyst.

Coin: {coin_name} ({symbol.upper()})
Current Price: ${price:,.4f}
RSI (14-period, hourly): {rsi}

Recent News Headlines:
{news_block}

Based on the RSI and sentiment from the news, provide a trading signal.
Respond ONLY in this exact format (no extra text):

SIGNAL: <BUY | SELL | NO TRADE>
CONFIDENCE: <HIGH | MEDIUM | LOW>
REASONING: <one concise sentence>
"""
    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=150,
            ),
        )
        text = response.text.strip()
        result = {"signal": "NO TRADE", "confidence": "N/A", "reasoning": text}

        for line in text.splitlines():
            if line.startswith("SIGNAL:"):
                result["signal"] = line.replace("SIGNAL:", "").strip()
            elif line.startswith("CONFIDENCE:"):
                result["confidence"] = line.replace("CONFIDENCE:", "").strip()
            elif line.startswith("REASONING:"):
                result["reasoning"] = line.replace("REASONING:", "").strip()

        log.info(f"AI signal for {symbol}: {result['signal']} ({result['confidence']})")
        return result
    except Exception as e:
        log.error(f"Gemini API error ({coin_name}): {e}")
        return default

# ─────────────────────────────────────────────
# Signal Runner — called every 15 minutes
# ─────────────────────────────────────────────
def run_signal_cycle():
    log.info("═" * 50)
    log.info("▶ Starting signal cycle...")
    log.info("═" * 50)

    coins = fetch_top_coins(20)
    if not coins:
        log.warning("No coins fetched — skipping cycle.")
        return

    for coin in coins:
        coin_id   = coin.get("id", "")
        coin_name = coin.get("name", "")
        symbol    = coin.get("symbol", "")
        price     = coin.get("current_price", 0.0)

        log.info(f"Processing {coin_name} ({symbol.upper()}) @ ${price:,.4f}")

        # ── RSI
        ohlcv = fetch_ohlcv(coin_id, days=2)
        rsi = calculate_rsi(ohlcv)
        if rsi is None:
            log.warning(f"  Insufficient OHLCV data for {coin_name}, skipping.")
            continue
        log.info(f"  RSI: {rsi}")

        # ── News
        headlines = fetch_news(coin_name)

        # ── AI Decision
        ai = get_ai_signal(coin_name, symbol, price, rsi, headlines)
        signal     = ai["signal"]
        confidence = ai["confidence"]
        reasoning  = ai["reasoning"]

        # ── Only act on BUY or SELL
        if signal not in ("BUY", "SELL"):
            log.info(f"  Signal: NO TRADE — skipping.")
            # Rate-limit CoinGecko: 10-12 calls/min on free tier
            time.sleep(6)
            continue

        # ── Format Telegram message
        direction = "🟢 BUY" if signal == "BUY" else "🔴 SELL"
        tp = round(price * (1 + TP_PERCENT / 100), 6) if signal == "BUY" else round(price * (1 - TP_PERCENT / 100), 6)
        sl = round(price * (1 - SL_PERCENT / 100), 6) if signal == "BUY" else round(price * (1 + SL_PERCENT / 100), 6)

        msg = (
            f"<b>{direction} Signal — {coin_name} ({symbol.upper()})</b>\n\n"
            f"💰 Price:      ${price:,.6f}\n"
            f"📊 RSI:        {rsi}\n"
            f"🎯 TP:         ${tp:,.6f}  (+{TP_PERCENT}%)\n"
            f"🛑 SL:         ${sl:,.6f}  (-{SL_PERCENT}%)\n"
            f"🧠 Confidence: {confidence}\n"
            f"📝 Reasoning:  {reasoning}\n\n"
            f"🕐 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        )
        send_telegram(msg)
        log.info(f"  Sent {signal} signal for {symbol.upper()}.")

        # ── Register trade for tracking
        active_trades[coin_id] = {
            "coin_id":   coin_id,
            "coin_name": coin_name,
            "symbol":    symbol,
            "signal":    signal,
            "entry":     price,
            "tp":        tp,
            "sl":        sl,
            "alerted":   False,
        }

        time.sleep(6)  # Respect CoinGecko rate limits

    log.info("✅ Signal cycle complete.")

# ─────────────────────────────────────────────
# Trade Tracker — called every 1 minute
# ─────────────────────────────────────────────
def run_trade_tracker():
    if not active_trades:
        return

    log.info(f"🔍 Tracking {len(active_trades)} active trade(s)...")
    closed = []

    for coin_id, trade in list(active_trades.items()):
        url = f"{COINGECKO_BASE}/simple/price"
        params = {"ids": coin_id, "vs_currencies": "usd"}
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            price_data = resp.json()
            current_price = price_data.get(coin_id, {}).get("usd")
            if current_price is None:
                log.warning(f"  No price data for {coin_id}.")
                continue
        except Exception as e:
            log.error(f"  Price fetch error ({coin_id}): {e}")
            continue

        symbol = trade["symbol"].upper()
        entry  = trade["entry"]
        tp     = trade["tp"]
        sl     = trade["sl"]
        signal = trade["signal"]

        pnl_pct = ((current_price - entry) / entry * 100) if signal == "BUY" else ((entry - current_price) / entry * 100)
        log.info(f"  {symbol}: entry=${entry:.4f} | now=${current_price:.4f} | PnL={pnl_pct:+.2f}%")

        hit_tp = (signal == "BUY"  and current_price >= tp) or (signal == "SELL" and current_price <= tp)
        hit_sl = (signal == "BUY"  and current_price <= sl) or (signal == "SELL" and current_price >= sl)

        if hit_tp:
            msg = (
                f"✅ <b>TAKE PROFIT HIT — {trade['coin_name']} ({symbol})</b>\n\n"
                f"📈 Signal:  {signal}\n"
                f"💰 Entry:   ${entry:,.6f}\n"
                f"🎯 TP:      ${tp:,.6f}\n"
                f"📊 Current: ${current_price:,.6f}\n"
                f"💹 PnL:     {pnl_pct:+.2f}%\n"
                f"🕐 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
            )
            send_telegram(msg)
            log.info(f"  TP hit for {symbol}!")
            closed.append(coin_id)

        elif hit_sl:
            msg = (
                f"🛑 <b>STOP LOSS HIT — {trade['coin_name']} ({symbol})</b>\n\n"
                f"📉 Signal:  {signal}\n"
                f"💰 Entry:   ${entry:,.6f}\n"
                f"🛑 SL:      ${sl:,.6f}\n"
                f"📊 Current: ${current_price:,.6f}\n"
                f"💹 PnL:     {pnl_pct:+.2f}%\n"
                f"🕐 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
            )
            send_telegram(msg)
            log.info(f"  SL hit for {symbol}!")
            closed.append(coin_id)

        time.sleep(2)

    for coin_id in closed:
        active_trades.pop(coin_id, None)
        log.info(f"  Trade closed and removed: {coin_id}")

# ─────────────────────────────────────────────
# Startup Validation
# ─────────────────────────────────────────────
def validate_env() -> bool:
    missing = []
    for var in ["GEMINI_API_KEY", "NEWS_API_KEY", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"]:
        if not os.environ.get(var):
            missing.append(var)
    if missing:
        log.error(f"Missing required env vars: {', '.join(missing)}")
        return False
    return True

# ─────────────────────────────────────────────
# Main Loop
# ─────────────────────────────────────────────
def main():
    log.info("🚀 Crypto Signal Bot starting up...")

    if not validate_env():
        log.error("Aborting — fix missing environment variables.")
        raise SystemExit(1)

    if not init_gemini():
        log.error("Aborting — Gemini client could not be initialised.")
        raise SystemExit(1)

    send_telegram(
        "🤖 <b>Crypto Signal Bot is now live!</b>\n"
        f"Signals every {SIGNAL_INTERVAL_SEC // 60} min | "
        f"TP: {TP_PERCENT}% | SL: {SL_PERCENT}%"
    )

    last_signal_time = 0.0

    log.info("⏳ Entering main loop...")
    while True:
        try:
            now = time.time()

            # ── Every 15 minutes: run full signal cycle
            if now - last_signal_time >= SIGNAL_INTERVAL_SEC:
                run_signal_cycle()
                last_signal_time = time.time()

            # ── Every loop tick (1 min sleep): track trades
            run_trade_tracker()

        except KeyboardInterrupt:
            log.info("Interrupted by user. Shutting down.")
            break
        except Exception as e:
            log.error(f"Unhandled error in main loop: {e}")
            log.error(traceback.format_exc())
            # Don't crash — sleep and retry
            time.sleep(30)

        time.sleep(TRACK_INTERVAL_SEC)


if __name__ == "__main__":
    main()
