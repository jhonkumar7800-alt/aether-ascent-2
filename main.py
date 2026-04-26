"""
Crypto Trading Signal Bot — Bybit Edition
==========================================
- Top 20 coins list   : CoinGecko  (1 call per cycle only)
- OHLCV for RSI       : Bybit      (free, no API key, no rate limit issues)
- Live price tracking : Bybit      (free, no API key)
- News                : NewsAPI
- AI decision         : Gemini (google-genai)
- Alerts              : Telegram
- Signals every 15 min | Trade tracking every 1 min
- Deployed on Railway with env vars
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
GEMINI_API_KEY      = os.environ.get("GEMINI_API_KEY", "")
NEWS_API_KEY        = os.environ.get("NEWS_API_KEY", "")
TELEGRAM_BOT_TOKEN  = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID    = os.environ.get("TELEGRAM_CHAT_ID", "")

SIGNAL_INTERVAL_SEC = int(os.environ.get("SIGNAL_INTERVAL_SEC", "900"))   # 15 min
TRACK_INTERVAL_SEC  = int(os.environ.get("TRACK_INTERVAL_SEC", "60"))     # 1 min

TP_PERCENT = float(os.environ.get("TP_PERCENT", "3.0"))
SL_PERCENT = float(os.environ.get("SL_PERCENT", "1.5"))

# ─────────────────────────────────────────────
# API Base URLs
# ─────────────────────────────────────────────
COINGECKO_BASE = "https://api.coingecko.com/api/v3"
BYBIT_BASE     = "https://api.bybit.com/v5"
NEWS_API_BASE  = "https://newsapi.org/v2"

# Stablecoins / wrapped tokens — skip
SKIP_SYMBOLS = {
    "usdt", "usdc", "busd", "dai", "tusd", "usdp",
    "usdd", "fdusd", "pyusd", "steth", "wbtc", "weth"
}

# ─────────────────────────────────────────────
# Gemini Client
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
# Active Trade Tracker  { bybit_symbol: trade }
# ─────────────────────────────────────────────
active_trades: dict = {}

# ─────────────────────────────────────────────
# Telegram
# ─────────────────────────────────────────────
def send_telegram(message: str) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log.warning("Telegram credentials missing — skipping.")
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
# CoinGecko — Top 20 Coins (only 1 call per cycle)
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
    wait = 30
    for attempt in range(1, 4):
        try:
            resp = requests.get(url, params=params, timeout=15)
            if resp.status_code == 429:
                log.warning(f"CoinGecko rate limited — waiting {wait}s (attempt {attempt}/3)")
                time.sleep(wait)
                wait = min(wait * 2, 120)
                continue
            resp.raise_for_status()
            coins = resp.json()
            log.info(f"Fetched {len(coins)} coins from CoinGecko.")
            return coins
        except requests.exceptions.HTTPError as e:
            log.error(f"CoinGecko HTTP error: {e}")
            return []
        except requests.exceptions.RequestException as e:
            log.error(f"CoinGecko request error: {e}")
            return []
        except Exception as e:
            log.error(f"CoinGecko unexpected error: {e}")
            return []
    log.error("CoinGecko: all retries failed.")
    return []

# ─────────────────────────────────────────────
# Bybit — Symbol Formatter
# ─────────────────────────────────────────────
def to_bybit_symbol(symbol: str) -> str:
    return f"{symbol.upper()}USDT"

# ─────────────────────────────────────────────
# Bybit — OHLCV (FREE, no API key, high rate limit)
# ─────────────────────────────────────────────
def fetch_ohlcv_bybit(bybit_symbol: str, interval: str = "60", limit: int = 50) -> list[list]:
    """
    Bybit V5 kline endpoint.
    interval: "1","3","5","15","30","60","120","240","D"
    Returns candles sorted oldest → newest.
    Candle format: [startTime, open, high, low, close, volume, turnover]
    close = index 4
    """
    url = f"{BYBIT_BASE}/market/kline"
    params = {
        "category": "spot",
        "symbol": bybit_symbol,
        "interval": interval,
        "limit": limit,
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if data.get("retCode") != 0:
            log.warning(f"  Bybit kline error for {bybit_symbol}: {data.get('retMsg')}")
            return []

        candles = data.get("result", {}).get("list", [])
        if not candles:
            log.warning(f"  Bybit: no candles returned for {bybit_symbol}")
            return []

        # Bybit returns newest first — reverse for RSI
        candles.reverse()
        log.debug(f"  Bybit OHLCV {bybit_symbol}: {len(candles)} candles.")
        return candles

    except requests.exceptions.RequestException as e:
        log.error(f"  Bybit OHLCV request error ({bybit_symbol}): {e}")
    except Exception as e:
        log.error(f"  Bybit OHLCV unexpected error ({bybit_symbol}): {e}")
    return []

# ─────────────────────────────────────────────
# Bybit — Live Price (FREE, no API key)
# ─────────────────────────────────────────────
def fetch_price_bybit(bybit_symbol: str) -> Optional[float]:
    url = f"{BYBIT_BASE}/market/tickers"
    params = {"category": "spot", "symbol": bybit_symbol}
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if data.get("retCode") != 0:
            return None

        tickers = data.get("result", {}).get("list", [])
        if not tickers:
            return None

        price = float(tickers[0].get("lastPrice", 0))
        return price if price > 0 else None

    except requests.exceptions.RequestException as e:
        log.error(f"  Bybit price request error ({bybit_symbol}): {e}")
    except Exception as e:
        log.error(f"  Bybit price unexpected error ({bybit_symbol}): {e}")
    return None

# ─────────────────────────────────────────────
# RSI Calculation
# ─────────────────────────────────────────────
def calculate_rsi(ohlcv: list[list], period: int = 14) -> Optional[float]:
    if not ohlcv or len(ohlcv) < period + 1:
        return None
    try:
        closes = [float(candle[4]) for candle in ohlcv]
    except (IndexError, ValueError, TypeError) as e:
        log.error(f"  RSI parse error: {e}")
        return None

    deltas   = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains    = [max(d, 0.0) for d in deltas]
    losses   = [abs(min(d, 0.0)) for d in deltas]
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
# NewsAPI — Fetch Headlines
# ─────────────────────────────────────────────
def fetch_news(coin_name: str, max_articles: int = 5) -> list[str]:
    if not NEWS_API_KEY:
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
        headlines = [a.get("title", "").strip() for a in articles if a.get("title")]
        log.debug(f"  News for {coin_name}: {len(headlines)} headlines.")
        return headlines[:max_articles]
    except requests.exceptions.HTTPError as e:
        log.error(f"  NewsAPI HTTP error ({coin_name}): {e}")
    except requests.exceptions.RequestException as e:
        log.error(f"  NewsAPI request error ({coin_name}): {e}")
    except Exception as e:
        log.error(f"  NewsAPI unexpected error ({coin_name}): {e}")
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
    default = {"signal": "NO TRADE", "confidence": "N/A", "reasoning": "AI unavailable"}
    if not gemini_client:
        return default

    news_block = (
        "\n".join(f"- {h}" for h in headlines)
        if headlines
        else "No recent news available."
    )
    prompt = f"""You are a professional crypto trading analyst.

Coin: {coin_name} ({symbol.upper()})
Current Price: ${price:,.6f}
RSI (14-period, 1h candles): {rsi}

Recent News Headlines:
{news_block}

Based on RSI and news sentiment, provide a trading signal.
Respond ONLY in this exact format — no extra text, no markdown:

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
        text   = response.text.strip()
        result = {"signal": "NO TRADE", "confidence": "N/A", "reasoning": text}

        for line in text.splitlines():
            line = line.strip()
            if line.startswith("SIGNAL:"):
                result["signal"] = line.replace("SIGNAL:", "").strip()
            elif line.startswith("CONFIDENCE:"):
                result["confidence"] = line.replace("CONFIDENCE:", "").strip()
            elif line.startswith("REASONING:"):
                result["reasoning"] = line.replace("REASONING:", "").strip()

        log.info(f"  AI -> {result['signal']} ({result['confidence']})")
        return result

    except Exception as e:
        log.error(f"  Gemini API error ({coin_name}): {e}")
        return default

# ─────────────────────────────────────────────
# Signal Runner — every 15 minutes
# ─────────────────────────────────────────────
def run_signal_cycle():
    log.info("=" * 55)
    log.info("Starting signal cycle")
    log.info("=" * 55)

    coins = fetch_top_coins(20)
    if not coins:
        log.warning("No coins fetched — skipping cycle.")
        return

    for coin in coins:
        coin_name = coin.get("name", "")
        symbol    = coin.get("symbol", "").lower()
        cg_price  = float(coin.get("current_price") or 0.0)

        if symbol in SKIP_SYMBOLS:
            log.info(f"  Skipping stablecoin: {symbol.upper()}")
            continue

        bybit_sym = to_bybit_symbol(symbol)
        log.info(f"--- {coin_name} ({bybit_sym})")

        # OHLCV from Bybit
        ohlcv = fetch_ohlcv_bybit(bybit_sym)
        if not ohlcv:
            log.warning(f"  No OHLCV for {bybit_sym} on Bybit — skipping.")
            continue

        # RSI
        rsi = calculate_rsi(ohlcv)
        if rsi is None:
            log.warning(f"  RSI failed for {coin_name} — skipping.")
            continue
        log.info(f"  RSI: {rsi}")

        # Live price from Bybit, fallback to CoinGecko price
        live_price = fetch_price_bybit(bybit_sym)
        price = live_price if live_price else cg_price
        if price <= 0:
            log.warning(f"  No valid price for {bybit_sym} — skipping.")
            continue
        source = "Bybit" if live_price else "CoinGecko"
        log.info(f"  Price: ${price:,.6f} ({source})")

        # News
        headlines = fetch_news(coin_name)

        # Gemini AI
        ai         = get_ai_signal(coin_name, symbol, price, rsi, headlines)
        signal     = ai["signal"]
        confidence = ai["confidence"]
        reasoning  = ai["reasoning"]

        if signal not in ("BUY", "SELL"):
            log.info(f"  Signal: NO TRADE — skipping.")
            time.sleep(0.5)
            continue

        # TP / SL
        if signal == "BUY":
            tp = round(price * (1 + TP_PERCENT / 100), 6)
            sl = round(price * (1 - SL_PERCENT / 100), 6)
        else:
            tp = round(price * (1 - TP_PERCENT / 100), 6)
            sl = round(price * (1 + SL_PERCENT / 100), 6)

        direction = "BUY" if signal == "BUY" else "SELL"
        emoji     = "🟢" if signal == "BUY" else "🔴"

        msg = (
            f"{emoji} <b>{direction} Signal — {coin_name} ({symbol.upper()})</b>\n\n"
            f"💰 Price:       ${price:,.6f}\n"
            f"📊 RSI (1h):    {rsi}\n"
            f"🎯 Take Profit: ${tp:,.6f}  (+{TP_PERCENT}%)\n"
            f"🛑 Stop Loss:   ${sl:,.6f}  (-{SL_PERCENT}%)\n"
            f"🧠 Confidence:  {confidence}\n"
            f"📝 Reasoning:   {reasoning}\n\n"
            f"📡 Data: Bybit\n"
            f"🕐 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        )
        send_telegram(msg)
        log.info(f"  Sent {signal} signal for {symbol.upper()}.")

        active_trades[bybit_sym] = {
            "bybit_sym": bybit_sym,
            "coin_name": coin_name,
            "symbol":    symbol,
            "signal":    signal,
            "entry":     price,
            "tp":        tp,
            "sl":        sl,
        }

        time.sleep(0.5)

    log.info("Signal cycle complete.")

# ─────────────────────────────────────────────
# Trade Tracker — every 1 minute
# ─────────────────────────────────────────────
def run_trade_tracker():
    if not active_trades:
        return

    log.info(f"Tracking {len(active_trades)} active trade(s)...")
    closed = []

    for bybit_sym, trade in list(active_trades.items()):
        current_price = fetch_price_bybit(bybit_sym)
        if current_price is None:
            log.warning(f"  Could not fetch price for {bybit_sym}, skipping.")
            continue

        symbol = trade["symbol"].upper()
        entry  = trade["entry"]
        tp     = trade["tp"]
        sl     = trade["sl"]
        signal = trade["signal"]

        pnl_pct = (
            (current_price - entry) / entry * 100
            if signal == "BUY"
            else (entry - current_price) / entry * 100
        )
        log.info(
            f"  {symbol}: entry=${entry:.4f} | now=${current_price:.4f} | PnL={pnl_pct:+.2f}%"
        )

        hit_tp = (
            (signal == "BUY"  and current_price >= tp) or
            (signal == "SELL" and current_price <= tp)
        )
        hit_sl = (
            (signal == "BUY"  and current_price <= sl) or
            (signal == "SELL" and current_price >= sl)
        )

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
            closed.append(bybit_sym)

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
            closed.append(bybit_sym)

        time.sleep(0.3)

    for bybit_sym in closed:
        active_trades.pop(bybit_sym, None)
        log.info(f"  Trade closed: {bybit_sym}")

# ─────────────────────────────────────────────
# Startup Validation
# ─────────────────────────────────────────────
def validate_env() -> bool:
    required = ["GEMINI_API_KEY", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"]
    missing  = [v for v in required if not os.environ.get(v)]
    if missing:
        log.error(f"Missing required env vars: {', '.join(missing)}")
        return False
    if not NEWS_API_KEY:
        log.warning("NEWS_API_KEY not set — bot will run without news sentiment.")
    return True

# ─────────────────────────────────────────────
# Main Loop
# ─────────────────────────────────────────────
def main():
    log.info("Crypto Signal Bot (Bybit Edition) starting up...")

    if not validate_env():
        log.error("Aborting — fix missing environment variables.")
        raise SystemExit(1)

    if not init_gemini():
        log.error("Aborting — Gemini client could not be initialised.")
        raise SystemExit(1)

    send_telegram(
        "🤖 <b>Crypto Signal Bot is LIVE!</b>\n\n"
        f"📡 Data Source: Bybit (no rate limits)\n"
        f"⏱ Signals every: {SIGNAL_INTERVAL_SEC // 60} min\n"
        f"🎯 Take Profit: {TP_PERCENT}%\n"
        f"🛑 Stop Loss:   {SL_PERCENT}%"
    )

    last_signal_time = 0.0

    log.info("Entering main loop...")
    while True:
        try:
            now = time.time()

            if now - last_signal_time >= SIGNAL_INTERVAL_SEC:
                run_signal_cycle()
                last_signal_time = time.time()

            run_trade_tracker()

        except KeyboardInterrupt:
            log.info("Interrupted by user. Shutting down.")
            break
        except Exception as e:
            log.error(f"Unhandled error in main loop: {e}")
            log.error(traceback.format_exc())
            time.sleep(30)

        time.sleep(TRACK_INTERVAL_SEC)


if __name__ == "__main__":
    main()
