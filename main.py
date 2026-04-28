import requests, json, time, os
from datetime import datetime
from google import genai

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
NEWS_API_KEY = os.environ["NEWS_API_KEY"]

client = genai.Client(api_key=GEMINI_API_KEY)
SENT_FILE = "sent_coins.json"
TRADE_FILE = "active_trades.json"

COINS = [
    "bitcoin", "ethereum", "solana", "binancecoin", "ripple",
    "cardano", "dogecoin", "avalanche-2", "polkadot", "matic-network",
    "chainlink", "uniswap", "litecoin", "stellar", "cosmos",
    "near", "algorand", "vechain", "tezos", "flow"
]

def load_json(file):
    try:
        with open(file) as f:
            return json.load(f)
    except:
        return [] if file == SENT_FILE else []

def save_json(file, data):
    with open(file, "w") as f:
        json.dump(data, f)

def fetch_market(cid):
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{cid}"
        params = {"localization": "false", "tickers": "false", "community_data": "false", "developer_data": "false", "sparkline": "false"}
        r = requests.get(url, params=params, timeout=10)
        if r.status_code == 200:
            d = r.json().get("market_data", {})
            return {
                "id": cid, "name": r.json().get("name"), "symbol": r.json().get("symbol").upper(),
                "price": d.get("current_price", {}).get("usd"),
                "change": d.get("price_change_percentage_24h", 0)
            }
    except:
        pass
    return None

def fetch_ohlcv(cid):
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{cid}/ohlc"
        r = requests.get(url, params={"vs_currency": "usd", "days": 3}, timeout=10)
        if r.status_code == 200:
            return [c[4] for c in r.json()]
    except:
        pass
    return []

def calc_rsi(prices):
    if len(prices) < 15:
        return None
    gains, losses = [], []
    for i in range(1, len(prices)):
        d = prices[i] - prices[i-1]
        gains.append(d if d > 0 else 0)
        losses.append(abs(d) if d < 0 else 0)
    avg_gain = sum(gains[-14:]) / 14
    avg_loss = sum(losses[-14:]) / 14
    if avg_loss == 0:
        return 100
    return round(100 - (100 / (1 + avg_gain / avg_loss)), 2)

def fetch_news(name):
    if not NEWS_API_KEY:
        return []
    try:
        url = "https://newsapi.org/v2/everything"
        params = {"q": f"{name} crypto", "apiKey": NEWS_API_KEY, "pageSize": 3, "sortBy": "publishedAt", "language": "en"}
        r = requests.get(url, params=params, timeout=10)
        if r.status_code == 200:
            return [a["title"] for a in r.json().get("articles", [])[:3]]
    except:
        pass
    return []

def get_signal(asset, rsi_val, news):
    news_text = "\n".join([f"- {n}" for n in news]) if news else "No news"
    prompt = f"""Strict crypto AI. Output ONLY valid JSON. No extra text.

Rules: RSI < 40 + positive news = BUY. RSI > 60 + negative news = SELL.
RSI 40-60 or mixed = NO TRADE. Only signal if confidence >= 65%.

{asset['name']} ({asset['symbol']})
Price: ${asset['price']}
RSI: {rsi_val}
24h Change: {asset['change']}%
News:
{news_text}

Return format:
{{"action":"BUY","confidence":72,"reasoning":"oversold bounce","entry":{asset['price']},"tp":{asset['price']*1.03:.2f},"sl":{asset['price']*0.985:.2f}}}"""
    try:
        resp = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        raw = resp.text.strip().replace("```json", "").replace("```", "")
        return json.loads(raw)
    except:
        return None

def send_telegram(msg):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": msg}, timeout=10)
    except:
        pass

def fetch_price(coin_id):
    try:
        url = "https://api.coingecko.com/api/v3/simple/price"
        r = requests.get(url, params={"ids": coin_id, "vs_currencies": "usd"}, timeout=10)
        if r.status_code == 200:
            data = r.json()
            if coin_id in data:
                return data[coin_id]["usd"]
    except:
        pass
    return None

def scan_signals():
    sent = load_json(SENT_FILE)
    trades = load_json(TRADE_FILE)
    found = 0
    print("[SCAN] Starting signal scan...")
    for cid in COINS:
        if cid in sent:
            continue
        print(f"[SCAN] Checking {cid}...")
        market = fetch_market(cid)
        if not market or not market.get("price"):
            print(f"[SKIP] {cid} - no market data")
            time.sleep(2)
            continue
        time.sleep(5)
        ohlcv = fetch_ohlcv(cid)
        if not ohlcv:
            print(f"[SKIP] {cid} - no OHLCV")
            time.sleep(2)
            continue
        rsi = calc_rsi(ohlcv)
        if rsi is None:
            print(f"[SKIP] {cid} - RSI failed")
            time.sleep(2)
            continue
        if 40 <= rsi <= 60:
            print(f"[SKIP] {cid} - RSI {rsi} in neutral zone")
            time.sleep(2)
            continue
        news = fetch_news(market["name"])
        signal = get_signal(market, rsi, news)
        if signal and signal.get("action") in ["BUY", "SELL"]:
            news_line = f"\n📰 {news[0]}" if news else ""
            msg = f"""⚡ AETHER ASCENT SIGNAL

📊 {market['name']} ({market['symbol']})
🎯 {signal['action']} | {signal['confidence']}% | RSI {rsi}

💰 Entry: ${signal['entry']:,.2f}
✅ TP: ${signal['tp']:,.2f}
🛑 SL: ${signal['sl']:,.2f}

📝 {signal['reasoning']}{news_line}

🕐 {datetime.now().strftime('%d/%m %H:%M UTC')}"""
            send_telegram(msg)
            sent.append(cid)
            save_json(SENT_FILE, sent[-50:])
            trades.append({
                "coin": cid, "symbol": market["symbol"],
                "action": signal["action"], "entry": signal["entry"],
                "tp": signal["tp"], "sl": signal["sl"],
                "time": time.time()
            })
            save_json(TRADE_FILE, trades)
            print(f"[SIGNAL] {market['symbol']} {signal['action']} {signal['confidence']}%")
            found += 1
        else:
            print(f"[SKIP] {cid} - NO TRADE")
        time.sleep(2)
        if found >= 3:
            break
    print(f"[SCAN] Done. Signals found: {found}")

def check_trades():
    trades = load_json(TRADE_FILE)
    if not trades:
        return
    print(f"[TRADE] Checking {len(trades)} active trades...")
    updated = []
    for t in trades:
        if t.get("done"):
            if time.time() - t["done"] > 86400:
                continue
            updated.append(t)
            continue
        px = fetch_price(t["coin"])
        if not px:
            updated.append(t)
            continue
        if t["action"] == "BUY":
            if px >= t["tp"]:
                msg = f"✅ TP HIT!\n📊 {t['symbol']} BUY\n💰 ${t['entry']} → ${px}\n💵 Profit: ${round(px-t['entry'],2)}"
                send_telegram(msg)
                t["done"] = time.time()
                t["result"] = "TP"
                print(f"[TP] {t['symbol']}")
            elif px <= t["sl"]:
                msg = f"🛑 SL HIT!\n📊 {t['symbol']} BUY\n💰 ${t['entry']} → ${px}\n💔 Loss: ${round(t['entry']-px,2)}"
                send_telegram(msg)
                t["done"] = time.time()
                t["result"] = "SL"
                print(f"[SL] {t['symbol']}")
        elif t["action"] == "SELL":
            if px <= t["tp"]:
                msg = f"✅ TP HIT!\n📊 {t['symbol']} SELL\n💰 ${t['entry']} → ${px}\n💵 Profit: ${round(t['entry']-px,2)}"
                send_telegram(msg)
                t["done"] = time.time()
                t["result"] = "TP"
                print(f"[TP] {t['symbol']}")
            elif px >= t["sl"]:
                msg = f"🛑 SL HIT!\n📊 {t['symbol']} SELL\n💰 ${t['entry']} → ${px}\n💔 Loss: ${round(px-t['entry'],2)}"
                send_telegram(msg)
                t["done"] = time.time()
                t["result"] = "SL"
                print(f"[SL] {t['symbol']}")
        updated.append(t)
    save_json(TRADE_FILE, updated)

print("=" * 40)
print("⚡ AETHER ASCENT TRADING SYSTEM")
print("=" * 40)
print("Scan: Every 15 min | 5s per coin")
print("Trade Check: Every 1 min")
print("=" * 40)

try:
    scan_signals()
except Exception as e:
    print(f"[ERROR] Init scan: {e}")

last_scan = time.time()

while True:
    now = time.time()
    try:
        check_trades()
        if now - last_scan >= 900:
            scan_signals()
            last_scan = time.time()
            print("[WAIT] 60s break before trade check...")
            time.sleep(60)
    except Exception as e:
        print(f"[ERROR] {e}")
    time.sleep(60)
