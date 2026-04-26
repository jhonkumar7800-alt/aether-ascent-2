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
    url = f"https://api.coingecko.com/api/v3/coins/{cid}"
    p = {"localization": "false", "tickers": "false", "community_data": "false", "developer_data": "false", "sparkline": "false"}
    r = requests.get(url, params=p)
    if r.status_code == 200:
        d = r.json().get("market_data", {})
        return {
            "id": cid, "name": r.json().get("name"), "symbol": r.json().get("symbol").upper(),
            "price": d.get("current_price", {}).get("usd"),
            "change": d.get("price_change_percentage_24h"),
            "vol": d.get("total_volume", {}).get("usd")
        }
    return None

def fetch_ohlcv(cid):
    url = f"https://api.coingecko.com/api/v3/coins/{cid}/ohlc"
    r = requests.get(url, params={"vs_currency": "usd", "days": 3})
    if r.status_code == 200:
        return [c[4] for c in r.json()]
    return []

def calc_rsi(prices, p=14):
    if len(prices) < p+1:
        return None
    g, l = [], []
    for i in range(1, len(prices)):
        d = prices[i] - prices[i-1]
        g.append(d if d>0 else 0)
        l.append(abs(d) if d<0 else 0)
    ag = sum(g[-p:])/p
    al = sum(l[-p:])/p
    return round(100 - (100/(1+ag/al)), 2) if al else 100

def fetch_news(name):
    try:
        url = "https://newsapi.org/v2/everything"
        p = {"q": f"{name} crypto", "apiKey": NEWS_API_KEY, "pageSize": 3, "sortBy": "publishedAt", "language": "en"}
        r = requests.get(url, params=p)
        if r.status_code == 200:
            return [a["title"] for a in r.json().get("articles", [])[:3]]
    except:
        pass
    return []

def ai_signal(asset, rsi_val, news):
    nt = "\n".join([f"- {n}" for n in news]) if news else "No news"
    prompt = f"""Strict crypto AI. Output ONLY valid JSON. No extra text.

Rules: RSI < 40 + positive news = BUY. RSI > 60 + negative news = SELL.
RSI 40-60 or mixed = NO TRADE. Only signal if 65%+ confidence.

{asset['name']} ({asset['symbol']})
Price: ${asset['price']}
RSI: {rsi_val}
24h Change: {asset['change']}%
Volume: ${asset['vol']}
News: {nt}

Return exact: {{"action":"BUY","confidence":72,"reasoning":"short","entry":42000,"tp":43000,"sl":41500}}"""
    try:
        resp = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        raw = resp.text.strip().replace("```json","").replace("```","")
        return json.loads(raw)
    except:
        return None

def send_tg(msg):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": msg})

def fetch_price(coin_id):
    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {"ids": coin_id, "vs_currencies": "usd"}
    r = requests.get(url, params=params)
    if r.status_code == 200:
        return r.json().get(coin_id, {}).get("usd")
    return None

def generate_signals():
    sent = load_json(SENT_FILE)
    trades = load_json(TRADE_FILE)
    found = 0
    for cid in COINS:
        if cid in sent:
            continue
        m = fetch_market(cid)
        if not m or not m["price"]:
            continue
        closes = fetch_ohlcv(cid)
        rv = calc_rsi(closes)
        if rv is None:
            continue
        if 40 <= rv <= 60:
            continue
        news = fetch_news(m["name"])
        sig = ai_signal(m, rv, news)
        if sig and sig["action"] in ["BUY","SELL"]:
            nh = news[0] if news else ""
            nl = f"\n📰 {nh}" if nh else ""
            msg = f"""⚡ AETHER ASCENT

📊 {m['name']} ({m['symbol']})
🎯 {sig['action']} | {sig['confidence']}% | RSI {rv}

💰 Entry: ${sig['entry']:,.2f}
✅ TP: ${sig['tp']:,.2f}
🛑 SL: ${sig['sl']:,.2f}

📝 {sig['reasoning']}{nl}

🕐 {datetime.now().strftime('%d/%m %H:%M UTC')}"""
            send_tg(msg)
            sent.append(cid)
            save_json(SENT_FILE, sent[-50:])
            trades.append({"coin": cid, "symbol": m["symbol"], "action": sig["action"], "entry": sig["entry"], "tp": sig["tp"], "sl": sig["sl"], "time": time.time()})
            save_json(TRADE_FILE, trades)
            print(f"SIGNAL: {m['symbol']} {sig['action']} {sig['confidence']}% RSI{rv}")
            found += 1
            time.sleep(5)
        if found >= 3:
            break
        time.sleep(2)

def check_trades():
    trades = load_json(TRADE_FILE)
    updated = []
    for t in trades:
        if t.get("done"):
            if time.time() - t["done"] > 86400:
                continue
            updated.append(t)
            continue
        price = fetch_price(t["coin"])
        if not price:
            updated.append(t)
            continue
        if t["action"] == "BUY":
            if price >= t["tp"]:
                msg = f"""✅ TP HIT!

📊 {t['symbol']}
🎯 BUY | ${t['entry']} → ${price}
💰 Profit: ${round(price - t['entry'], 2)}
🕐 {datetime.now().strftime('%d/%m %H:%M UTC')}"""
                send_tg(msg)
                t["done"] = time.time()
                t["result"] = "TP"
                print(f"TP HIT: {t['symbol']}")
            elif price <= t["sl"]:
                msg = f"""🛑 SL HIT!

📊 {t['symbol']}
🎯 BUY | ${t['entry']} → ${price}
💔 Loss: ${round(t['entry'] - price, 2)}
🕐 {datetime.now().strftime('%d/%m %H:%M UTC')}"""
                send_tg(msg)
                t["done"] = time.time()
                t["result"] = "SL"
                print(f"SL HIT: {t['symbol']}")
        elif t["action"] == "SELL":
            if price <= t["tp"]:
                msg = f"""✅ TP HIT!

📊 {t['symbol']}
🎯 SELL | ${t['entry']} → ${price}
💰 Profit: ${round(t['entry'] - price, 2)}
🕐 {datetime.now().strftime('%d/%m %H:%M UTC')}"""
                send_tg(msg)
                t["done"] = time.time()
                t["result"] = "TP"
                print(f"TP HIT: {t['symbol']}")
            elif price >= t["sl"]:
                msg = f"""🛑 SL HIT!

📊 {t['symbol']}
🎯 SELL | ${t['entry']} → ${price}
💔 Loss: ${round(price - t['entry'], 2)}
🕐 {datetime.now().strftime('%d/%m %H:%M UTC')}"""
                send_tg(msg)
                t["done"] = time.time()
                t["result"] = "SL"
                print(f"SL HIT: {t['symbol']}")
        updated.append(t)
    save_json(TRADE_FILE, updated)

print("="*30)
print("AETHER ASCENT LIVE")
print("="*30)

generate_signals()
last_signal_scan = time.time()

while True:
    now = time.time()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Running...")
    try:
        if now - last_signal_scan >= 900:
            generate_signals()
            last_signal_scan = now
        check_trades()
    except Exception as e:
        print(f"Error: {e}")
    time.sleep(60)
