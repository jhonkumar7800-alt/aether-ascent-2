import requests, json, time, os
from datetime import datetime
from google import genai

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
NEWS_API_KEY = os.environ.get("NEWS_API_KEY", "")
COINGECKO_API_KEY = os.environ.get("COINGECKO_API_KEY", "")

HEADERS = {"x-cg-pro-api-key": COINGECKO_API_KEY} if COINGECKO_API_KEY else {}
BASE_URL = "https://pro-api.coingecko.com/api/v3" if COINGECKO_API_KEY else "https://api.coingecko.com/api/v3"

client = genai.Client(api_key=GEMINI_API_KEY)
SENT_FILE = "sent_coins.json"
TRADE_FILE = "active_trades.json"

COINS = [
    "bitcoin", "ethereum", "solana", "binancecoin", "ripple",
    "cardano", "dogecoin", "avalanche-2", "polkadot", "matic-network",
    "chainlink", "uniswap", "litecoin", "stellar", "cosmos",
    "near", "algorand", "vechain", "tezos", "flow"
]

def api_call(url, params=None, timeout=15):
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
        if r.status_code == 429:
            print("[RATE] 429 hit, waiting 60s...")
            time.sleep(60)
            r = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return None

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
    data = api_call(f"{BASE_URL}/coins/{cid}", params={"localization":"false","tickers":"false","community_data":"false","developer_data":"false","sparkline":"false"})
    if data and "market_data" in data:
        d = data["market_data"]
        return {"id": cid, "name": data.get("name",""), "symbol": data.get("symbol","").upper(), "price": d.get("current_price",{}).get("usd",0), "change": d.get("price_change_percentage_24h",0)}
    return None

def fetch_ohlcv(cid):
    data = api_call(f"{BASE_URL}/coins/{cid}/ohlc", params={"vs_currency":"usd","days":3})
    if data:
        return [c[4] for c in data]
    return None

def calc_rsi(prices):
    if not prices or len(prices) < 15:
        return None
    gains, losses = [], []
    for i in range(1, len(prices)):
        d = prices[i] - prices[i-1]
        gains.append(d if d > 0 else 0)
        losses.append(abs(d) if d < 0 else 0)
    avg_gain = sum(gains[-14:]) / 14
    avg_loss = sum(losses[-14:]) / 14
    if avg_loss == 0:
        return 100.0
    return round(100.0 - (100.0 / (1.0 + avg_gain / avg_loss)), 2)

def fetch_news(name):
    if not NEWS_API_KEY:
        return []
    try:
        r = requests.get("https://newsapi.org/v2/everything", params={"q":f"{name} crypto","apiKey":NEWS_API_KEY,"pageSize":3,"sortBy":"publishedAt","language":"en"}, timeout=10)
        if r.status_code == 200:
            return [a["title"] for a in r.json().get("articles", [])[:3]]
    except:
        pass
    return []

def get_signal(asset, rsi_val, news):
    nt = "\n".join([f"- {n}" for n in news]) if news else "No recent news"
    prompt = f"""Strict trading AI. Output ONLY valid JSON. No other text.

Rules:
- RSI < 40 AND positive news = BUY
- RSI > 60 AND negative news = SELL  
- RSI 40-60 OR mixed signals = NO TRADE
- Minimum 65% confidence required

{asset['name']} ({asset['symbol']})
Price: ${asset['price']:.6f}
RSI(14): {rsi_val}
24h Change: {asset['change']:.2f}%
News:
{nt}

Return JSON:
{{"action":"BUY","confidence":72,"reasoning":"short reason","entry":{asset['price']},"tp":{asset['price']*1.03:.2f},"sl":{asset['price']*0.985:.2f}}}"""
    try:
        resp = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        raw = resp.text.strip()
        for prefix in ["```json","```"]:
            raw = raw.replace(prefix,"")
        return json.loads(raw.strip())
    except Exception as e:
        print(f"[AI] Error: {e}")
    return None

def send_tg(msg):
    try:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage", json={"chat_id":TELEGRAM_CHAT_ID,"text":msg}, timeout=10)
    except:
        pass

def fetch_price(cid):
    data = api_call(f"{BASE_URL}/simple/price", params={"ids":cid,"vs_currencies":"usd"})
    if data and cid in data:
        return data[cid].get("usd")
    return None

def scan_signals():
    sent = load_json(SENT_FILE)
    trades = load_json(TRADE_FILE)
    found = 0
    print(f"[SCAN] Starting... ({len(COINS)} coins, 3s/coin)")
    for cid in COINS:
        if cid in sent:
            continue
        print(f"  [{cid}] ", end="", flush=True)
        m = fetch_market(cid)
        if not m or not m.get("price"):
            print("NO DATA")
            time.sleep(2)
            continue
        time.sleep(3)
        ohlcv = fetch_ohlcv(cid)
        if not ohlcv:
            print("NO OHLCV")
            time.sleep(2)
            continue
        rsi = calc_rsi(ohlcv)
        if rsi is None:
            print("RSI FAIL")
            time.sleep(2)
            continue
        if 40 <= rsi <= 60:
            print(f"RSI {rsi} → NEUTRAL")
            time.sleep(2)
            continue
        news = fetch_news(m["name"])
        signal = get_signal(m, rsi, news)
        if signal and signal.get("action") in ["BUY","SELL"]:
            print(f"RSI {rsi} → {signal['action']} ({signal['confidence']}%)")
            nl = f"\n📰 {news[0]}" if news else ""
            msg = f"""⚡ AETHER ASCENT

📊 {m['name']} ({m['symbol']})
🎯 {signal['action']} | {signal['confidence']}% | RSI {rsi}

💰 Entry: ${signal['entry']:,.2f}
✅ TP: ${signal['tp']:,.2f}
🛑 SL: ${signal['sl']:,.2f}

📝 {signal['reasoning']}{nl}

🕐 {datetime.utcnow().strftime('%d/%m %H:%M UTC')}"""
            send_tg(msg)
            sent.append(cid)
            save_json(SENT_FILE, sent[-50:])
            trades.append({"coin":cid,"symbol":m["symbol"],"action":signal["action"],"entry":signal["entry"],"tp":signal["tp"],"sl":signal["sl"],"time":time.time()})
            save_json(TRADE_FILE, trades)
            found += 1
        else:
            print(f"RSI {rsi} → NO TRADE")
        time.sleep(2)
        if found >= 3:
            break
    print(f"[SCAN] Complete. Signals: {found}")

def check_trades():
    trades = load_json(TRADE_FILE)
    if not trades:
        return
    updated = []
    for t in trades:
        if t.get("done"):
            if time.time() - t["done"] < 86400:
                updated.append(t)
            continue
        px = fetch_price(t["coin"])
        if not px:
            updated.append(t)
            continue
        if t["action"] == "BUY":
            if px >= t["tp"]:
                send_tg(f"✅ TP HIT!\n{t['symbol']} BUY\n${t['entry']:.2f}→${px:.2f}\nProfit: ${px-t['entry']:.2f}")
                t["done"], t["result"] = time.time(), "TP"
                print(f"[TP] {t['symbol']}")
            elif px <= t["sl"]:
                send_tg(f"🛑 SL HIT!\n{t['symbol']} BUY\n${t['entry']:.2f}→${px:.2f}\nLoss: ${t['entry']-px:.2f}")
                t["done"], t["result"] = time.time(), "SL"
                print(f"[SL] {t['symbol']}")
        elif t["action"] == "SELL":
            if px <= t["tp"]:
                send_tg(f"✅ TP HIT!\n{t['symbol']} SELL\n${t['entry']:.2f}→${px:.2f}\nProfit: ${t['entry']-px:.2f}")
                t["done"], t["result"] = time.time(), "TP"
                print(f"[TP] {t['symbol']}")
            elif px >= t["sl"]:
                send_tg(f"🛑 SL HIT!\n{t['symbol']} SELL\n${t['entry']:.2f}→${px:.2f}\nLoss: ${px-t['entry']:.2f}")
                t["done"], t["result"] = time.time(), "SL"
                print(f"[SL] {t['symbol']}")
        updated.append(t)
    save_json(TRADE_FILE, updated)

print("="*40)
print("⚡ AETHER ASCENT — CoinGecko API")
print("="*40)
print(f"API: {'Pro' if COINGECKO_API_KEY else 'Free'}")
print(f"Coins: {len(COINS)} | Scan: 15min | Per coin: 3s")
print(f"Rate Limit: 30/min | Usage: ~6/min → SAFE")
print("="*40)

try:
    scan_signals()
except Exception as e:
    print(f"[INIT] Error: {e}")

last_scan = time.time()

while True:
    try:
        check_trades()
        if time.time() - last_scan >= 900:
            scan_signals()
            last_scan = time.time()
    except Exception as e:
        print(f"[LOOP] Error: {e}")
    time.sleep(60)
