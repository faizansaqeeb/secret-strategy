from binance.client import Client
import pandas as pd
import numpy as np
import time
import requests

# ================= TELEGRAM =================
TELEGRAM_TOKEN = "8565575662:AAGkqeUhSI0qXzXBFDdzIgEzR4gzm2iohAw"
TELEGRAM_CHAT_ID = "2137177601"

def send_telegram(msg):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"}
    try:
        requests.post(url, json=payload, timeout=5)
    except:
        pass

# ================= CONFIG =================
TOP_COINS = 500
CANDLE_LIMIT = 220

TIMEFRAMES = {
    "5m":  {"interval": Client.KLINE_INTERVAL_5MINUTE, "seconds": 300, "fresh": 25},
    "15m": {"interval": Client.KLINE_INTERVAL_15MINUTE, "seconds": 900, "fresh": 90}
}

HULL_LENGTH = 55
EMA_PERIOD = 200
TOUCH_ATR_MULT = 0.30

client = Client()

# ================= SYMBOL LIST =================
def get_top_symbols():
    df = pd.DataFrame(client.futures_ticker())
    df = df[df["symbol"].str.endswith("USDT")]
    df["vol"] = df["quoteVolume"].astype(float)
    return df.sort_values("vol", ascending=False)["symbol"].head(TOP_COINS).tolist()

# ================= DATA =================
def fetch_data(symbol, interval):
    klines = client.futures_klines(symbol=symbol, interval=interval, limit=CANDLE_LIMIT)
    df = pd.DataFrame(klines, columns=[
        "time","open","high","low","close","volume",
        "x1","x2","x3","x4","x5","x6"
    ])
    df[["open","high","low","close","volume"]] = df[
        ["open","high","low","close","volume"]
    ].astype(float)
    return df.iloc[:-1]

# ================= INDICATORS =================
def ema(s, l):
    return s.ewm(span=l, adjust=False).mean()

def wma(s, l):
    w = np.arange(1, l + 1)
    return s.rolling(l).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)

def hma(s, l):
    return wma(2 * wma(s, l // 2) - wma(s, l), int(np.sqrt(l)))

def atr(df, l=14):
    pc = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - pc).abs(),
        (df["low"] - pc).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/l, adjust=False).mean()

def adx(df, l=14):
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0)
    minus_dm = np.where((down > up) & (down > 0), down, 0)
    tr = atr(df, l)
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/l).mean() / tr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/l).mean() / tr
    dx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100
    return dx.ewm(alpha=1/l).mean()

def rsi(s, l=14):
    d = s.diff()
    g = d.clip(lower=0).ewm(alpha=1/l).mean()
    l_ = (-d.clip(upper=0)).ewm(alpha=1/l).mean()
    rs = g / l_
    return 100 - (100 / (1 + rs))

# ================= SUCCESS RATE =================
def success_rate(direction, df):
    score = 0

    if adx(df).iloc[-1] >= 18: score += 25
    if atr(df).iloc[-1] / df["close"].iloc[-1] >= 0.002: score += 20

    r = rsi(df["close"]).iloc[-1]
    if direction == "LONG" and 45 <= r <= 70: score += 20
    if direction == "SHORT" and 30 <= r <= 55: score += 20

    if df["volume"].iloc[-1] > df["volume"].rolling(20).mean().iloc[-1]:
        score += 15

    return min(score, 100)

# ================= MAIN LOOP =================
symbols = get_top_symbols()
send_telegram("🚀 HULL EMA 200 TOUCH SCANNER STARTED")

while True:
    now = time.time()

    for symbol in symbols:
        for tf, cfg in TIMEFRAMES.items():
            try:
                df = fetch_data(symbol, cfg["interval"])

                close = df["close"]
                ema200 = ema(close, EMA_PERIOD)
                hull = hma(close, HULL_LENGTH)
                shull = hull.shift(2)

                atr_val = atr(df).iloc[-1]

                # Touch detection
                if abs(hull.iloc[-1] - ema200.iloc[-1]) > TOUCH_ATR_MULT * atr_val:
                    continue

                # Direction
                direction = "LONG" if hull.iloc[-1] > shull.iloc[-1] else "SHORT"

                # Freshness
                candle_close = df["time"].iloc[-1] / 1000 + cfg["seconds"]
                age = int(now - candle_close)
                if age < 0 or age > cfg["fresh"]:
                    continue

                score = success_rate(direction, df)
                if score < 70:
                    continue

                entry = close.iloc[-1]
                sl = entry - 2.5 * atr_val if direction == "LONG" else entry + 2.5 * atr_val
                tp1 = entry + 2 * atr_val if direction == "LONG" else entry - 2 * atr_val
                tp2 = entry + 4 * atr_val if direction == "LONG" else entry - 4 * atr_val

                pos_size = "2.5%" if score < 80 else "3.5%" if score < 90 else "5%"
                leverage = "5x" if score < 80 else "7x" if score < 90 else "10x"

                msg = (
                    f"🔥 <b>{direction} HULL EMA 200 TOUCH</b>\n\n"
                    f"🪙 <b>Coin:</b> {symbol}\n"
                    f"⏱ <b>TF:</b> {tf}\n"
                    f"⏳ <b>Fresh:</b> {age}s\n"
                    f"🎯 <b>Score:</b> {score}%\n"
                    f"📍 <b>Entry:</b> {entry:.6f}\n"
                    f"🛑 <b>SL:</b> {sl:.6f}\n"
                    f"💰 <b>TP1:</b> {tp1:.6f}\n"
                    f"💰 <b>TP2:</b> {tp2:.6f}\n"
                    f"📦 <b>Size:</b> {pos_size}\n"
                    f"⚡ <b>Leverage:</b> {leverage}\n"
                )

                print(msg.replace("<b>", "").replace("</b>", ""))
                send_telegram(msg)

            except Exception as e:
                print(symbol, tf, "error:", e)

    time.sleep(5)
