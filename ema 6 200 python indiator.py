from binance.client import Client
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ================= CONFIG =================
SYMBOL = "BTCUSDT"

TIMEFRAMES = {
    "5m": Client.KLINE_INTERVAL_5MINUTE,
    "15m": Client.KLINE_INTERVAL_15MINUTE,
    "1h": Client.KLINE_INTERVAL_1HOUR
}

EMA_FAST = 6
EMA_SLOW = 200
HULL_LEN = 55
CANDLE_LIMIT = 300

client = Client()

# ================= DATA =================
def fetch_data(interval):
    klines = client.futures_klines(
        symbol=SYMBOL,
        interval=interval,
        limit=CANDLE_LIMIT
    )

    df = pd.DataFrame(klines, columns=[
        "time","open","high","low","close","volume",
        "c1","c2","c3","c4","c5","c6"
    ])

    df["close"] = df["close"].astype(float)
    return df

# ================= INDICATORS =================
def ema(series, length):
    return series.ewm(span=length, adjust=False).mean()

def wma(series, length):
    weights = np.arange(1, length + 1)
    return series.rolling(length).apply(
        lambda x: np.dot(x, weights) / weights.sum(),
        raw=True
    )

def hull_ma(series, length):
    return wma(
        2 * wma(series, length // 2) - wma(series, length),
        int(np.sqrt(length))
    )

# ================= STRATEGY (LATEST SIGNAL ONLY) =================
def compute_latest_signal(df):
    close = df["close"]

    ema6 = ema(close, EMA_FAST)
    ema200 = ema(close, EMA_SLOW)
    hull = hull_ma(close, HULL_LEN)

    hull_bull = hull > hull.shift(2)
    hull_bear = hull < hull.shift(2)

    position = 0
    last_signal = None  # (index, price, label)

    for i in range(1, len(df)):
        # OPEN LONG
        if position == 0 and ema6[i-1] <= ema200[i-1] and ema6[i] > ema200[i]:
            position = 1
            last_signal = (i, close[i], "OPEN LONG")

        # CLOSE LONG
        elif position == 1 and hull_bear[i]:
            position = 0
            last_signal = (i, close[i], "CLOSE LONG")

        # OPEN SHORT
        elif position == 0 and ema6[i-1] >= ema200[i-1] and ema6[i] < ema200[i]:
            position = -1
            last_signal = (i, close[i], "OPEN SHORT")

        # CLOSE SHORT
        elif position == -1 and hull_bull[i]:
            position = 0
            last_signal = (i, close[i], "CLOSE SHORT")

    return ema6, ema200, last_signal

# ================= PLOT =================
fig, axes = plt.subplots(3, 1, figsize=(16, 10))

for ax, (tf_name, tf_interval) in zip(axes, TIMEFRAMES.items()):
    df = fetch_data(tf_interval)
    price = df["close"].values

    ema6, ema200, signal = compute_latest_signal(df)

    ax.plot(price, color="white", linewidth=1, label="Price")
    ax.plot(ema6, color="blue", linewidth=2, label="EMA 6")
    ax.plot(ema200, color="orange", linewidth=2, label="EMA 200")

    # ---- Plot ONLY latest signal ----
    if signal:
        i, price_val, label = signal

        color_map = {
            "OPEN LONG": "green",
            "CLOSE LONG": "lime",
            "OPEN SHORT": "red",
            "CLOSE SHORT": "yellow"
        }

        marker_map = {
            "OPEN LONG": "^",
            "CLOSE LONG": "x",
            "OPEN SHORT": "v",
            "CLOSE SHORT": "x"
        }

        ax.scatter(
            i, price_val,
            color=color_map[label],
            marker=marker_map[label],
            s=140,
            zorder=5
        )

        ax.set_title(
            f"{SYMBOL} {tf_name} | {label}",
            color=color_map[label],
            fontsize=11
        )
    else:
        ax.set_title(f"{SYMBOL} {tf_name} | NO SIGNAL", fontsize=11)

    ax.grid(alpha=0.25)
    ax.legend()

plt.tight_layout()
plt.show()
