import streamlit as st
import requests
import pandas as pd
import ta
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

# List of available coins
coins = [
    "bitcoin", "ethereum", "ripple", "litecoin", "bitcoin-cash",
    "eos", "stellar", "cardano", "tron", "monero"
]

# Time frame options
time_frames = ["5m", "15m", "30m", "1h", "4h", "1d", "1M", "3M", "6M", "1Y"]

def fetch_price_data(coin_id, time_frame):
    if time_frame.endswith("m"):
        minutes = int(time_frame[:-1])
        days = (minutes * 60) / (24 * 60)  # Convert minutes to days
    elif time_frame.endswith("h"):
        hours = int(time_frame[:-1])
        days = hours / 24
    else:
        days = get_days_from_time_frame(time_frame)

    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days={days}"
    response = requests.get(url)
    data = response.json()["prices"]
    return data

def get_ema_signal(prices, window1=12, window2=26):
    prices_series = pd.Series([price[1] for price in prices])
    ema1 = prices_series.ewm(span=window1, adjust=False).mean()
    ema2 = prices_series.ewm(span=window2, adjust=False).mean()
    last_ema1 = ema1.iloc[-1]
    last_ema2 = ema2.iloc[-1]
    if last_ema1 > last_ema2:
        return "Buy"
    elif last_ema1 < last_ema2:
        return "Sell"
    else:
        return "Neutral"

def get_rsi_signal(prices, window=14):
    prices_series = pd.Series([price[1] for price in prices])
    delta = prices_series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    last_rsi = rsi.iloc[-1]
    if last_rsi < 20:
        return "Buy"
    elif last_rsi > 70:
        return "Sell"
    else:
        return "Neutral"

def get_stochastic_signal(prices, window=14, k_window=3, d_window=3):
    prices_series = pd.Series([price[1] for price in prices])
    stochastic = calculate_stochastic(prices_series, window, k_window, d_window)
    last_k = stochastic['k'].iloc[-1]
    last_d = stochastic['d'].iloc[-1]
    if last_k > 70 and last_d > 70:
        return "Sell"
    elif last_k < 20 and last_d < 20:
        return "Buy"
    else:
        return "Neutral"

def calculate_stochastic(prices, window, k_window, d_window):
    high = prices.rolling(window).max()
    low = prices.rolling(window).min()
    k = (prices - low) / (high - low) * 100
    k = k.rolling(k_window).mean()
    d = k.rolling(d_window).mean()
    stochastic = pd.DataFrame({"k": k, "d": d})
    return stochastic

def get_macd_signal(prices, fast=12, slow=26, signal=9):
    prices_series = pd.Series([price[1] for price in prices])
    macd_line = calculate_macd(prices_series, fast, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    last_macd = macd_line.iloc[-1]
    last_signal = signal_line.iloc[-1]
    if last_macd > last_signal:
        return "Buy"
    elif last_macd < last_signal:
        return "Sell"
    else:
        return "Neutral"

def calculate_macd(prices, fast, slow):
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    return macd_line

def get_bollinger_signal(prices, window=20, window_dev=2):
    prices_series = pd.Series([price[1] for price in prices])
    bollinger_bands = calculate_bollinger_bands(prices_series, window, window_dev)
    last_price = prices_series.iloc[-1]
    last_hband = bollinger_bands['hband'].iloc[-1]
    last_lband = bollinger_bands['lband'].iloc[-1]
    if last_price > last_hband:
        return "Sell"
    elif last_price < last_lband:
        return "Buy"
    else:
        return "Neutral"

def calculate_bollinger_bands(prices, window, window_dev):
    mid_band = prices.rolling(window).mean()
    std_dev = prices.rolling(window).std()
    hband = mid_band + std_dev * window_dev
    lband = mid_band - std_dev * window_dev
    bollinger_bands = pd.DataFrame({"hband": hband, "lband": lband})
    return bollinger_bands

def get_atr_signal(prices, window=14):
    prices_series = pd.Series([price[1] for price in prices])
    atr = calculate_atr(prices_series, window)
    last_atr = atr.iloc[-1]
    # You can use the ATR value to determine position sizing and stop-loss levels
    return "Neutral"

def calculate_atr(prices, window):
    true_range = np.maximum(
        prices.high - prices.low,
        np.abs(prices.high - np.roll(prices.close, 1)),
        np.abs(prices.low - np.roll(prices.close, 1))
    )
    atr = true_range.rolling(window).mean()
    return atr

def plot_price_chart(prices):
    dates = [datetime.fromtimestamp(price[0] / 1000) for price in prices]
    price_values = [price[1] for price in prices]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=price_values, mode='lines'))
    fig.update_layout(
        title='Price Chart',
        xaxis_title='Date',
        yaxis_title='Price (USD)'
    )

    st.plotly_chart(fig)

def combine_signals(signals):
    buy_count = signals.count("Buy")
    sell_count = signals.count("Sell")
    neutral_count = signals.count("Neutral")

    if buy_count > sell_count and buy_count > neutral_count:
        return "Buy"
    elif sell_count > buy_count and sell_count > neutral_count:
        return "Sell"
    else:
        return "Neutral"

def main():
    st.title("Crypto Signal Maker")

    # Dropdown menu for coin selection
    selected_coin = st.selectbox("Select a Coin", coins)

    # Dropdown menu for time frame selection
    selected_time_frame = st.selectbox("Select Time Frame", time_frames)

    if st.button("Generate Signal"):
        price_data = fetch_price_data(selected_coin, selected_time_frame)

        ema_signal = get_ema_signal(price_data)
        rsi_signal = get_rsi_signal(price_data)
        stochastic_signal = get_stochastic_signal(price_data)
        macd_signal = get_macd_signal(price_data)
        bollinger_signal = get_bollinger_signal(price_data)
        # atr_signal = get_atr_signal(price_data)

        st.write(f"EMA Signal is: {ema_signal}")
        st.write(f"RSI Signal is: {rsi_signal}")
        st.write(f"stochastic Signal is: {stochastic_signal}")
        st.write(f"MACD Signal is: {macd_signal}")
        st.write(f"Bollinger Signal is: {bollinger_signal}")
        signals = [ema_signal, rsi_signal, stochastic_signal, macd_signal, bollinger_signal]
        final_signal = combine_signals(signals)

        st.subheader(f"Signal for {selected_coin} ({selected_time_frame}): {final_signal}")
        plot_price_chart(price_data)

def get_days_from_time_frame(time_frame):
    if time_frame == "1M":
        return 30
    elif time_frame == "3M":
        return 90
    elif time_frame == "6M":
        return 180
    elif time_frame == "1Y":
        return 365
    else:
        return 30

if __name__ == "__main__":
    main()