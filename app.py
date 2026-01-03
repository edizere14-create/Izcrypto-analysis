import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Crypto Analysis Dashboard", layout="wide")

st.title("📊 Crypto Analysis Dashboard")

# -----------------------------
# FILE UPLOAD
# -----------------------------
uploaded = st.file_uploader("Upload OHLCV CSV", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)

    # Ensure timestamp is datetime
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    st.subheader("Raw Data")
    st.dataframe(df.head())

    # -----------------------------
    # CANDLESTICK CHART
    # -----------------------------
    st.subheader("📈 Candlestick Chart")

    fig = go.Figure(data=[
        go.Candlestick(
            x=df["timestamp"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Price"
        )
    ])

    # -----------------------------
    # INDICATORS
    # -----------------------------
    st.subheader("📊 Indicators")

    # EMA
    ema_len = st.slider("EMA Length", 5, 200, 50)
    df["EMA"] = df["close"].ewm(span=ema_len).mean()
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["EMA"], name=f"EMA {ema_len}"))

    # RSI
    rsi_len = st.slider("RSI Length", 5, 50, 14)
    delta = df["close"].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(rsi_len).mean()
    avg_loss = pd.Series(loss).rolling(rsi_len).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    st.line_chart(df["RSI"], height=200)

    # MACD
    fast = 12
    slow = 26
    signal = 9
    df["EMA_fast"] = df["close"].ewm(span=fast).mean()
    df["EMA_slow"] = df["close"].ewm(span=slow).mean()
    df["MACD"] = df["EMA_fast"] - df["EMA_slow"]
    df["Signal"] = df["MACD"].ewm(span=signal).mean()

    st.line_chart(df[["MACD", "Signal"]], height=200)

    # OBV
    df["OBV"] = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()
    st.line_chart(df["OBV"], height=200)

    # -----------------------------
    # STRATEGY BACKTEST
    # -----------------------------
    st.subheader("🧪 Moving Average Strategy Backtest")

    fast_ma = st.slider("Fast MA", 5, 50, 10)
    slow_ma = st.slider("Slow MA", 20, 200, 50)

    df["fast_ma"] = df["close"].rolling(fast_ma).mean()
    df["slow_ma"] = df["close"].rolling(slow_ma).mean()

    df["signal"] = np.where(df["fast_ma"] > df["slow_ma"], 1, -1)
    df["returns"] = df["close"].pct_change()
    df["strategy"] = df["signal"].shift(1) * df["returns"]
    df["equity"] = (1 + df["strategy"]).cumprod()

    st.subheader("📈 Equity Curve")
    st.line_chart(df["equity"], height=300)

    # -----------------------------
    # FINAL CHART RENDER
    # -----------------------------
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Upload a CSV file to begin.")
