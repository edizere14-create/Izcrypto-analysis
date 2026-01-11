import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from io import StringIO

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="IZ Crypto Analysis", layout="wide")

st.title("📊 IZ Crypto Analysis Dashboard")

# -----------------------------
# HELPERS
# -----------------------------
def fetch_kraken_ohlcv(pair="XXBTZUSD", interval=60, since=None, limit=500, retries=3):
    url = "https://api.kraken.com/0/public/OHLC"
    params = {"pair": pair, "interval": interval}

    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, params=params, timeout=10)
            data = r.json()

            if data.get("error"):
                raise ValueError(data["error"])

            key = list(data["result"].keys())[0]
            ohlc = data["result"][key]

            if not ohlc:
                raise ValueError("Empty OHLC data returned")

            cols = ["timestamp", "open", "high", "low", "close", "vwap", "volume", "count"]
            df = pd.DataFrame(ohlc, columns=cols)

            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
            for col in ["open", "high", "low", "close", "vwap", "volume", "count"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            df = df.dropna()
            df = df[["timestamp", "open", "high", "low", "close", "volume"]]
            df = df.tail(limit)

            if len(df) > 0:
                return df

        except Exception:
            if attempt == retries:
                raise

    raise ValueError("Failed to fetch valid data after retries")


def compute_rsi(series, length=14):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain = pd.Series(gain, index=series.index)
    loss = pd.Series(loss, index=series.index)
    avg_gain = gain.rolling(length).mean()
    avg_loss = loss.rolling(length).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - signal_line
    return macd, signal_line, hist


def compute_obv(close, volume):
    direction = np.sign(close.diff()).fillna(0)
    obv = (direction * volume).cumsum()
    return obv


def find_divergences(price, indicator, lookback=5):
    div_points = []
    n = len(price)

    for i in range(lookback, n - lookback):
        # Define window safely
        price_window = price[i - lookback : i + lookback + 1]
        ind_window = indicator[i - lookback : i + lookback + 1]

        # Skip if windows are empty
        if len(price_window) == 0 or len(ind_window) == 0:
            continue

        # Bearish divergence
        if price[i] == price_window.max():
            for j in range(i - lookback, i):
                prev_window = price[j - lookback : j + lookback + 1]
                if len(prev_window) == 0:
                    continue
                if price[j] == prev_window.max() and indicator[i] < indicator[j]:
                    div_points.append((i, "bearish"))
                    break

        # Bullish divergence
        if price[i] == price_window.min():
            for j in range(i - lookback, i):
                prev_window = price[j - lookback : j + lookback + 1]
                if len(prev_window) == 0:
                    continue
                if price[j] == prev_window.min() and indicator[i] > indicator[j]:
                    div_points.append((i, "bullish"))
                    break

    return div_points



def backtest_strategy(df, signal_col, fee=0.001):
    df = df.copy()
    df["returns"] = df["close"].pct_change().fillna(0)
    df["position"] = df[signal_col].shift(1).fillna(0)
    # Apply fees when position changes
    df["trade"] = df["position"].diff().abs().fillna(0)
    df["strategy_ret"] = df["position"] * df["returns"] - df["trade"] * fee
    df["equity"] = (1 + df["strategy_ret"]).cumprod()
    return df


def add_buy_sell_markers(fig, df, signal_col):
    df = df.copy()
    df["pos"] = df[signal_col].shift(1).fillna(0)
    df["change"] = df["pos"].diff()
    buys = df[df["change"] > 0]
    sells = df[df["change"] < 0]

    fig.add_trace(
        go.Scatter(
            x=buys["timestamp"],
            y=buys["close"],
            mode="markers",
            marker=dict(symbol="triangle-up", color="green", size=10),
            name="Buy",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sells["timestamp"],
            y=sells["close"],
            mode="markers",
            marker=dict(symbol="triangle-down", color="red", size=10),
            name="Sell",
        )
    )
    return fig


def download_link(df, filename):
    csv = df.to_csv(index=False)
    return csv, filename


# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("Navigation")

auto_refresh = st.sidebar.checkbox("Auto-refresh live data")
refresh_interval = st.sidebar.number_input("Refresh every (seconds)", 5, 300, 30)


page = st.sidebar.radio(
    "Go to",
    [
        "Home",
        "Indicators",
        "Backtesting",
        "Live Data",
        "Downloads",
    ],
)

st.sidebar.markdown("---")

data_source = st.sidebar.selectbox("Data source", ["Upload CSV", "Kraken API"])

pair = st.sidebar.text_input("Kraken pair (for API)", "XXBTZUSD")
interval = st.sidebar.selectbox("Kraken interval (minutes)", [1, 5, 15, 60, 240, 1440], index=3)
auto_refresh = st.sidebar.checkbox("Auto-refresh live data")
refresh_interval = st.sidebar.number_input("Refresh every (seconds)", 5, 300, 30)



# -----------------------------
# DATA LOADING
# -----------------------------
df = None

if data_source == "Upload CSV":
    uploaded = st.sidebar.file_uploader("Upload OHLCV CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        # Try to normalize columns
        cols = [c.lower() for c in df.columns]
        df.columns = cols
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        else:
            st.error("CSV must contain a 'timestamp' column.")
            df = None
        required = {"open", "high", "low", "close", "volume"}
        if not required.issubset(set(df.columns)):
            st.error(f"CSV must contain columns: {required}")
            df = None
else:
    if auto_refresh:
        try:
            df = fetch_kraken_ohlcv(pair=pair, interval=interval)
            st.sidebar.success("Auto-refreshed data")
        except Exception as e:
            st.sidebar.error(f"Error fetching data: {e}")
            df = None

        # Trigger refresh
        import time
        time.sleep(refresh_interval)
        st.experimental_rerun()

    else:
        if st.sidebar.button("Fetch from Kraken"):
            try:
                df = fetch_kraken_ohlcv(pair=pair, interval=interval)
                st.sidebar.success("Fetched data from Kraken.")
            except Exception as e:
                st.sidebar.error(f"Error fetching data: {e}")
                df = None


if df is not None:
    df = df.sort_values("timestamp").reset_index(drop=True)

else:
    if auto_refresh:
        try:
            df = fetch_kraken_ohlcv(pair=pair, interval=interval)
            st.sidebar.success("Auto-refreshed data")
            st.experimental_rerun()
        except Exception as e:
            st.sidebar.error(f"Error fetching data: {e}")
            df = None
    else:
        if st.sidebar.button("Fetch from Kraken"):
            try:
                df = fetch_kraken_ohlcv(pair=pair, interval=interval)
                st.sidebar.success("Fetched data from Kraken.")
            except Exception as e:
                st.sidebar.error(f"Error fetching data: {e}")
                df = None


# ============================
# DEBUG PANEL
# ============================
    with st.expander("🔍 Debug Panel: Kraken Data"):
        st.write("**Number of rows:**", len(df))
        st.write("**DataFrame head:**")
        st.dataframe(df.head())
        st.write("**DataFrame tail:**")
        st.dataframe(df.tail())
        st.write("**DataFrame dtypes:**")
        st.write(df.dtypes)


# -----------------------------
# HOME PAGE
# -----------------------------
if page == "Home":
    st.subheader("Overview")

    if df is None:
        st.info("Upload a CSV or fetch data from Kraken to begin.")
    else:
        st.write("Data preview:")
        st.dataframe(df.head())

        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=df["timestamp"],
                    open=df["open"],
                    high=df["high"],
                    low=df["low"],
                    close=df["close"],
                    name="Price",
                )
            ]
        )
        fig.update_layout(height=500, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# INDICATORS PAGE
# -----------------------------
elif page == "Indicators":
    if df is None:
        st.info("Upload or fetch data first.")
    else:
        st.subheader("Indicators & Divergences")

        col1, col2 = st.columns(2)

        with col1:
            ema_len = st.slider("EMA Length", 5, 200, 50)
        with col2:
            rsi_len = st.slider("RSI Length", 5, 50, 14)

        df["EMA"] = df["close"].ewm(span=ema_len, adjust=False).mean()
        df["RSI"] = compute_rsi(df["close"], rsi_len)
        df["MACD"], df["MACD_signal"], df["MACD_hist"] = compute_macd(df["close"])
        df["OBV"] = compute_obv(df["close"], df["volume"])

        # Candlestick + EMA
        st.markdown("### Price with EMA")
        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=df["timestamp"],
                    open=df["open"],
                    high=df["high"],
                    low=df["low"],
                    close=df["close"],
                    name="Price",
                )
            ]
        )
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["EMA"],
                name=f"EMA {ema_len}",
                line=dict(color="orange"),
            )
        )
        fig.update_layout(height=500, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

        # RSI with divergences
        st.markdown("### RSI with Divergences")
        rsi_divs = find_divergences(df["close"], df["RSI"])
        rsi_fig = go.Figure()
        rsi_fig.add_trace(go.Scatter(x=df["timestamp"], y=df["RSI"], name="RSI"))
        for idx, kind in rsi_divs:
            color = "green" if kind == "bullish" else "red"
            rsi_fig.add_trace(
                go.Scatter(
                    x=[df["timestamp"].iloc[idx]],
                    y=[df["RSI"].iloc[idx]],
                    mode="markers",
                    marker=dict(color=color, size=10),
                    name=f"{kind.capitalize()} div",
                )
            )
        rsi_fig.update_layout(height=250, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(rsi_fig, use_container_width=True)

        # MACD with divergences
        st.markdown("### MACD with Divergences")
        macd_divs = find_divergences(df["close"], df["MACD"])
        macd_fig = go.Figure()
        macd_fig.add_trace(go.Scatter(x=df["timestamp"], y=df["MACD"], name="MACD"))
        macd_fig.add_trace(
            go.Scatter(x=df["timestamp"], y=df["MACD_signal"], name="Signal")
        )
        for idx, kind in macd_divs:
            color = "green" if kind == "bullish" else "red"
            macd_fig.add_trace(
                go.Scatter(
                    x=[df["timestamp"].iloc[idx]],
                    y=[df["MACD"].iloc[idx]],
                    mode="markers",
                    marker=dict(color=color, size=10),
                    name=f"{kind.capitalize()} div",
                )
            )
        macd_fig.update_layout(height=250, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(macd_fig, use_container_width=True)

        # OBV with divergences
        st.markdown("### OBV with Divergences")
        obv_divs = find_divergences(df["close"], df["OBV"])
        obv_fig = go.Figure()
        obv_fig.add_trace(go.Scatter(x=df["timestamp"], y=df["OBV"], name="OBV"))
        for idx, kind in obv_divs:
            color = "green" if kind == "bullish" else "red"
            obv_fig.add_trace(
                go.Scatter(
                    x=[df["timestamp"].iloc[idx]],
                    y=[df["OBV"].iloc[idx]],
                    mode="markers",
                    marker=dict(color=color, size=10),
                    name=f"{kind.capitalize()} div",
                )
            )
        obv_fig.update_layout(height=250, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(obv_fig, use_container_width=True)

# -----------------------------
# BACKTESTING PAGE
# -----------------------------
elif page == "Backtesting":
    if df is None:
        st.info("Upload or fetch data first.")
    else:
        st.subheader("Strategy Backtesting & Portfolio Simulation")

        tab1, tab2 = st.tabs(["Strategies", "Portfolio"])

        with tab1:
            st.markdown("### Strategy Parameters")

            col1, col2, col3 = st.columns(3)
            with col1:
                fast_ma = st.slider("Fast EMA", 5, 50, 10)
                slow_ma = st.slider("Slow EMA", 20, 200, 50)
            with col2:
                rsi_len_bt = st.slider("RSI Length (BT)", 5, 50, 14)
                rsi_buy = st.slider("RSI Buy < ", 10, 50, 30)
                rsi_sell = st.slider("RSI Sell > ", 50, 90, 70)
            with col3:
                macd_fast = st.slider("MACD Fast", 5, 20, 12)
                macd_slow = st.slider("MACD Slow", 20, 40, 26)
                macd_signal = st.slider("MACD Signal", 5, 20, 9)

            fee = st.number_input("Trading fee (per side, decimal)", 0.0, 0.01, 0.001, 0.0001)

            # Compute indicators
            df["EMA_fast_bt"] = df["close"].ewm(span=fast_ma, adjust=False).mean()
            df["EMA_slow_bt"] = df["close"].ewm(span=slow_ma, adjust=False).mean()
            df["RSI_bt"] = compute_rsi(df["close"], rsi_len_bt)
            df["MACD_bt"], df["MACD_sig_bt"], _ = compute_macd(
                df["close"], fast=macd_fast, slow=macd_slow, signal=macd_signal
            )

            # Signals
            df["sig_ema"] = np.where(df["EMA_fast_bt"] > df["EMA_slow_bt"], 1, -1)
            df["sig_rsi"] = np.where(df["RSI_bt"] < rsi_buy, 1, np.where(df["RSI_bt"] > rsi_sell, -1, 0))
            df["sig_macd"] = np.where(df["MACD_bt"] > df["MACD_sig_bt"], 1, -1)

            strategy_choice = st.selectbox(
                "Strategy",
                ["EMA Crossover", "RSI", "MACD", "Combined (EMA + RSI + MACD)"],
            )

            if strategy_choice == "EMA Crossover":
                df["signal"] = df["sig_ema"]
            elif strategy_choice == "RSI":
                df["signal"] = df["sig_rsi"].replace(0, method="ffill").fillna(0)
            elif strategy_choice == "MACD":
                df["signal"] = df["sig_macd"]
            else:
                # Combined: majority vote
                sig_sum = df["sig_ema"] + df["sig_rsi"].replace(0, method="ffill").fillna(0) + df["sig_macd"]
                df["signal"] = np.where(sig_sum > 0, 1, np.where(sig_sum < 0, -1, 0))

            bt = backtest_strategy(df, "signal", fee=fee)

            st.markdown("### Equity Curve")
            st.line_chart(bt.set_index("timestamp")["equity"])

            st.markdown("### Price with Buy/Sell Markers")
            fig_bt = go.Figure(
                data=[
                    go.Candlestick(
                        x=bt["timestamp"],
                        open=bt["open"],
                        high=bt["high"],
                        low=bt["low"],
                        close=bt["close"],
                        name="Price",
                    )
                ]
            )
            fig_bt = add_buy_sell_markers(fig_bt, bt, "signal")
            fig_bt.update_layout(height=500, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_bt, use_container_width=True)

            st.markdown("### Strategy Stats")
            final_equity = bt["equity"].iloc[-1]
            total_return = (final_equity - 1) * 100
            st.write(f"**Final equity:** {final_equity:.2f}x")
            st.write(f"**Total return:** {total_return:.2f}%")

        with tab2:
            st.markdown("### Portfolio Simulation")

            start_balance = st.number_input("Starting balance ($)", 100.0, 1_000_000.0, 10_000.0, 100.0)
            risk_per_trade = st.number_input("Risk per trade (% of equity)", 0.1, 10.0, 1.0, 0.1)

            # Simple interpretation: equity curve * starting balance
            if "equity" in bt.columns:
                bt["portfolio_value"] = bt["equity"] * start_balance
                st.line_chart(bt.set_index("timestamp")["portfolio_value"])
                st.write(f"Final portfolio value: ${bt['portfolio_value'].iloc[-1]:,.2f}")
            else:
                st.info("Run a strategy first in the Strategies tab.")

# -----------------------------
# DOWNLOADS PAGE
# -----------------------------
elif page == "Downloads":
    if df is None:
        st.info("Upload or fetch data first.")
    else:
        st.subheader("Download Processed Data")

        st.write("You can download signals and equity curve after running a backtest on the Backtesting page.")

        # Recompute a default simple strategy for download if needed
        fast_ma = 10
        slow_ma = 50
        df["EMA_fast_bt"] = df["close"].ewm(span=fast_ma, adjust=False).mean()
        df["EMA_slow_bt"] = df["close"].ewm(span=slow_ma, adjust=False).mean()
        df["signal"] = np.where(df["EMA_fast_bt"] > df["EMA_slow_bt"], 1, -1)
        bt = backtest_strategy(df, "signal", fee=0.001)

        sig_csv, sig_name = download_link(bt[["timestamp", "close", "signal", "equity"]], "signals_equity.csv")
        st.download_button(
            label="Download signals & equity CSV",
            data=sig_csv,
            file_name=sig_name,
            mime="text/csv",
        )

        raw_csv, raw_name = download_link(df, "raw_ohlcv_processed.csv")
        st.download_button(
            label="Download processed OHLCV CSV",
            data=raw_csv,
            file_name=raw_name,
            mime="text/csv",
        )
# Streamlit's build-in refresh timer

if auto_refresh:
    st.experimental_rerun()
    time.sleep(refresh_interval)




