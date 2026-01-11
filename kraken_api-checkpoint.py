import requests
import pandas as pd

BASE_URL = "https://api.kraken.com/0/public/OHLC"

def fetch_ohlcv(pair="BTCUSD", interval=1, since=None):
    params = {
        "pair": pair,
        "interval": interval,
    }
    if since:
        params["since"] = since

    response = requests.get(BASE_URL, params=params).json()

    if response.get("error"):
        raise Exception(response["error"])

    key = list(response["result"].keys())[0]
    raw = response["result"][key]

    df = pd.DataFrame(raw, columns=[
        "timestamp", "open", "high", "low", "close",
        "vwap", "volume", "count"
    ])

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df = df.set_index("timestamp")

    return df

