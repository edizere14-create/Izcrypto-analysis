def add_returns(df):
    df["return"] = df["close"].pct_change()
    return df

def add_moving_averages(df, windows=[20, 50, 200]):
    for w in windows:
        df[f"sma_{w}"] = df["close"].rolling(w).mean()
    return df

def add_volatility(df, window=20):
    df["volatility"] = df["return"].rolling(window).std()
    return df
