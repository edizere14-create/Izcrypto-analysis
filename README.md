# Crypto Analysis Dashboard

# Live Crypto Trading Bot

A comprehensive, production-ready cryptocurrency trading bot with technical analysis, risk management, and safety features.

## Features

- **Multiple Trading Strategies**: MA Crossover, RSI, and Combined strategies
- **Risk Management**: Position sizing, stop loss, take profit, drawdown protection
- **Technical Indicators**: RSI, MACD, Bollinger Bands, ATR, Moving Averages
- **Paper Trading Mode**: Test strategies without risking real money
- **Live Trading**: Connect to real exchanges (Kraken, Binance, etc.) via CCXT
- **Comprehensive Logging**: All trades and events are logged
- **Safety Features**: Maximum positions, drawdown limits, position size limits

## Installation

1. Install required packages:
```bash
pip install ccxt pandas numpy
```

2. (Optional) Install TA-Lib for advanced indicators:
```bash
# For Anaconda users (recommended on Windows):
conda install -c conda-forge ta-lib

# Or via pip (may require TA-Lib C library):
pip install TA-Lib
```

## Configuration

1. Edit `trading_config.json`:

```json
{
    "exchange": "kraken",           // Exchange name (kraken, binance, etc.)
    "api_key": "your_api_key",      // Your API key (leave empty for paper trading)
    "api_secret": "your_secret",    // Your API secret (leave empty for paper trading)
    "symbol": "BTC/USD",           // Trading pair
    "timeframe": "1h",              // Timeframe (1m, 5m, 15m, 1h, 4h, 1d, etc.)
    "strategy": "combined",         // Strategy: "ma_cross", "rsi", or "combined"
    "paper_trading": true,          // Set to false for live trading (USE WITH CAUTION!)
    "initial_balance": 10000.0,     // Starting balance
    "risk_per_trade": 0.02,         // Risk 2% per trade
    "max_position_size": 0.1,       // Maximum 10% of balance per position
    "max_drawdown": 0.15,           // Stop trading if 15% drawdown
    "max_open_positions": 5         // Maximum concurrent positions
}
```

## Usage

### Paper Trading (Recommended for Testing)

1. Make sure `paper_trading` is set to `true` in the config
2. Run the bot:

```bash
python run_trading_bot.py
```

Or with custom parameters:

```bash
python run_trading_bot.py --duration 120 --interval 300
```

This runs for 120 minutes, checking every 5 minutes (300 seconds).

### Live Trading (USE WITH EXTREME CAUTION!)

1. **IMPORTANT**: Test thoroughly in paper trading mode first!
2. Set `paper_trading` to `false` in config
3. Add your API credentials (keep them secure!)
4. Start with small amounts
5. Run:

```bash
python run_trading_bot.py
```

The bot will ask for confirmation before starting live trading.

## Strategies

### 1. MA Cross Strategy (`ma_cross`)
- Uses fast and slow moving averages
- Buys when fast MA crosses above slow MA
- Sells when fast MA crosses below slow MA

### 2. RSI Strategy (`rsi`)
- Uses Relative Strength Index
- Buys when RSI crosses above oversold level (30)
- Sells when RSI crosses below overbought level (70)

### 3. Combined Strategy (`combined`) - Recommended
- Combines RSI, MACD, and Bollinger Bands
- More robust signal generation
- Better risk management

## Risk Management

The bot includes several risk management features:

1. **Position Sizing**: Automatically calculates position size based on risk per trade and stop loss distance
2. **Stop Loss**: Automatically set based on ATR (Average True Range)
3. **Take Profit**: Set based on risk/reward ratio (default 2:1)
4. **Drawdown Protection**: Stops trading if drawdown exceeds maximum
5. **Position Limits**: Limits number of concurrent open positions

## Logging

All trading activity is logged to:
- Console output (real-time)
- Log files in `logs/` directory (one file per day)

Log files include:
- Trade entries and exits
- Order placements
- Errors and warnings
- Performance metrics

## Monitoring

The bot prints a summary when stopped:
- Initial and current balance
- Total PnL and return percentage
- Number of trades
- Win rate
- Open positions

## Safety Tips

1. **Always start with paper trading** to test your strategy
2. **Start with small amounts** when going live
3. **Monitor the bot** regularly, especially in the beginning
4. **Set appropriate risk parameters** (don't risk more than you can afford to lose)
5. **Use stop losses** - they're automatically calculated but can be adjusted
6. **Keep API keys secure** - never share them or commit to version control
7. **Test on testnet** if available for your exchange

## Customization

### Adding New Strategies

Create a new strategy class inheriting from `TradingStrategy`:

```python
class MyStrategy(TradingStrategy):
    def __init__(self):
        super().__init__("MyStrategy")
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        # Your signal generation logic
        signals = pd.Series(0, index=df.index)
        # ... your logic ...
        return signals
```

Then add it to the `_init_strategy` method in `LiveTradingBot`.

### Adjusting Indicators

Modify indicator parameters in the strategy classes or create custom indicator calculations in the `TechnicalIndicators` class.

## Troubleshooting

### "TA-Lib not available" warning
- Install TA-Lib (see Installation section)
- The bot will use pandas/numpy implementations as fallback

### API connection errors
- Check your API credentials
- Verify API permissions (trading enabled)
- Check exchange status
- Ensure you're not rate-limited

### No trades being executed
- Check if signals are being generated (check logs)
- Verify you have sufficient balance
- Check if maximum positions limit is reached
- Verify drawdown limits aren't triggered

## Disclaimer

**This software is for educational purposes only. Trading cryptocurrencies involves substantial risk of loss. Use at your own risk. The authors are not responsible for any financial losses.**

Always:
- Test thoroughly in paper trading mode
- Start with small amounts
- Understand the risks
- Never risk more than you can afford to lose
- Monitor your bot regularly

## License

This code is provided as-is for educational purposes.



