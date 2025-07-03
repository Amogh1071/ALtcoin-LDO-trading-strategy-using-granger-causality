# LDO Trading Strategy with Granger Causality

This repository contains a Python script (`strategy.py`) designed to generate buy, sell, and hold signals for trading the LDO cryptocurrency based on movements in BTC and ETH anchor coins. The strategy leverages Granger causality tests to dynamically determine optimal lag periods and incorporates adaptive technical indicators for robust performance across different timeframes (1H, 4H, 1D).

## Overview

The strategy is built to:
- Use BTC and ETH as anchor coins to predict LDO price movements.
- Dynamically adjust lag periods using a custom Granger causality implementation.
- Apply enhanced buy/sell rules with momentum, trend, volatility, and price action filters.
- Implement risk management with ATR-based stop-loss, take-profit, and trailing stop levels.
- Adapt to various market conditions with timeframe-specific parameters.

### Key Features
- **Granger Causality:** Dynamically computes the optimal lag for BTC and ETH to predict LDO returns.
- **Adaptive Indicators:** Adjusts moving averages, volatility filters, and momentum scores based on the chosen timeframe.
- **Risk Management:** Uses ATR multipliers for stop-loss (1.5x), take-profit (2.5x), and trailing stop (1x) levels.
- **Configurable Rules:** Allows customization of buy/sell thresholds and anchor coins via the configuration section.

## Requirements

- **Python 3.8+**
- **Libraries:**
  - `pandas`
  - `numpy`

Install required dependencies using:
```bash
pip install pandas numpy
```

## Setup

1. **Clone the Repository:**
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Configure the Strategy:**
   - Open `strategy.py` and edit the `CONFIGURATION` section to set:
     - `TARGET_COIN`: The cryptocurrency to trade (default: "LDO").
     - `TIMEFRAME`: The trading timeframe ("1H", "4H", or "1D", default: "1H").
     - `ANCHORS`: List of anchor coins with their timeframes and initial lag settings (default: BTC and ETH with 2-hour lag).
     - `BUY_RULES`: Conditions for buy signals (default: 0.8% change for BTC/ETH).
     - `SELL_RULES`: Conditions for sell signals (default: -1.0% change for BTC/ETH).

3. **Prepare Data:**
   - Ensure you have a data source providing OHLCV (open, high, low, close, volume) data for LDO, BTC, and ETH in the specified timeframe. The script expects a pandas DataFrame with columns `timestamp`, `close`, `high`, `low`, and optionally volume-related columns.

## Usage

### Running the Strategy

The strategy is designed to be integrated with a trading pipeline. Here's a basic example of how to use it:

```python
import pandas as pd
from strategy import generate_signals, get_coin_metadata

# Example DataFrames (replace with your data source)
candles_target = pd.DataFrame({
    'timestamp': [...],  # List of datetime objects
    'close': [...],      # List of closing prices
    'high': [...],       # List of high prices
    'low': [...]         # List of low prices
})

candles_anchor = pd.DataFrame({
    'timestamp': [...],  # List of datetime objects
    'close_BTC': [...],  # BTC closing prices
    'close_ETH': [...]   # ETH closing prices
})

# Generate signals
signals_df = generate_signals(candles_target, candles_anchor)

# Get metadata for evaluation
metadata = get_coin_metadata()

print(signals_df.head())  # Display first 5 rows of signals
```

### Output
- The `generate_signals` function returns a DataFrame with columns:
  - `timestamp`: The timestamp of each candle.
  - `signal`: One of "BUY", "SELL", or "HOLD".
  - (Diagnostic columns: `btc_causality_strength`, `eth_causality_strength`, `optimal_btc_lag`, `optimal_eth_lag` for analysis.)

### Integration
- Use the output signals with a trading simulator or live trading platform.
- The `get_coin_metadata` function provides metadata required by an evaluation engine (e.g., target coin and anchors).

## Strategy Details

### Signal Generation
- **Granger Causality:** Tests if BTC or ETH returns Granger-cause LDO returns to determine the optimal lag (up to ~24 hours).
- **Buy Conditions:** Requires:
  - BTC or ETH to exceed a 0.8% change at the optimal lag.
  - Confirmation from trend (dual MA crossover), momentum score (>0), low volatility (or strong causality), and recent price increase.
  - A confidence score > 60%, weighted by causality strength.
- **Sell Conditions:** Triggers on:
  - BTC or ETH dropping by -1.0% at the optimal lag.
  - Stop-loss (1.5x ATR), take-profit (2.5x ATR), trailing stop (1x ATR), or trend breakdown after minimum hold period.

### Risk Management
- **Stop-Loss:** 1.5x ATR below entry price.
- **Take-Profit:** 2.5x ATR above entry price.
- **Trailing Stop:** 1x ATR below the highest price reached.
- **Minimum Hold:** 1 candle duration to avoid whipsaws.

### Adaptive Parameters
- Moving average windows, volatility filters, and momentum lookbacks scale with the timeframe (1H, 4H, 1D).

## Performance Considerations

- **Expected Metrics:** The strategy aims for Profitability ~55–56%, Sharpe Ratio ~3.0–3.2, and Max Drawdown ~8–9%. Current performance may vary based on market conditions.
- **Optimization:** Adjust `BUY_RULES`, `SELL_RULES`, and causality thresholds in the configuration section to optimize for your risk tolerance and market environment.

## Troubleshooting

- **No Signals Generated:** Check if data contains enough rows (>30) and if Granger causality detects significant relationships. Relax `BUY_RULES` thresholds or increase data range.
- **Errors:** Ensure column names match the expected format (e.g., `close_BTC`, `close_ETH`). Review error messages for specific issues.

## Contributing

Feel free to fork this repository, submit pull requests, or open issues for enhancements (e.g., adding volume filters, improving causality tests, or supporting more coins).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

```

### Notes
- **Structure:** The README follows a standard format with sections for overview, requirements, setup, usage, strategy details, performance, troubleshooting, contributing, and licensing.
- **Customization:** You can add a repository URL, a specific license file, or additional sections (e.g., a changelog) if needed.
- **Integration:** The usage example assumes a data source; you may want to link it to your existing `trading_pipeline.py` for a complete workflow.
- **Date Awareness:** The README does not include the current date (July 03, 2025) explicitly, as it’s a static document, but you can update the "Performance Considerations" section with the latest metrics if desired.

Save this content as `README.md` in the same directory as `strategy.py`. Let me know if you'd like to adjust or expand any section!
