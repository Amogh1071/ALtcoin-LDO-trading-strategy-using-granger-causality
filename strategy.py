
import pandas as pd
import numpy as np

# ========== CONFIGURATION (EDIT THIS SECTION ONLY) ==========

# TARGET COIN to trade (this is the coin you'll generate BUY/SELL/HOLD for)
TARGET_COIN = "LDO"
TIMEFRAME = "1H"  # Timeframe of your strategy: "1H", "4H", "1D"

# ANCHOR COINS used to derive signals (these are the coins you observe for movement)
# LAG means how many candles back to look when calculating % change
ANCHORS = [
    {"symbol": "BTC", "timeframe": "1H", "lag": 2},  # BTC with 2-hour lag
    {"symbol": "ETH", "timeframe": "1H", "lag": 2},  # ETH with 2-hour lag
]

# BUY RULES: More relaxed thresholds based on Granger causality
BUY_RULES = [
    {"symbol": "BTC", "timeframe": "1H", "lag": 2, "change_pct": 0.8, "direction": "up"},
    {"symbol": "ETH", "timeframe": "1H", "lag": 2, "change_pct": 0.8, "direction": "up"},
]

# SELL RULES: Defensive exits
SELL_RULES = [
    {"symbol": "BTC", "timeframe": "1H", "lag": 2, "change_pct": -1.0, "direction": "down"},
    {"symbol": "ETH", "timeframe": "1H", "lag": 2, "change_pct": -1.0, "direction": "down"},
]

# ========== STRATEGY ENGINE (DO NOT EDIT BELOW UNLESS NECESSARY) ==========

def get_column_name(candles_anchor: pd.DataFrame, symbol: str, timeframe: str) -> str:
    """
    Dynamically determine the column name for an anchor coin's close price.
    """
    expected_col = f"close_{symbol}_{timeframe}"
    fallback_col = f"close_{symbol}"
    if expected_col in candles_anchor.columns:
        return expected_col
    elif fallback_col in candles_anchor.columns:
        return fallback_col
    else:
        raise ValueError(f"Neither {expected_col} nor {fallback_col} found in candles_anchor columns")

def get_timeframe_hours(timeframe: str) -> int:
    """
    Convert timeframe to hours per candle.
    """
    if timeframe == "1H":
        return 1
    elif timeframe == "4H":
        return 4
    elif timeframe == "1D":
        return 24
    else:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

def granger_causality_test(y, x, max_lags: int):
    """
    Simple Granger causality test implementation without external libraries.
    Tests if x Granger-causes y.
    Returns: (f_statistic, optimal_lag, causality_strength)
    """
    n = len(y)
    if n < max_lags + 10:  # Need sufficient data
        return 0, 1, 0

    best_f_stat = 0
    best_lag = 1

    for lag in range(1, min(max_lags + 1, n // 4)):
        try:
            # Create lagged variables
            y_lagged = []
            x_lagged = []
            y_current = []

            for i in range(lag, n):
                y_lag_vals = [y.iloc[i - j - 1] for j in range(lag)]
                x_lag_vals = [x.iloc[i - j - 1] for j in range(lag)]
                y_lagged.append(y_lag_vals)
                x_lagged.append(x_lag_vals)
                y_current.append(y.iloc[i])

            if len(y_current) < 10:
                continue

            y_current = np.array(y_current)
            y_lagged = np.array(y_lagged)
            x_lagged = np.array(x_lagged)

            # Restricted model: y regressed on its own lags
            X_restricted = np.column_stack([np.ones(len(y_current)), y_lagged])

            # Unrestricted model: y regressed on its own lags + x lags
            X_unrestricted = np.column_stack([np.ones(len(y_current)), y_lagged, x_lagged])

            # Calculate R-squared for both models using simple linear algebra
            try:
                # Restricted model R²
                beta_r = np.linalg.lstsq(X_restricted, y_current, rcond=None)[0]
                y_pred_r = X_restricted @ beta_r
                ss_res_r = np.sum((y_current - y_pred_r) ** 2)
                ss_tot = np.sum((y_current - np.mean(y_current)) ** 2)
                r2_restricted = 1 - (ss_res_r / ss_tot) if ss_tot > 0 else 0

                # Unrestricted model R²
                beta_u = np.linalg.lstsq(X_unrestricted, y_current, rcond=None)[0]
                y_pred_u = X_unrestricted @ beta_u
                ss_res_u = np.sum((y_current - y_pred_u) ** 2)
                r2_unrestricted = 1 - (ss_res_u / ss_tot) if ss_tot > 0 else 0

                # F-statistic calculation
                if r2_restricted < r2_unrestricted and ss_res_r > ss_res_u:
                    f_stat = ((ss_res_r - ss_res_u) / lag) / (ss_res_u / (len(y_current) - X_unrestricted.shape[1]))
                    f_stat = max(0, f_stat)  # Ensure non-negative
                else:
                    f_stat = 0

                if f_stat > best_f_stat:
                    best_f_stat = f_stat
                    best_lag = lag

            except np.linalg.LinAlgError:
                continue

        except Exception:
            continue

    # Normalize causality strength (0-1 scale)
    causality_strength = min(1.0, best_f_stat / 10.0) if best_f_stat > 0 else 0

    return best_f_stat, best_lag, causality_strength

def calculate_momentum_score(prices, lookback_periods):
    """Calculate weighted momentum score across multiple timeframes"""
    momentum_scores = []
    weights = [0.4, 0.3, 0.2, 0.1]  # More weight on recent periods

    for i, period in enumerate(lookback_periods):
        if len(prices) > period:
            momentum = prices.pct_change(periods=period).iloc[-1]
            if not pd.isna(momentum):
                momentum_scores.append(momentum * weights[i])

    return sum(momentum_scores) if momentum_scores else 0

def calculate_atr(df, period=14):
    """Calculate Average True Range (ATR) for dynamic risk management"""
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period, min_periods=1).mean()
    return atr

def generate_signals(candles_target: pd.DataFrame, candles_anchor: pd.DataFrame) -> pd.DataFrame:
    """
    Enhanced strategy with adaptive parameters for consistent performance across timeframes.
    """
    try:
        # Get column names
        btc_col = get_column_name(candles_anchor, "BTC", "1H")
        eth_col = get_column_name(candles_anchor, "ETH", "1H")

        # Merge data
        df = pd.merge(
            candles_target[['timestamp', 'close', 'high', 'low']],
            candles_anchor[['timestamp', btc_col, eth_col]],
            on='timestamp',
            how='inner'
        )

        data_length = len(df)
        if data_length < 30:
            raise ValueError(f"Insufficient data: only {data_length} rows")

        # Calculate timeframe in hours per candle
        hours_per_candle = get_timeframe_hours(TIMEFRAME)

        # Calculate returns for Granger causality
        df['ldo_returns'] = df['close'].pct_change()
        df['btc_returns'] = df[btc_col].pct_change()
        df['eth_returns'] = df[eth_col].pct_change()

        # Remove NaN values for causality tests
        clean_data = df.dropna()

        if len(clean_data) < 20:
            raise ValueError("Not enough clean data for analysis")

        # Adaptive Granger causality: max_lags to cover ~24 hours
        max_lags = max(1, int(24 / hours_per_candle))

        # Perform Granger causality tests
        btc_f_stat, btc_optimal_lag, btc_causality = granger_causality_test(
            clean_data['ldo_returns'], clean_data['btc_returns'], max_lags=max_lags
        )

        eth_f_stat, eth_optimal_lag, eth_causality = granger_causality_test(
            clean_data['ldo_returns'], clean_data['eth_returns'], max_lags=max_lags
        )

        # Use optimal lags from Granger causality for signals
        optimal_btc_lag = max(1, min(btc_optimal_lag, 4))
        optimal_eth_lag = max(1, min(eth_optimal_lag, 4))

        # Calculate technical indicators with adaptive parameters
        # Moving averages: ~5 hours (short), ~15 hours (long)
        window_short = max(5, min(12, int(5 / hours_per_candle)))
        window_long = max(15, min(30, int(15 / hours_per_candle)))

        # Simple moving averages
        df['ma_short'] = df['close'].rolling(window=window_short, min_periods=1).mean()
        df['ma_long'] = df['close'].rolling(window=window_long, min_periods=1).mean()
        df['trend'] = df['ma_short'] > df['ma_long']

        # Momentum indicators: lookback periods scaled to ~1-6 hours
        momentum_lookbacks = [max(1, int(h / hours_per_candle)) for h in [1, 2, 4, 6]]
        df['momentum_score'] = df['close'].rolling(window=10, min_periods=1).apply(
            lambda x: calculate_momentum_score(x, momentum_lookbacks), raw=False
        )

        # Volatility filter: ~12 hours for short window, ~24 hours for quantile
        vol_window = max(1, int(12 / hours_per_candle))
        vol_quantile_window = max(1, int(24 / hours_per_candle))
        df['volatility'] = df['ldo_returns'].rolling(window=vol_window, min_periods=1).std()
        # Dynamic quantile based on timeframe
        quantile_threshold = 0.7 if TIMEFRAME == "1H" else 0.65 if TIMEFRAME == "4H" else 0.6
        df['low_vol'] = df['volatility'] < df['volatility'].rolling(window=vol_quantile_window, min_periods=1).quantile(quantile_threshold)

        # Market regime detection: ~8 hours (short), ~20 hours (long)
        btc_ma_short_window = max(1, int(8 / hours_per_candle))
        btc_ma_long_window = max(1, int(20 / hours_per_candle))
        df['btc_ma_short'] = df[btc_col].rolling(window=btc_ma_short_window, min_periods=1).mean()
        df['btc_ma_long'] = df[btc_col].rolling(window=btc_ma_long_window, min_periods=1).mean()
        df['btc_uptrend'] = df['btc_ma_short'] > df['btc_ma_long']

        # Calculate ATR for dynamic risk management (~14 periods, scaled to timeframe)
        atr_window = max(1, int(14 / hours_per_candle))
        df['atr'] = calculate_atr(df, period=atr_window)

        # Enhanced signal generation using Granger causality insights
        has_position = False
        entry_price = None
        entry_time = None
        highest_price = None

        # Dynamic risk management parameters based on ATR
        atr_multiplier_sl = 1.5  # Stop-loss: 1.5x ATR
        atr_multiplier_tp = 2.5  # Take-profit: 2.5x ATR
        atr_multiplier_ts = 1.0  # Trailing stop: 1x ATR
        min_hold_hours = hours_per_candle  # 1 candle duration (1H -> 1 hour, 4H -> 4 hours, 1D -> 24 hours)

        # Adjust thresholds based on causality strength
        causality_multiplier = 1 + (btc_causality + eth_causality) * 0.5

        signals = []

        for i in range(len(df)):
            row = df.iloc[i]
            timestamp = row['timestamp']
            close_price = row['close']
            atr_value = row['atr'] if not pd.isna(row['atr']) else 0.01 * close_price  # Fallback ATR

            # Dynamic risk levels
            base_stop_loss = atr_value * atr_multiplier_sl / close_price
            base_take_profit = atr_value * atr_multiplier_tp / close_price
            base_trailing_stop = atr_value * atr_multiplier_ts / close_price

            # Initialize conditions
            buy_signal = False
            sell_signal = False

            # Basic buy rules from config with optimal lags
            btc_return_lag = df[btc_col].pct_change().shift(optimal_btc_lag).iloc[i] if i >= optimal_btc_lag else np.nan
            eth_return_lag = df[eth_col].pct_change().shift(optimal_eth_lag).iloc[i] if i >= optimal_eth_lag else np.nan

            # Apply BUY_RULES with Granger-optimal lags
            for rule in BUY_RULES:
                if rule['symbol'] == 'BTC' and not pd.isna(btc_return_lag):
                    if btc_return_lag > rule['change_pct'] / 100:
                        buy_signal = True
                        break
                elif rule['symbol'] == 'ETH' and not pd.isna(eth_return_lag):
                    if eth_return_lag > rule['change_pct'] / 100:
                        buy_signal = True
                        break

            # Enhanced buy conditions
            if buy_signal:
                # Trend confirmation
                trend_ok = row['trend'] and row['btc_uptrend']

                # Momentum confirmation
                momentum_ok = row['momentum_score'] > 0

                # Volatility filter
                vol_ok = row['low_vol'] or btc_causality > 0.3

                # Price action confirmation
                price_action_ok = df['close'].iloc[i] > df['close'].iloc[max(0, i - 2)]

                # Final buy decision with causality weighting
                buy_confidence = (
                    0.3 * (1 if trend_ok else 0) +
                    0.25 * (1 if momentum_ok else 0) +
                    0.2 * (1 if vol_ok else 0) +
                    0.25 * (1 if price_action_ok else 0)
                )

                # Weight by causality strength
                buy_confidence *= causality_multiplier

                buy_signal = buy_confidence > 0.6  # Require 60% confidence

            # Apply SELL_RULES with optimal lags
            for rule in SELL_RULES:
                if rule['symbol'] == 'BTC' and not pd.isna(btc_return_lag):
                    if btc_return_lag <= rule['change_pct'] / 100:
                        sell_signal = True
                        break
                elif rule['symbol'] == 'ETH' and not pd.isna(eth_return_lag):
                    if eth_return_lag <= rule['change_pct'] / 100:
                        sell_signal = True
                        break

            # Additional sell conditions
            if has_position:
                hours_held = (timestamp - entry_time).total_seconds() / 3600 if entry_time else 0

                # Update highest price
                highest_price = max(highest_price or entry_price, close_price)

                # Risk management
                if entry_price:
                    stop_loss_hit = close_price <= entry_price * (1 - base_stop_loss)
                    take_profit_hit = close_price >= entry_price * (1 + base_take_profit)
                    trailing_stop_hit = close_price <= highest_price * (1 - base_trailing_stop)

                    # Trend breakdown
                    trend_breakdown = not row['trend'] and hours_held > min_hold_hours

                    if (stop_loss_hit or take_profit_hit or trailing_stop_hit or
                        sell_signal or trend_breakdown) and hours_held >= min_hold_hours:
                        sell_signal = True

            # Execute trades
            if has_position:
                if sell_signal:
                    has_position = False
                    entry_price = None
                    entry_time = None
                    highest_price = None
                    signals.append("SELL")
                else:
                    signals.append("HOLD")
            else:
                if buy_signal:
                    has_position = True
                    entry_price = close_price
                    entry_time = timestamp
                    highest_price = close_price
                    signals.append("BUY")
                else:
                    signals.append("HOLD")

        df['signal'] = signals

        # Add diagnostic columns
        df['btc_causality_strength'] = btc_causality
        df['eth_causality_strength'] = eth_causality
        df['optimal_btc_lag'] = optimal_btc_lag
        df['optimal_eth_lag'] = optimal_eth_lag

        return df[['timestamp', 'signal']]

    except Exception as e:
        print(f"Strategy error: {e}")
        # Fallback to simple strategy
        df = pd.merge(
            candles_target[['timestamp', 'close', 'high', 'low']],
            candles_anchor[['timestamp', btc_col, eth_col]],
            on='timestamp',
            how='inner'
        )

        # Simple fallback signals
        signals = []
        for i in range(len(df)):
            if i < 3:
                signals.append("HOLD")
                continue

            btc_change = df[btc_col].pct_change(3).iloc[i]
            ldo_ma = df['close'].rolling(5, min_periods=1).mean().iloc[i]

            if not pd.isna(btc_change) and btc_change > 0.01 and df['close'].iloc[i] > ldo_ma:
                signals.append("BUY")
            elif not pd.isna(btc_change) and btc_change < -0.01:
                signals.append("SELL")
            else:
                signals.append("HOLD")

        df['signal'] = signals
        return df[['timestamp', 'signal']]

def get_coin_metadata() -> dict:
    """
    Provides metadata required by the evaluation engine.
    """
    return {
        "target": {
            "symbol": TARGET_COIN,
            "timeframe": TIMEFRAME
        },
        "anchors": [
            {"symbol": a["symbol"], "timeframe": a["timeframe"]} for a in ANCHORS
        ]
    }