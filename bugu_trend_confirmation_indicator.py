import pandas as pd
import ta
import numpy as np

def analyze_market_trend(data, lookback_period=20, atr_period=14):
    # Calculate indicators
    data['SMA20'] = ta.trend.sma_indicator(data['close'], window=20)
    data['EMA20'] = ta.trend.ema_indicator(data['close'], window=20)
    data['RSI'] = ta.momentum.rsi(data['close'], window=14)
    data['MACD'] = ta.trend.macd(data['close'])
    data['MACD_signal'] = ta.trend.macd_signal(data['close'])
    bb = ta.volatility.BollingerBands(data['close'], window=20, window_dev=2)
    data['bb_high'] = bb.bollinger_hband()
    data['bb_low'] = bb.bollinger_lband()
    data['ATR'] = ta.volatility.average_true_range(data['high'], data['low'], data['close'], window=atr_period)
    data['Volume_SMA'] = data['tick_volume'].rolling(window=atr_period).mean()

    # Support and Resistance
    data['Support'] = data['low'].rolling(window=lookback_period).min()
    data['Resistance'] = data['high'].rolling(window=lookback_period).max()

    # Price Action
    data['Higher_High'] = data['high'] > data['high'].shift(1)
    data['Higher_Low'] = data['low'] > data['low'].shift(1)
    data['Lower_High'] = data['high'] < data['high'].shift(1)
    data['Lower_Low'] = data['low'] < data['low'].shift(1)

    data['Uptrend'] = data['Higher_High'] & data['Higher_Low']
    data['Downtrend'] = data['Lower_High'] & data['Lower_Low']

    # Analyzing Trend Strength with Price Action
    if data['close'].iloc[-1] > data['SMA20'].iloc[-1] and data['close'].iloc[-1] > data['EMA20'].iloc[-1] and data['Uptrend'].iloc[-1]:
        trend = 'Uptrend'
    elif data['close'].iloc[-1] < data['SMA20'].iloc[-1] and data['close'].iloc[-1] < data['EMA20'].iloc[-1] and data['Downtrend'].iloc[-1]:
        trend = 'Downtrend'
    else:
        trend = 'Sideways'

    volume_direction = data['tick_volume'].iloc[-1] - data['tick_volume'].iloc[-2]
    if volume_direction > 0:
        volume_strength = 'Strong'
    else:
        volume_strength = 'Weak'

    support_level = data['Support'].iloc[-1]
    resistance_level = data['Resistance'].iloc[-1]

    return trend, volume_strength, support_level, resistance_level