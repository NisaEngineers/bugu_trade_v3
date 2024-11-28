import pandas as pd
import talib
import numpy as np
import matplotlib.pyplot as plt
import talib as ta
import statsmodels.api as sm
import ta as trade
from numba import njit


def dynamic_trading_with_key_level(df, plot=None):
    # Input Parameters
    lookback_period = 20
    atr_period = 14
    atr_multiplier_sl = 1.5
    atr_multiplier_tp1 = 1.5
    atr_multiplier_tp2 = 2.0
    reward_to_risk = 2.0

    # ATR Calculation
    df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=atr_period)

    # Volume SMA Calculation
    df['Volume_SMA'] = df['tick_volume'].rolling(window=atr_period).mean()

    # Key Levels Identification (Support & Resistance Zones)
    df['Support'] = df['low'].rolling(window=lookback_period).min()
    df['Resistance'] = df['high'].rolling(window=lookback_period).max()
    df['Support_Buffer'] = df['Support'] - df['ATR'] * 0.5
    df['Resistance_Buffer'] = df['Resistance'] + df['ATR'] * 0.5

    # Define Entry Points
    df['Bullish_Entry'] = (df['close'] > df['Support_Buffer']) & (df['low'] <= df['Support']) & (df['tick_volume'] > df['Volume_SMA'])
    df['Bearish_Entry'] = (df['close'] < df['Resistance_Buffer']) & (df['high'] >= df['Resistance']) & (df['tick_volume'] > df['Volume_SMA'])

    # Stop Loss and Take Profit Calculations for Bullish and Bearish Scenarios
    df['Bullish_SL'] = df['Support'] - df['ATR'] * atr_multiplier_sl
    df['Bullish_TP1'] = df['Support'] + df['ATR'] * reward_to_risk * atr_multiplier_tp1
    df['Bullish_TP2'] = df['Support'] + df['ATR'] * reward_to_risk * atr_multiplier_tp2

    df['Bearish_SL'] = df['Resistance'] + df['ATR'] * atr_multiplier_sl
    df['Bearish_TP1'] = df['Resistance'] - df['ATR'] * reward_to_risk * atr_multiplier_tp1
    df['Bearish_TP2'] = df['Resistance'] - df['ATR'] * reward_to_risk * atr_multiplier_tp2

    # Visualization
    if plot:
        plt.figure(figsize=(12, 8))
        plt.plot(df.index, df['close'], label='Close Price', color='black')
        plt.plot(df.index, df['Support'], label='Support', color='green')
        plt.plot(df.index, df['Resistance'], label='Resistance', color='red')

        # Mark Bullish Entry Points
        bullish_entries = df[df['Bullish_Entry']]
        plt.scatter(bullish_entries.index, bullish_entries['close'], marker='^', color='green', label='Bullish Entry')

        # Mark Bearish Entry Points
        bearish_entries = df[df['Bearish_Entry']]
        plt.scatter(bearish_entries.index, bearish_entries['close'], marker='v', color='red', label='Bearish Entry')

        # Add SL, TP Lines for visualization (only for the last bullish and bearish entry for simplicity)
        if not bullish_entries.empty:
            last_bullish_entry = bullish_entries.iloc[-1]
            plt.axhline(y=last_bullish_entry['Bullish_SL'], color='red', linestyle='--', label='Bullish SL')
            plt.axhline(y=last_bullish_entry['Bullish_TP1'], color='green', linestyle='--', label='Bullish TP1')
            plt.axhline(y=last_bullish_entry['Bullish_TP2'], color='blue', linestyle='--', label='Bullish TP2')

        if not bearish_entries.empty:
            last_bearish_entry = bearish_entries.iloc[-1]
            plt.axhline(y=last_bearish_entry['Bearish_SL'], color='red', linestyle='--', label='Bearish SL')
            plt.axhline(y=last_bearish_entry['Bearish_TP1'], color='green', linestyle='--', label='Bearish TP1')
            plt.axhline(y=last_bearish_entry['Bearish_TP2'], color='blue', linestyle='--', label='Bearish TP2')

        plt.title('Dynamic Trading Strategy with Key Levels')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

    return df



class MachineLearningSupertrend:
    def __init__(self, df, atr_length=10, factor=3, training_datrade_period=100, highvol=0.75, midvol=0.5, lowvol=0.25):
        self.df = df
        self.atr_length = atr_length
        self.factor = factor
        self.training_datrade_period = training_datrade_period
        self.highvol = highvol
        self.midvol = midvol
        self.lowvol = lowvol

    def pine_supertrend(self):
        hl2 = (self.df['high'] + self.df['low']) / 2
        atr = trade.volatility.AverageTrueRange(high=self.df['high'], low=self.df['low'], close=self.df['close'], window=self.atr_length).average_true_range()

        self.df['upper_band'] = hl2 + self.factor * atr
        self.df['lower_band'] = hl2 - self.factor * atr

        self.df['prev_lower_band'] = self.df['lower_band'].shift(1)
        self.df['prev_upper_band'] = self.df['upper_band'].shift(1)

        self.df['lower_band'] = np.where((self.df['lower_band'] > self.df['prev_lower_band']) | (self.df['close'].shift(1) < self.df['prev_lower_band']), self.df['lower_band'], self.df['prev_lower_band'])
        self.df['upper_band'] = np.where((self.df['upper_band'] < self.df['prev_upper_band']) | (self.df['close'].shift(1) > self.df['prev_upper_band']), self.df['upper_band'], self.df['prev_upper_band'])

        self.df['direction'] = np.nan
        self.df['supertrend'] = np.nan

        for i in range(1, len(self.df)):
            if pd.isna(atr[i-1]):
                self.df['direction'].iat[i] = 1
            elif self.df['supertrend'].iat[i-1] == self.df['prev_upper_band'].iat[i]:
                self.df['direction'].iat[i] = -1 if self.df['close'].iat[i] > self.df['upper_band'].iat[i] else 1
            else:
                self.df['direction'].iat[i] = 1 if self.df['close'].iat[i] < self.df['lower_band'].iat[i] else -1

            self.df['supertrend'].iat[i] = self.df['lower_band'].iat[i] if self.df['direction'].iat[i] == -1 else self.df['upper_band'].iat[i]

        return self.df['supertrend'], self.df['direction']

    def kmeans_volatility_clustering(self, volatility):
        upper = volatility.rolling(self.training_datrade_period).max()
        lower = volatility.rolling(self.training_datrade_period).min()

        high_volatility = lower + (upper - lower) * self.highvol
        medium_volatility = lower + (upper - lower) * self.midvol
        low_volatility = lower + (upper - lower) * self.lowvol

        amean = high_volatility
        bmean = medium_volatility
        cmean = low_volatility

        while True:
            hv, mv, lv = [], [], []

            for i in range(len(volatility) - self.training_datrade_period, len(volatility)):
                _1 = abs(volatility[i] - amean.iloc[i])
                _2 = abs(volatility[i] - bmean.iloc[i])
                _3 = abs(volatility[i] - cmean.iloc[i])
                if _1 < _2 and _1 < _3:
                    hv.append(volatility[i])
                elif _2 < _1 and _2 < _3:
                    mv.append(volatility[i])
                else:
                    lv.append(volatility[i])

            new_amean = np.mean(hv)
            new_bmean = np.mean(mv)
            new_cmean = np.mean(lv)

            if new_amean == amean.iloc[-1] and new_bmean == bmean.iloc[-1] and new_cmean == cmean.iloc[-1]:
                break

            amean.iloc[-1], bmean.iloc[-1], cmean.iloc[-1] = new_amean, new_bmean, new_cmean

        return amean.iloc[-1], bmean.iloc[-1], cmean.iloc[-1]

    def adaptive_supertrend(self):
        self.df['volatility'] = trade.volatility.AverageTrueRange(high=self.df['high'], low=self.df['low'], close=self.df['close'], window=self.atr_length).average_true_range()

        hv_new, mv_new, lv_new = self.kmeans_volatility_clustering(self.df['volatility'])

        assigned_centroid = np.where(abs(self.df['volatility'] - hv_new) < abs(self.df['volatility'] - mv_new),
                                     np.where(abs(self.df['volatility'] - hv_new) < abs(self.df['volatility'] - lv_new), hv_new, lv_new),
                                     np.where(abs(self.df['volatility'] - mv_new) < abs(self.df['volatility'] - lv_new), mv_new, lv_new))

        supertrend, direction = self.pine_supertrend()

        return self.df, supertrend, direction

    def plot_supertrend(self, supertrend):
        plt.figure(figsize=(14, 7))
        plt.plot(self.df['close'], label='Close Price')
        plt.plot(supertrend, label='SuperTrend', color='blue')

        bullish_trend = self.df['close'] > supertrend
        bearish_trend = self.df['close'] < supertrend

        plt.fill_between(self.df.index, supertrend, self.df['close'], where=bullish_trend, color='green', alpha=0.1)
        plt.fill_between(self.df.index, supertrend, self.df['close'], where=bearish_trend, color='red', alpha=0.1)

        plt.legend()
        plt.show()

    def machine_learning_supertrend(self, plot=None):
        df, supertrend, direction = self.adaptive_supertrend()

        # Combine additional indicators for more precise signals
        df['rsi'] = trade.momentum.RSIIndicator(df['close'], window=14).rsi()
        df['macd_diff'] = trade.trend.MACD(df['close']).macd_diff()

        # Generate buy and sell signals with added conditions to reduce false signals
        df['long_signal'] = (df['close'] > df['supertrend']) & (df['rsi'] > 30) & (df['rsi'] < 60) & (df['macd_diff'] > 0)
        df['short_signal'] = (df['close'] < df['supertrend']) & (df['rsi'] > 30) & (df['rsi'] < 60) & (df['macd_diff'] < 0)


        if plot:
            fig, ax = plt.subplots(figsize=(14, 7))

            ax.plot(df['close'], label='Close Price', color='black')
            ax.plot(df['supertrend'], label='SuperTrend', color='blue')

            buy_signals = df[df['long_signal']]
            sell_signals = df[df['short_signal']]

            ax.scatter(buy_signals.index, buy_signals['low'], marker='^', color='green', label='BUY', s=100)
            ax.scatter(sell_signals.index, sell_signals['high'], marker='v', color='red', label='SELL', s=100)

            plt.title("SuperTrend Buy and Sell Signals")
            plt.xlabel("Index")
            plt.ylabel("Price")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        return df


class FutureTrend:
    def __init__(self, df, length=100, multi=3, extend=50, period_atr=200, period_sma=100, color_up='#16d897', color_dn='#da853f'):
        self.df = df
        self.length = length
        self.multi = multi
        self.extend = extend
        self.period_atr = period_atr
        self.period_sma = period_sma
        self.color_up = color_up
        self.color_dn = color_dn

    def calculate_atr(self):
        return ta.ATR(self.df['high'], self.df['low'], self.df['close'], timeperiod=self.period_atr)

    def calculate_sma(self):
        return ta.SMA(self.df['close'], timeperiod=self.period_sma)

    def calculate_future_price(self, x1, x2, y1, y2, index):
        slope = (y2 - y1) / (x2 - x1)
        return y1 + slope * (index - x1)

    def update_lines(self, line_data, src, high_src, low_src, trend, current_index, extend_right, extend_none):
        if trend:
            line_data['mid']['x2'], line_data['mid']['y2'] = current_index, ta.SMA(src, 20)[current_index]
            line_data['top']['x2'], line_data['top']['y2'] = current_index, ta.SMA(high_src, 20)[current_index]
            line_data['low']['x2'], line_data['low']['y2'] = current_index, ta.SMA(low_src, 20)[current_index]
            line_data['mid']['extend'] = extend_right
            line_data['top']['extend'] = extend_right
            line_data['low']['extend'] = extend_right
        else:
            line_data['mid']['extend'] = extend_none
            line_data['top']['extend'] = extend_none
            line_data['low']['extend'] = extend_none
        return line_data

    def plot_lines(self, line_data):
        for key in ['mid', 'top', 'low']:
            plt.plot([line_data[key]['x1'], line_data[key]['x2']], [line_data[key]['y1'], line_data[key]['y2']], label=f"{key} line")
            plt.fill_between([line_data[key]['x1'], line_data[key]['x2']], line_data['low']['y1'], line_data['top']['y1'], color=self.color_up if key == 'mid' else self.color_dn, alpha=0.1)
        plt.legend()
        plt.show()

    def draw_channel(self, plot=True):
        atr = self.calculate_atr()
        sma = self.calculate_sma()
        src = (self.df['high'] + self.df['low']) / 2
        high_src = src + atr * self.multi
        low_src = src - atr * self.multi
        
        line_data = {
            'mid': {'x1': 0, 'x2': 0, 'y1': 0, 'y2': 0, 'extend': 'none'},
            'top': {'x1': 0, 'x2': 0, 'y1': 0, 'y2': 0, 'extend': 'none'},
            'low': {'x1': 0, 'x2': 0, 'y1': 0, 'y2': 0, 'extend': 'none'}
        }

        future_trend = None

        for i in range(len(self.df)):
            if i > 0:
                trend = ta.CDLENGULFING(self.df['open'], self.df['high'], self.df['low'], self.df['close'])[i] == 100
                line_data = self.update_lines(line_data, src, high_src, low_src, trend, i, 'right', 'none')
                if trend and not self.df['close'][i-1] > sma[i-1]:
                    line_data['mid']['x1'], line_data['mid']['y1'] = i, src[i]
                    line_data['top']['x1'], line_data['top']['y1'] = i, high_src[i]
                    line_data['low']['x1'], line_data['low']['y1'] = i, low_src[i]
                    future_trend = 'up'
                elif not trend and self.df['close'][i-1] < sma[i-1]:
                    line_data['mid']['x1'], line_data['mid']['y1'] = i, src[i]
                    line_data['top']['x1'], line_data['top']['y1'] = i, high_src[i]
                    line_data['low']['x1'], line_data['low']['y1'] = i, low_src[i]
                    future_trend = 'down'

        if plot:
            self.plot_lines(line_data)
        
        return future_trend






class HalfTrendRegression:
    def __init__(self, df, linreg_len=14, amplitude=5, channel_deviation=2.0):
        self.df = df
        self.linreg_len = linreg_len
        self.amplitude = amplitude
        self.channel_deviation = channel_deviation

    @staticmethod
    @njit
    def apply_linear_regression(x):
        n = len(x)
        X = np.arange(n)
        X_mean = np.mean(X)
        y_mean = np.mean(x)
        num = np.sum((X - X_mean) * (x - y_mean))
        den = np.sum((X - X_mean) ** 2)
        slope = num / den
        intercept = y_mean - slope * X_mean
        return intercept + slope * (n - 1)

    def linear_regression(self, series):
        result = np.empty_like(series)
        for i in range(len(series)):
            if i < self.linreg_len - 1:
                result[i] = np.nan
            else:
                result[i] = self.apply_linear_regression(series[i - self.linreg_len + 1:i + 1])
        return result

    def calculate_trend_line(self):
        high_values = self.df['high'].values
        low_values = self.df['low'].values
        close_values = self.df['close'].values
        
        highs = self.linear_regression(high_values)
        lows = self.linear_regression(low_values)
        closes = self.linear_regression(close_values)

        atr = trade.volatility.average_true_range(self.df['high'], self.df['low'], self.df['close'], window=14) / 2
        channel_offset = self.channel_deviation * atr

        high_ma = trade.trend.sma_indicator(pd.Series(highs), window=self.amplitude)
        low_ma = trade.trend.sma_indicator(pd.Series(lows), window=self.amplitude)
        highest_high = pd.Series(highs).rolling(window=self.amplitude).max()
        lowest_low = pd.Series(lows).rolling(window=self.amplitude).min()

        trend_line = np.zeros_like(closes)
        trend_direction = np.zeros_like(closes, dtype=int)
        max_low = low_values.copy()
        min_high = high_values.copy()

        # Initialize trend direction to 1 for the first element
        trend_direction[0] = 1

        for i in range(1, len(close_values)):
            if trend_direction[i-1] == 1:
                max_low[i] = max(max_low[i-1], lowest_low[i])
                if high_ma[i] < max_low[i] and closes[i] < lows[i-1]:
                    trend_direction[i] = -1
                    trend_line[i] = highest_high[i]
                    min_high[i] = highs[i]
                else:
                    trend_line[i] = max(trend_line[i-1], lowest_low[i])
            else:
                min_high[i] = min(min_high[i-1], highest_high[i])
                if low_ma[i] > min_high[i] and closes[i] > highs[i-1]:
                    trend_direction[i] = 1
                    trend_line[i] = lowest_low[i]
                    max_low[i] = lows[i]
                else:
                    trend_line[i] = min(trend_line[i-1], highest_high[i])

        upper_channel = trend_line + channel_offset
        lower_channel = trend_line - channel_offset

        return trend_line, trend_direction, upper_channel, lower_channel

    def calculate_signals(self, trend_line, trend_direction, upper_channel, lower_channel):
        stoch_rsi = trade.momentum.StochRSIIndicator(self.df['close'], window=14, smooth1=3, smooth2=3, fillna=False)
        self.df['rsi'] = stoch_rsi.stochrsi() * 100  # Convert Stoch RSI to a percentage

        self.df['Buy_Signal'] = np.where((trend_direction == 1) & (self.df['close'] > lower_channel) & (self.df['rsi'] < 30), self.df['close'], np.nan)
        self.df['Sell_Signal'] = np.where((trend_direction == -1) & (self.df['close'] < upper_channel) & (self.df['rsi'] > 70), self.df['close'], np.nan)
        return self.df

    def plot_trend(self, trend_line, upper_channel, lower_channel, trend_direction):
        plt.figure(figsize=(12, 6))
        plt.plot(self.df['date'], self.df['close'], label='Close Price', color='black')
        plt.plot(self.df['date'], trend_line, label='Trend Line', color='blue')
        
        plt.fill_between(self.df['date'], trend_line, lower_channel, where=(trend_direction == 1), color='green', alpha=0.1)
        plt.fill_between(self.df['date'], trend_line, upper_channel, where=(trend_direction == -1), color='red', alpha=0.1)
        
        plt.scatter(self.df['date'], self.df['Buy_Signal'], marker='^', color='green', label='Buy Signal', s=100)
        plt.scatter(self.df['date'], self.df['Sell_Signal'], marker='v', color='red', label='Sell Signal', s=100)
        
        plt.title('Half Trend Regression')
        plt.legend()
        plt.show()

    def half_trend_regression(self, plot=True):
        trend_line, trend_direction, upper_channel, lower_channel = self.calculate_trend_line()
        self.df = self.calculate_signals(trend_line, trend_direction, upper_channel, lower_channel)
        
        future_trend = 'up' if trend_direction[-1] == 1 else 'down'

        if plot:
            self.plot_trend(trend_line, upper_channel, lower_channel, trend_direction)
        
        return self.df, future_trend

# Example usage
# ht_regression = HalfTrendRegression(df)
# df, future_trend = ht_regression.half_trend_regression(plot=True)



class BuySellSignal:
    def __init__(self, df, period1=27, mult1=1.6, period2=55, mult2=2.0, length_ma=20, rsi_length=14, rsi_overbought=70, rsi_oversold=30):
        self.df = df
        self.period1 = period1
        self.mult1 = mult1
        self.period2 = period2
        self.mult2 = mult2
        self.length_ma = length_ma
        self.rsi_length = rsi_length
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold

    def smoothrng(self, source, period, multiplier):
        wper = period * 2 - 1
        avrng = ta.EMA(np.abs(source - source.shift(1)), period)
        return ta.EMA(avrng, wper) * multiplier

    def rngfilt(self, source, smrng):
        rngfilt = np.copy(source)
        for i in range(1, len(rngfilt)):
            rngfilt[i] = max(source[i] - smrng[i], rngfilt[i-1]) if source[i] > rngfilt[i-1] else min(source[i] + smrng[i], rngfilt[i-1])
        return rngfilt

    def generate_signals(self):
        self.df['smrng1'] = self.smoothrng(self.df['close'], self.period1, self.mult1)
        self.df['smrng2'] = self.smoothrng(self.df['close'], self.period2, self.mult2)
        self.df['smrng'] = (self.df['smrng1'] + self.df['smrng2']) / 2
        self.df['sma'] = ta.SMA(self.df['close'], self.length_ma)
        self.df['filt'] = self.rngfilt(self.df['close'], self.df['smrng'])

        self.df['upward'] = (self.df['filt'] > self.df['filt'].shift(1)).astype(int).cumsum()
        self.df['downward'] = (self.df['filt'] < self.df['filt'].shift(1)).astype(int).cumsum()

        self.df['longCond'] = ((self.df['close'] > self.df['filt']) & (self.df['close'] > self.df['close'].shift(1)) & (self.df['upward'] > 0)) | ((self.df['close'] > self.df['filt']) & (self.df['close'] < self.df['close'].shift(1)) & (self.df['upward'] > 0))
        self.df['shortCond'] = ((self.df['close'] < self.df['filt']) & (self.df['close'] < self.df['close'].shift(1)) & (self.df['downward'] > 0)) | ((self.df['close'] < self.df['filt']) & (self.df['close'] > self.df['close'].shift(1)) & (self.df['downward'] > 0))

        self.df['CondIni'] = np.where(self.df['longCond'], 1, np.where(self.df['shortCond'], -1, np.nan))
        self.df['CondIni'] = self.df['CondIni'].ffill().shift(1).fillna(0).astype(int)

        self.df['long'] = (self.df['longCond']) & (self.df['CondIni'] == -1)
        self.df['short'] = (self.df['shortCond']) & (self.df['CondIni'] == 1)

        self.df['bullishCandle'] = (self.df['close'] >= self.df['open'].shift(1)) & (self.df['close'].shift(1) < self.df['open'].shift(1))
        self.df['bearishCandle'] = (self.df['close'] <= self.df['open'].shift(1)) & (self.df['close'].shift(1) > self.df['open'].shift(1))

        self.df['rsi'] = ta.RSI(self.df['close'], timeperiod=self.rsi_length)
        self.df['isRSIOB'] = self.df['rsi'] >= self.rsi_overbought
        self.df['isRSIOS'] = self.df['rsi'] <= self.rsi_oversold

        self.df['tradeSignal'] = (((self.df['isRSIOS']) | (self.df['isRSIOS'].shift(1)) | (self.df['isRSIOS'].shift(2))) & (self.df['bullishCandle'])) | (((self.df['isRSIOB']) | (self.df['isRSIOB'].shift(1)) | (self.df['isRSIOB'].shift(2))) & (self.df['bearishCandle']))

        self.df['signal'] = np.where(self.df['long'], 'Buy', np.where(self.df['short'], 'Sell', 'Hold'))
        return self.df

    def plot_signals(self):
        plt.figure(figsize=(14, 7))
        plt.plot(self.df.index, self.df['close'], label='Close Price', color='black')
        plt.scatter(self.df.index[self.df['signal'] == 'Buy'], self.df['close'][self.df['signal'] == 'Buy'], marker='^', color='green', label='Buy')
        plt.scatter(self.df.index[self.df['signal'] == 'Sell'], self.df['close'][self.df['signal'] == 'Sell'], marker='v', color='red', label='Sell')
        plt.scatter(self.df.index[self.df['tradeSignal'] & self.df['bullishCandle']], self.df['close'][self.df['tradeSignal'] & self.df['bullishCandle']], marker='^', color='lime', label='Bullish Candle Buy')
        plt.scatter(self.df.index[self.df['tradeSignal'] & self.df['bearishCandle']], self.df['close'][self.df['tradeSignal'] & self.df['bearishCandle']], marker='v', color='orange', label='Bearish Candle Sell')
        plt.legend()
        plt.show()

    def analyze_trend(self, plot=True):
        self.generate_signals()
        if plot:
            self.plot_signals()
        trend = 'Up' if self.df['long'].iloc[-1] else 'Down' if self.df['short'].iloc[-1] else 'Hold'
        return trend




class HeikinAshiSignals:
    def __init__(self, df, consecCandles_L=3, mult_buy_on=False, consecCandles_S=3, mult_sell_on=False):
        self.df = df
        self.consecCandles_L = consecCandles_L
        self.mult_buy_on = mult_buy_on
        self.consecCandles_S = consecCandles_S
        self.mult_sell_on = mult_sell_on

    def calculate_heikin_ashi(self):
        self.df['hkClose'] = (self.df['open'] + self.df['high'] + self.df['low'] + self.df['close']) / 4
        self.df['hkOpen'] = (self.df['open'].shift(1) + self.df['close'].shift(1)) / 2
        self.df['hkOpen'].iloc[0] = (self.df['open'].iloc[0] + self.df['close'].iloc[0]) / 2
        
        for i in range(1, len(self.df)):
            self.df.at[self.df.index[i], 'hkOpen'] = (self.df['hkOpen'].iloc[i-1] + self.df['hkClose'].iloc[i-1]) / 2
        
        self.df['hkHigh'] = self.df[['high', 'hkOpen', 'hkClose']].max(axis=1)
        self.df['hkLow'] = self.df[['low', 'hkOpen', 'hkClose']].min(axis=1)

        self.df['isBullish'] = self.df['hkClose'] >= self.df['hkOpen']
        self.df['isBearish'] = self.df['hkClose'] < self.df['hkOpen']

    def generate_signals(self):
        self.calculate_heikin_ashi()

        self.df['bullishCount'] = self.df['isBullish'].astype(int).groupby((self.df['isBullish'] != self.df['isBullish'].shift()).cumsum()).cumsum()
        self.df['bearishCount'] = self.df['isBearish'].astype(int).groupby((self.df['isBearish'] != self.df['isBearish'].shift()).cumsum()).cumsum()

        self.df['buy_signal'] = np.where(self.mult_buy_on, self.df['bullishCount'] > self.consecCandles_L, 
                                    (self.df['bullishCount'] > self.consecCandles_L) & ~(self.df['bullishCount'].shift(1) > self.consecCandles_L))
        self.df['sell_signal'] = np.where(self.mult_sell_on, self.df['bearishCount'] > self.consecCandles_S, 
                                     (self.df['bearishCount'] > self.consecCandles_S) & ~(self.df['bearishCount'].shift(1) > self.consecCandles_S))

        self.df['signal'] = np.where(self.df['buy_signal'], 'Buy', np.where(self.df['sell_signal'], 'Sell', 'Hold'))
        return self.df  # Return self.df instead of df

    def plot_signals(self):
        plt.figure(figsize=(14, 7))
        plt.plot(self.df['date'], self.df['close'], label='Close Price', color='black')
        plt.scatter(self.df['date'][self.df['buy_signal']], self.df['close'][self.df['buy_signal']], marker='^', color='green', label='Buy Signal')
        plt.scatter(self.df['date'][self.df['sell_signal']], self.df['close'][self.df['sell_signal']], marker='v', color='red', label='Sell Signal')
        plt.title('Heikin Ashi Buy and Sell Signals')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

    def analyze_trend(self, plot=True):
        self.generate_signals()
        if plot:
            self.plot_signals()
        trend = 'Up' if self.df['buy_signal'].iloc[-1] else 'Down' if self.df['sell_signal'].iloc[-1] else 'Hold'
        return trend


class TargetTrend:
    def __init__(self, df, length=10, target=0, period_atr=200):
        self.df = df
        self.length = length
        self.target = target
        self.period_atr = period_atr

    def calculate_atr(self):
        return trade.SMA(trade.ATR(self.df['high'], self.df['low'], self.df['close'], timeperiod=self.period_atr), self.period_atr) * 0.8

    def calculate_sma(self):
        sma_high = trade.SMA(self.df['high'], timeperiod=self.length) + self.calculate_atr()
        sma_low = trade.SMA(self.df['low'], timeperiod=self.length) - self.calculate_atr()
        return sma_high, sma_low

    def calculate_signals(self, sma_high, sma_low):
        self.df['trend'] = np.nan
        self.df['trend_value'] = np.nan
        self.df['trend_color'] = np.nan

        for i in range(1, len(self.df)):
            if self.df['close'].iloc[i] > sma_high.iloc[i] and self.df['close'].iloc[i - 1] <= sma_high.iloc[i - 1]:
                self.df.at[self.df.index[i], 'trend'] = True
            elif self.df['close'].iloc[i] < sma_low.iloc[i] and self.df['close'].iloc[i - 1] >= sma_low.iloc[i - 1]:
                self.df.at[self.df.index[i], 'trend'] = False
            
            if self.df['trend'].iloc[i] == True:
                self.df.at[self.df.index[i], 'trend_value'] = sma_low.iloc[i]
                self.df.at[self.df.index[i], 'trend_color'] = '#06b690'
            elif self.df['trend'].iloc[i] == False:
                self.df.at[self.df.index[i], 'trend_value'] = sma_high.iloc[i]
                self.df.at[self.df.index[i], 'trend_color'] = '#b67006'

    def draw_targets(self):
        targets = []
        signal_up = self.df['trend'].shift(1) == False
        signal_down = self.df['trend'].shift(1) == True

        for i in range(len(self.df)):
            if signal_up.iloc[i] or signal_down.iloc[i]:
                base = self.df['trend_value'].iloc[i]
                atr_value = self.calculate_atr().iloc[i]
                direction = self.df['trend'].iloc[i]

                atr_multiplier = atr_value * (1 if direction else -1)
                target_len1 = atr_multiplier * (5 + self.target)
                target_len2 = atr_multiplier * (10 + self.target * 2)
                target_len3 = atr_multiplier * (15 + self.target * 3)

                targets.append({
                    'stop_loss': base,
                    'entry': self.df['close'].iloc[i],
                    'target1': self.df['close'].iloc[i] + target_len1,
                    'target2': self.df['close'].iloc[i] + target_len2,
                    'target3': self.df['close'].iloc[i] + target_len3
                })

        return targets

    def plot_signals(self):
        plt.figure(figsize=(14, 7))
        plt.plot(self.df.index, self.df['close'], label='Close Price', color='black')
        plt.plot(self.df.index, self.df['trend_value'], label='Trend Value', color='blue')
        plt.fill_between(self.df.index, self.df['trend_value'], self.df['close'], where=self.df['trend'] == True, color='green', alpha=0.1, label='Uptrend')
        plt.fill_between(self.df.index, self.df['trend_value'], self.df['close'], where=self.df['trend'] == False, color='red', alpha=0.1, label='Downtrend')
        plt.legend()
        plt.show()

    def analyze_trend(self, plot=True):
        sma_high, sma_low = self.calculate_sma()
        self.calculate_signals(sma_high, sma_low)
        targets = self.draw_targets()
        if plot:
            self.plot_signals()
        return self.df, targets


class SignalGenerator:
    def __init__(self, df, rsi_length=16, williams_length=16, atr_period=35, atr_multiplier=1.5, ema_short_len=9, ema_long_len=21,
                 volume_threshold_b1=1.1, volume_threshold_b2=1.5, volume_threshold_s1=1.1, volume_threshold_s2=1.5,
                 psar_increment=0.003, psar_max=0.2):
        self.df = df
        self.rsi_length = rsi_length
        self.williams_length = williams_length
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.ema_short_len = ema_short_len
        self.ema_long_len = ema_long_len
        self.volume_threshold_b1 = volume_threshold_b1
        self.volume_threshold_b2 = volume_threshold_b2
        self.volume_threshold_s1 = volume_threshold_s1
        self.volume_threshold_s2 = volume_threshold_s2
        self.psar_increment = psar_increment
        self.psar_max = psar_max

    def calculate_indicators(self):
        self.df['hlc3'] = (self.df['high'] + self.df['low'] + self.df['close']) / 3
        self.df['rsi'] = ta.RSI(self.df['hlc3'], timeperiod=self.rsi_length)
        self.df['smoothed_rsi'] = ta.EMA(self.df['rsi'], timeperiod=3)

        self.df['highest_high'] = self.df['high'].rolling(window=self.williams_length).max()
        self.df['lowest_low'] = self.df['low'].rolling(window=self.williams_length).min()
        self.df['williams_r'] = ((self.df['highest_high'] - self.df['close']) / (self.df['highest_high'] - self.df['lowest_low'])) * -100
        self.df['smoothed_williams_r'] = ta.EMA(self.df['williams_r'], timeperiod=3)

        self.df['volume_average'] = ta.SMA(self.df['tick_volume'], timeperiod=30)
        self.df['recent_volume_average'] = ta.SMA(self.df['tick_volume'], timeperiod=15)

        self.df['is_green_candle'] = self.df['close'] > self.df['open']
        self.df['is_red_candle'] = self.df['close'] < self.df['open']

        self.df['is_high_volume_b1'] = (self.df['tick_volume'] > (self.df['volume_average'] * self.volume_threshold_b1)) & (self.df['tick_volume'] > self.df['recent_volume_average'])
        self.df['is_high_volume_b2'] = (self.df['tick_volume'] > (self.df['volume_average'] * self.volume_threshold_b2)) & (self.df['tick_volume'] > self.df['recent_volume_average'])
        self.df['is_high_volume_s1'] = (self.df['tick_volume'] > (self.df['volume_average'] * self.volume_threshold_s1)) & (self.df['tick_volume'] > self.df['recent_volume_average'])
        self.df['is_high_volume_s2'] = (self.df['tick_volume'] > (self.df['volume_average'] * self.volume_threshold_s2)) & (self.df['tick_volume'] > self.df['recent_volume_average'])

        self.df['ema_short'] = ta.EMA(self.df['close'], timeperiod=self.ema_short_len)
        self.df['ema_long'] = ta.EMA(self.df['close'], timeperiod=self.ema_long_len)

        self.df['bullish_ema_cross'] = (self.df['ema_short'] > self.df['ema_long']) & (self.df['ema_short'].shift(1) <= self.df['ema_long'].shift(1))
        self.df['bearish_ema_cross'] = (self.df['ema_short'] < self.df['ema_long']) & (self.df['ema_short'].shift(1) >= self.df['ema_long'].shift(1))

        self.df['atr'] = self.atr_multiplier * ta.ATR(self.df['high'], self.df['low'], self.df['close'], timeperiod=self.atr_period)
        self.df['long_stop'] = self.df['hlc3'] - self.df['atr']
        self.df['short_stop'] = self.df['hlc3'] + self.df['atr']

        self.df['psar'] = ta.SAR(self.df['high'], self.df['low'], acceleration=self.psar_increment, maximum=self.psar_max)
        self.df['psar_dir'] = np.where(self.df['psar'] < self.df['close'], 1, -1)
        self.df['sell_signal'] = (self.df['psar_dir'] == -1) & (self.df['psar_dir'].shift(1) == 1)

    def generate_signals(self):
        self.calculate_indicators()
        
        # Calculate combined signals
        self.df['bull_weight_2'] = ((self.df['is_high_volume_b2'] * 0.35) + (self.df['is_green_candle'] * 0.2))
        self.df['bear_weight_2'] = ((self.df['is_high_volume_s2'] * 0.35) + (self.df['is_red_candle'] * 0.2))
        
        self.df['bull_signal_2'] = self.df['bull_weight_2'] >= 0.6
        self.df['bear_signal_2'] = self.df['bear_weight_2'] >= 0.6
        
        self.df['rsi_williams_r_bull'] = (self.df['smoothed_rsi'].shift(1) < 45) & (self.df['smoothed_williams_r'].shift(1) < -75)
        self.df['rsi_williams_r_bear'] = (self.df['smoothed_rsi'].shift(1) > 60) & (self.df['smoothed_williams_r'].shift(1) > -25)
        
        self.df['b1_buy'] = self.df['rsi_williams_r_bull'] & self.df['is_high_volume_b1'] & self.df['is_green_candle']
        self.df['b2_buy'] = self.df['bull_signal_2'] & self.df['is_green_candle']
        self.df['b2_strong_buy'] = self.df['b1_buy'] & self.df['bull_signal_2']
        self.df['s1_sell'] = self.df['rsi_williams_r_bear'] & self.df['is_high_volume_s1'] & self.df['is_red_candle']
        self.df['s2_sell'] = self.df['bear_signal_2'] & self.df['is_red_candle']
        self.df['s2_strong_sell'] = self.df['s1_sell'] & self.df['bear_signal_2']
        
        # Prepare final signal column
        self.df['signal'] = np.where(self.df['b1_buy'], 'b1_buy', np.where(
                        self.df['b2_buy'], 'b2_buy', np.where(
                        self.df['b2_strong_buy'], 'b2_strong_buy', np.where(
                        self.df['s1_sell'], 's1_sell', np.where(
                        self.df['s2_sell'], 's2_sell', np.where(
                        self.df['s2_strong_sell'], 's2_strong_sell', 'hold'))))))
        
        return self.df  # Return the DataFrame with signals

    def plot_signals(self):
        fig, ax = plt.subplots(figsize=(14, 7))

        # Plot the close price
        ax.plot(self.df['close'], label='Close Price', color='black')

        # Plot the buy and sell signals
        buy_signals = self.df[self.df['signal'].str.contains('buy')]
        sell_signals = self.df[self.df['signal'].str.contains('sell')]

        ax.scatter(buy_signals.index, buy_signals['close'], marker='^', color='green', label='BUY', s=100)
        ax.scatter(sell_signals.index, sell_signals['close'], marker='v', color='red', label='SELL', s=100)

        plt.title('Buy and Sell Signals')
        plt.xlabel('Index')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def analyze_trend(self, plot=True):
        self.generate_signals()
        if plot:
            self.plot_signals()
        trend = 'Up' if self.df['signal'].str.contains('buy').iloc[-1] else 'Down' if self.df['signal'].str.contains('sell').iloc[-1] else 'Hold'
        return trend

class SwingHighLowAnalyzer:
    def __init__(self, df, swingHighLength=8, swingLowLength=8, smaLength=50, orderBlockLength=20):
        self.df = df
        self.swingHighLength = swingHighLength
        self.swingLowLength = swingLowLength
        self.smaLength = smaLength
        self.orderBlockLength = orderBlockLength

    def find_swing_highs_lows(self, data, length):
        swing_highs = []
        swing_lows = []
        
        for i in range(len(data)):
            if i >= length:
                high_window = data['high'][i-length:i]
                low_window = data['low'][i-length:i]
                
                if data['high'][i] == high_window.max():
                    swing_highs.append(data['high'][i])
                else:
                    swing_highs.append(np.nan)
                    
                if data['low'][i] == low_window.min():
                    swing_lows.append(data['low'][i])
                else:
                    swing_lows.append(np.nan)
            else:
                swing_highs.append(np.nan)
                swing_lows.append(np.nan)
                
        return pd.Series(swing_highs), pd.Series(swing_lows)

    def generate_signals(self):
        self.df['swingHigh'], self.df['swingLow'] = self.find_swing_highs_lows(self.df, self.swingHighLength)

        # Calculate Equilibrium, Premium, and Discount Zones
        self.df['equilibrium'] = (self.df['swingHigh'] + self.df['swingLow']) / 2
        self.df['premiumZone'] = self.df['swingHigh']
        self.df['discountZone'] = self.df['swingLow']

        # Calculate SMA for trend direction
        self.df['sma'] = talib.SMA(self.df['close'], timeperiod=self.smaLength)

        # Define buy and sell signals based on zones and trend direction
        self.df['buySignal'] = (self.df['close'] < self.df['equilibrium']) & (self.df['close'] > self.df['discountZone']) & (self.df['close'] > self.df['sma'])
        self.df['sellSignal'] = (self.df['close'] > self.df['equilibrium']) & (self.df['close'] < self.df['premiumZone']) & (self.df['close'] < self.df['sma'])

        # Print statements for debugging
        print("Equilibrium:", self.df['equilibrium'].tail())
        print("Premium Zone:", self.df['premiumZone'].tail())
        print("Discount Zone:", self.df['discountZone'].tail())
        print("SMA:", self.df['sma'].tail())
        print("Buy Signal:", self.df['buySignal'].tail())
        print("Sell Signal:", self.df['sellSignal'].tail())

        # Order Blocks
        self.df['orderBlockHigh'] = self.df['high'].rolling(window=self.orderBlockLength).max()
        self.df['orderBlockLow'] = self.df['low'].rolling(window=self.orderBlockLength).min()

        # Buy and sell signals with order block confirmation
        self.df['buySignalOB'] = self.df['buySignal'] & (self.df['close'] >= self.df['orderBlockLow'])
        self.df['sellSignalOB'] = self.df['sellSignal'] & (self.df['close'] <= self.df['orderBlockHigh'])
        
        # Print order block signals for debugging
        print("Buy Signal OB:", self.df['buySignalOB'].tail())
        print("Sell Signal OB:", self.df['sellSignalOB'].tail())

    def plot_signals(self):
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot close prices
        ax.plot(self.df['close'], label='Close Price', color='black')

        # Plot Equilibrium, Premium, and Discount Zones
        ax.plot(self.df['equilibrium'], label='Equilibrium', color='blue', linewidth=2)
        ax.plot(self.df['premiumZone'], label='Premium Zone (Resistance)', color='red', linewidth=1)
        ax.plot(self.df['discountZone'], label='Discount Zone (Support)', color='green', linewidth=1)

        # Plot SMA
        ax.plot(self.df['sma'], label='SMA', color='orange')

        # Plot Buy and Sell signals
        buy_signals = self.df[self.df['buySignalOB']]
        sell_signals = self.df[self.df['sellSignalOB']]

        ax.scatter(buy_signals.index, buy_signals['low'], marker='^', color='green', label='BUY', s=100)
        ax.scatter(sell_signals.index, sell_signals['high'], marker='v', color='red', label='SELL', s=100)

        # Mark liquidity zones
        liquidityZoneHigh = self.df['high'][self.df['high'] == self.df['swingHigh']]
        liquidityZoneLow = self.df['low'][self.df['low'] == self.df['swingLow']]
        ax.scatter(liquidityZoneHigh.index, liquidityZoneHigh, marker='^', color='red', label='Liquidity Zone High', s=100)
        ax.scatter(liquidityZoneLow.index, liquidityZoneLow, marker='v', color='green', label='Liquidity Zone Low', s=100)

        plt.title("SMC Strategy with Premium, Equilibrium, and Discount Zones")
        plt.xlabel("Index")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

class FRAMAChannel:
    def __init__(self, filepath, N=26, distance=1.5):
        self.filepath = filepath
        self.N = N
        self.distance = distance
        self.data = pd.read_csv(filepath)
        self.data['date'] = pd.to_datetime(self.data['time'])
        self.data['hl2'] = (self.data['high'] + self.data['low']) / 2
        self.data['volatility'] = trade.volatility.average_true_range(
            self.data['high'], self.data['low'], self.data['close'], window=200)
    
    def calculate_frama(self):
        # Highest and lowest over N periods
        HH = self.data['high'].rolling(self.N).max()
        LL = self.data['low'].rolling(self.N).min()

        # Highest and lowest over N/2 periods
        HH1 = self.data['high'].rolling(self.N//2).max()
        LL1 = self.data['low'].rolling(self.N//2).min()
        HH2 = self.data['high'].shift(self.N//2).rolling(self.N//2).max()
        LL2 = self.data['low'].shift(self.N//2).rolling(self.N//2).min()

        # Compute the dimensions and alpha
        N1 = (HH1 - LL1) / (self.N // 2)
        N2 = (HH2 - LL2) / (self.N // 2)
        N3 = (HH - LL) / self.N

        Dimen = (np.log(N1 + N2) - np.log(N3)).replace(-np.inf, 0) / np.log(2)
        alpha = np.exp(-4.6 * (Dimen - 1))
        alpha = np.clip(alpha, 0.01, 1)
        alpha = alpha.fillna(0.01)
    
        # Compute FRAMA
        self.data['Filt'] = trade.trend.ema_indicator(self.data['hl2'], window=self.N, fillna=True)
        for i in range(1, len(self.data)):
            if not pd.isna(alpha[i]):
                self.data.loc[self.data.index[i], 'Filt'] = alpha[i] * self.data['hl2'].iloc[i] + (1 - alpha[i]) * self.data['Filt'].iloc[i-1]

        self.data['Filt'] = self.data['Filt'].rolling(5).mean()
    
    def calculate_channel_bands(self):
        self.calculate_frama()
        self.data['Filt1'] = self.data['Filt'] + self.data['volatility'] * self.distance
        self.data['Filt2'] = self.data['Filt'] - self.data['volatility'] * self.distance

    def generate_signals(self):
        self.data['buy_signal'] = (self.data['close'] > self.data['Filt2']) & (self.data['close'].shift(1) <= self.data['Filt2'].shift(1))
        self.data['sell_signal'] = (self.data['close'] < self.data['Filt1']) & (self.data['close'].shift(1) >= self.data['Filt1'].shift(1))

    def plot(self):
        plt.figure(figsize=(14, 7))
        plt.plot(self.data['date'], self.data['close'], label='Close Price', color='black')
        plt.plot(self.data['date'], self.data['Filt'], label='FRAMA', color='blue')
        plt.plot(self.data['date'], self.data['Filt1'], label='FRAMA Upper Band', color='green')
        plt.plot(self.data['date'], self.data['Filt2'], label='FRAMA Lower Band', color='red')

        plt.scatter(self.data['date'][self.data['buy_signal']], self.data['close'][self.data['buy_signal']], marker='^', color='green', label='Buy Signal', s=100)
        plt.scatter(self.data['date'][self.data['sell_signal']], self.data['close'][self.data['sell_signal']], marker='v', color='red', label='Sell Signal', s=100)

        plt.fill_between(self.data['date'], self.data['Filt1'], self.data['Filt'], where=(self.data['Filt1'] >= self.data['Filt']), facecolor='green', alpha=0.5)
        plt.fill_between(self.data['date'], self.data['Filt2'], self.data['Filt'], where=(self.data['Filt2'] <= self.data['Filt']), facecolor='red', alpha=0.5)

        plt.title('FRAMA Channel with Buy and Sell Signals')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

    def analyze(self, plot=True):
        self.calculate_channel_bands()
        self.generate_signals()
        if plot:
            self.plot()
        return self.data, self.data.tail(5)


class AbnormalCandleDetector:
    def __init__(self, filepath, length=20, size_multiplier=2.0, volume_multiplier=2.0):
        self.filepath = filepath
        self.length = length
        self.size_multiplier = size_multiplier
        self.volume_multiplier = volume_multiplier
        self.data = pd.read_csv(filepath)
        self.data['date'] = pd.to_datetime(self.data['time'])
    
    def calculate_indicators(self):
        # Calculate candle size and average
        self.data['candle_size'] = abs(self.data['high'] - self.data['low'])
        self.data['avg_candle_size'] = self.data['candle_size'].rolling(window=self.length).mean()
        
        # Calculate volume average
        self.data['volume_avg'] = self.data['tick_volume'].rolling(window=self.length).mean()
        
        # Abnormal candle conditions
        self.data['is_abnormal_size'] = self.data['candle_size'] > self.data['avg_candle_size'] * self.size_multiplier
        self.data['is_abnormal_volume'] = self.data['tick_volume'] > self.data['volume_avg'] * self.volume_multiplier
        self.data['is_abnormal_candle'] = self.data['is_abnormal_size'] & self.data['is_abnormal_volume']

    def plot_abnormal_candles(self):
        plt.figure(figsize=(14, 7))
        plt.plot(self.data['date'], self.data['close'], label='Close Price', color='black')
        plt.fill_between(self.data['date'], self.data['low'], self.data['high'], where=self.data['is_abnormal_candle'], facecolor='purple', alpha=0.5, label='Abnormal Candle')
        
        for idx, row in self.data[self.data['is_abnormal_candle']].iterrows():
            plt.plot(row['date'], row['high'], 'v', color='orange', label='Abnormal Candle Marker' if idx == 0 else "", markersize=10)
        
        plt.title('Abnormal Candle Detector')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

    def analyze(self, plot=True):
        self.calculate_indicators()
        if plot:
            self.plot_abnormal_candles()
        return self.data

class OrderBlocksDetector:
    def __init__(self, filepath, timeframe='D', count=3, block_type='Both', reaction_factor=3,
                 keep_history=True, remove_broken=True, reaction_show=True, take_out=True,
                 consecutive_rising_or_falling=True, fair_value_gap=True, 
                 color_zones='fuchsia', color_replaced_zones='aqua', color_broken_zones='red'):
        self.filepath = filepath
        self.timeframe = timeframe
        self.count = count
        self.block_type = block_type
        self.reaction_factor = reaction_factor
        self.keep_history = keep_history
        self.remove_broken = remove_broken
        self.reaction_show = reaction_show
        self.take_out = take_out
        self.consecutive_rising_or_falling = consecutive_rising_or_falling
        self.fair_value_gap = fair_value_gap
        self.color_zones = color_zones
        self.color_replaced_zones = color_replaced_zones
        self.color_broken_zones = color_broken_zones
        
        self.data = pd.read_csv(filepath)
        self.data['date'] = pd.to_datetime(self.data['time'])

    def calculate_atr(self, n=14):
        self.data['ATR'] = trade.volatility.AverageTrueRange(
            self.data['high'], self.data['low'], self.data['close'], window=n).average_true_range()

    def detect_order_blocks(self):
        self.data['order_block'] = np.nan
        self.data['order_block_type'] = np.nan  # 1 for bullish, -1 for bearish

        for i in range(len(self.data)):
            if i < self.count:  # Skip initial rows
                continue

            if self.block_type in ['Both', 'Bullish']:
                if (self.data['close'].iloc[i] > self.data['open'].iloc[i] and 
                    self.data['close'].iloc[i] > self.data['high'].iloc[i-self.count:i].max() and 
                    (self.data['close'].iloc[i] - self.data['low'].iloc[i]) > self.reaction_factor * self.data['ATR'].iloc[i]):
                    self.data.at[self.data.index[i], 'order_block'] = self.data['low'].iloc[i]
                    self.data.at[self.data.index[i], 'order_block_type'] = 1

            if self.block_type in ['Both', 'Bearish']:
                if (self.data['close'].iloc[i] < self.data['open'].iloc[i] and 
                    self.data['close'].iloc[i] < self.data['low'].iloc[i-self.count:i].min() and 
                    (self.data['high'].iloc[i] - self.data['close'].iloc[i]) > self.reaction_factor * self.data['ATR'].iloc[i]):
                    self.data.at[self.data.index[i], 'order_block'] = self.data['high'].iloc[i]
                    self.data.at[self.data.index[i], 'order_block_type'] = -1

    def generate_signals(self):
        self.data['buy_signal'] = np.where((self.data['order_block_type'] == 1) & (self.data['close'] > self.data['order_block']), self.data['close'], np.nan)
        self.data['sell_signal'] = np.where((self.data['order_block_type'] == -1) & (self.data['close'] < self.data['order_block']), self.data['close'], np.nan)

    def plot_order_blocks(self):
        plt.figure(figsize=(14, 7))
        plt.plot(self.data['date'], self.data['close'], label='Close Price', color='black')

        # Plot Order Blocks
        for idx, row in self.data.dropna(subset=['order_block']).iterrows():
            plt.axhline(y=row['order_block'], color=self.color_zones, linestyle='--', label='Order Block' if idx == 0 else "")
            if row['order_block_type'] == 1:
                plt.plot(row['date'], row['order_block'], 'go', markersize=8, label='Bullish Order Block' if idx == 0 else "")
            elif row['order_block_type'] == -1:
                plt.plot(row['date'], row['order_block'], 'ro', markersize=8, label='Bearish Order Block' if idx == 0 else "")

        # Plot Buy and Sell Signals
        plt.plot(self.data['date'], self.data['buy_signal'], '^', color='green', markersize=10, label='Buy Signal')
        plt.plot(self.data['date'], self.data['sell_signal'], 'v', color='red', markersize=10, label='Sell Signal')

        plt.title('Order Blocks Detector with Buy/Sell Signals')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

    def analyze(self, plot=True):
        self.calculate_atr()
        self.detect_order_blocks()
        self.generate_signals()
        if plot:
            self.plot_order_blocks()
        return self.data

class PsychoSignal:
    def __init__(self, filepath, timeframeHigher='60', lookbackBars=5, minImpulse=1.0, 
                 showOrderBlocks=True, useWickToBody=True):
        self.filepath = filepath
        self.timeframeHigher = timeframeHigher
        self.lookbackBars = lookbackBars
        self.minImpulse = minImpulse
        self.showOrderBlocks = showOrderBlocks
        self.useWickToBody = useWickToBody
        
        self.data = pd.read_csv(filepath)
        self.data['date'] = pd.to_datetime(self.data['time'])
    
    def calculate_levels(self):
        # Calculate Higher Timeframe Levels
        self.data['higherHigh'] = self.data['high'].rolling(window=self.lookbackBars).max()
        self.data['higherLow'] = self.data['low'].rolling(window=self.lookbackBars).min()

        # Key Levels
        self.data['highestLevel'] = self.data['high'].rolling(window=self.lookbackBars).max()
        self.data['lowestLevel'] = self.data['low'].rolling(window=self.lookbackBars).min()

        # Detect Impulsive Move
        self.data['impulseMoveUp'] = (self.data['high'] - self.data['lowestLevel']) / self.data['lowestLevel'] * 100 >= self.minImpulse
        self.data['impulseMoveDown'] = (self.data['highestLevel'] - self.data['low']) / self.data['highestLevel'] * 100 >= self.minImpulse

    def calculate_order_blocks(self):
        # Order Block Calculation
        self.data['orderBlockUp'] = np.where(self.data['impulseMoveDown'], self.data['high'], np.nan)
        self.data['orderBlockDown'] = np.where(self.data['impulseMoveUp'], self.data['low'], np.nan)

        self.data['orderBlockRangeUp'] = np.where(self.data['impulseMoveDown'], self.data['high'] if self.useWickToBody else self.data['close'], np.nan)
        self.data['orderBlockRangeDown'] = np.where(self.data['impulseMoveUp'], self.data['low'] if self.useWickToBody else self.data['close'], np.nan)

    def crossover(self, series1, series2):
        return (series1 > series2) & (series1.shift(1) <= series2.shift(1))

    def crossunder(self, series1, series2):
        return (series1 < series2) & (series1.shift(1) >= series2.shift(1))

    def generate_signals(self):
        # Define Buy and Sell Signals
        self.data['buySignal'] = np.where(self.crossover(self.data['close'], self.data['orderBlockDown']), self.data['close'], np.nan)
        self.data['sellSignal'] = np.where(self.crossunder(self.data['close'], self.data['orderBlockUp']), self.data['close'], np.nan)

    def plot_signals(self):
        fig, ax = plt.subplots(figsize=(14, 7))

        if self.showOrderBlocks:
            ax.plot(self.data['date'], self.data['orderBlockRangeUp'], label='Order Block Up', color='red')
            ax.plot(self.data['date'], self.data['orderBlockRangeDown'], label='Order Block Down', color='green')

        # Plot Higher Timeframe Levels
        ax.plot(self.data['date'], self.data['higherHigh'], label='Higher High', color='blue', linestyle='dashed')
        ax.plot(self.data['date'], self.data['higherLow'], label='Higher Low', color='orange', linestyle='dashed')

        # Plot Buy and Sell Signals
        ax.plot(self.data['date'], self.data['buySignal'], '^', color='green', markersize=10, label='Buy Signal')
        ax.plot(self.data['date'], self.data['sellSignal'], 'v', color='red', markersize=10, label='Sell Signal')

        ax.set_title('Order Block Finder with Signals')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        plt.xticks(rotation=45)
        plt.legend()
        plt.show()

    def analyze(self, plot=True):
        self.calculate_levels()
        self.calculate_order_blocks()
        self.generate_signals()
        if plot:
            self.plot_signals()
        return self.data



class PrevGood:
    def __init__(self, df):
        self.df = df

    def calculate_indicators(self):
        self.df['SMA_50'] = trade.trend.sma_indicator(self.df['close'], window=50)
        stoch = trade.momentum.StochasticOscillator(high=self.df['high'], low=self.df['low'], close=self.df['close'], window=2, smooth_window=3)
        self.df['STOCH_%K'] = stoch.stoch()
        macd = trade.trend.MACD(close=self.df['close'], window_slow=20, window_fast=5, window_sign=9)
        self.df['MACD'] = macd.macd_diff()
        
        # Add RSI for further confirmation
        self.df['RSI'] = trade.momentum.RSIIndicator(close=self.df['close'], window=14).rsi()

        stoch_mt = trade.momentum.StochasticOscillator(high=self.df['high'], low=self.df['low'], close=self.df['close'], window=6, smooth_window=3)
        self.df['SMA_50_MT'] = self.df['SMA_50']  # Assuming medium time frame uses same SMA 50
        self.df['STOCH_%K_MT'] = stoch_mt.stoch()
    
    def generate_signals(self):
        self.calculate_indicators()

        # Calculate Buy and Sell signals with additional filters
        self.df['Buy_Signal'] = np.where(
            (self.df['SMA_50'].diff() > 0.05) &
            (self.df['STOCH_%K'] < 77) &
            (self.df['MACD'].diff() > 0) &
            (self.df['STOCH_%K_MT'].diff() > 0) &
            (self.df['RSI'] < 70),  # Additional filter to avoid overbought conditions
            self.df['close'], np.nan
        )

        self.df['Sell_Signal'] = np.where(
            (self.df['SMA_50'].diff() < -0.05) &
            (self.df['STOCH_%K'] > 23) &
            (self.df['MACD'].diff() < 0) &
            (self.df['STOCH_%K_MT'].diff() < 0) &
            (self.df['RSI'] > 30),  # Additional filter to avoid oversold conditions
            self.df['close'], np.nan
        )

        return self.df

    def plot_signals(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.df['date'], self.df['close'], label='Close Price', color='black')
        plt.plot(self.df['date'], self.df['SMA_50'], label='SMA 50', color='blue')
        plt.scatter(self.df['date'], self.df['Buy_Signal'], marker='^', color='green', label='Buy Signal', s=100)
        plt.scatter(self.df['date'], self.df['Sell_Signal'], marker='v', color='red', label='Sell Signal', s=100)
        
        plt.title('Prev Good Indicator Signals')
        plt.legend()
        plt.show()

    def prev_good_indicator(self, plot=True):
        self.generate_signals()

        if plot:
            self.plot_signals()
        
        return self.df




class PivotPoints:
    
    @staticmethod
    def sma(series, window):
        mavg = series.rolling(window=window, min_periods=window).mean()
        return mavg

    @staticmethod
    def ewma(series, span):
        ema = series.ewm(span=span).mean()
        return ema

    @staticmethod
    def pivot_points(df):
        n = len(df) - 1  # pivotPoints would be based on the last candle
        pp = (df['high'] + df['low'] + df['close']) / 3
        r1 = (2 * pp) - df['low']
        s1 = (2 * pp) - df['high']
        r2 = pp + (df['high'] - df['low'])
        s2 = pp - (df['high'] - df['low'])
        r3 = df['high'] + 2 * (pp - df['low'])
        s3 = df['low'] - 2 * (df['high'] - pp)

        pivots = pd.DataFrame({'PP': pp, 'R1': r1, 'R2': r2, 'R3': r3, 'S1': s1, 'S2': s2, 'S3': s3})
        return pivots

    @staticmethod
    def roc(df):
        closes = df['close'].apply(float)
        df['ROC'] = closes.diff() * 100
        df['ROC'].fillna(0, inplace=True)
        return df

    @staticmethod
    def stoch(df, period_K, period_D, graph=False):
        df['low'] = pd.to_numeric(df['low'], errors='coerce')
        df['Lstoch'] = df['low'].rolling(window=period_K).min()
        df['high'] = pd.to_numeric(df['high'], errors='coerce')
        df['Hstoch'] = df['high'].rolling(window=period_K).max()
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df['%K'] = 100 * ((df['close'] - df['Lstoch']) / (df['Hstoch'] - df['Lstoch']))
        df['%D'] = df['%K'].rolling(window=period_D).mean()

        if graph:
            fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(20, 10))
            df['close'].plot(ax=axes[0])
            df[['%K', '%D']].plot(ax=axes[1])
            plt.show()

        return df

    @staticmethod
    def boll_bands(df, window, n_std, graph=False):
        close = df['close']
        rolling_mean = close.rolling(window).mean()
        rolling_std = close.rolling(window).std()

        df['rolling_mean'] = rolling_mean
        df['boll_high'] = rolling_mean + (rolling_std * n_std)
        df['boll_low'] = rolling_mean - (rolling_std * n_std)

        if graph:
            plt.plot(close)
            plt.plot(df['rolling_mean'])
            plt.plot(df['boll_high'])
            plt.plot(df['boll_low'])
            plt.show()

        return df

    def generate_signals(self, df):
        pivots = self.pivot_points(df)
        df = pd.concat([df, pivots], axis=1)
        df['signal'] = 0
        
        df['signal'] = np.where((df['close'] > df['PP']) & (df['close'].shift(1) <= df['PP']), 1, df['signal'])
        df['signal'] = np.where((df['close'] < df['PP']) & (df['close'].shift(1) >= df['PP']), -1, df['signal'])
        df['signal'] = df['signal'].replace(0, np.nan).ffill().fillna(0)
        
        return df

    def plot_signals(self, df):
        fig, ax = plt.subplots(figsize=(12, 6))
        
        df['close'].plot(ax=ax, color='black', label='Close Price')
        df['PP'].plot(ax=ax, color='blue', linestyle='--', label='Pivot Point')
        df['R1'].plot(ax=ax, color='green', linestyle='--', label='Resistance 1')
        df['S1'].plot(ax=ax, color='red', linestyle='--', label='Support 1')
        
        buy_signals = df[df['signal'] == 1]
        sell_signals = df[df['signal'] == -1]
        ax.plot(buy_signals.index, buy_signals['close'], '^', markersize=10, color='g', label='Buy Signal')
        ax.plot(sell_signals.index, sell_signals['close'], 'v', markersize=10, color='r', label='Sell Signal')

        plt.legend()
        plt.show()

# Example usage
# Load data
#data = pd.read_csv('data.csv')

# Initialize the strategy class
#pivot_points_strategy = PivotPoints()

# Generate signals
#data_with_signals = pivot_points_strategy.generate_signals(data)

# Plot signals
#pivot_points_strategy.plot_signals(data_with_signals)








# Example usage
#df = pd.read_csv("data/BTCUSDm_D1_data.csv")
#df['date'] = pd.to_datetime(df['time'])  # Ensure 'date' column is correctly set
#prev_good = PrevGood(df)
#df_with_signals = prev_good.prev_good_indicator(plot=True)
