import numpy as np
import pandas as pd
from sklearn import ensemble, tree
from sklearn.model_selection import train_test_split
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

class MachineLearningStrategy:
    
    @staticmethod
    def sma(series, window):
        mavg = series.rolling(window=window, min_periods=window).mean()
        return mavg

    @staticmethod
    def ewma(series, span):
        ema = pd.Series.ewm(series, span=span).mean()
        return ema

    @staticmethod
    def pivot_points(df):
        n = len(df) - 1  # pivotPoints would be based on the last candle
        pp = (df['high'] + df['low'] + df['close']) / 3
        r1 = (2 * pp) - df['low']
        s1 = (2 * pp) - df['high']
        r2 = (pp - s1) + r1
        s2 = pp - (r1 - s1)
        r3 = (pp - s2) + r2
        s3 = pp - (r2 - s2)

        pivots = {'PP': pp, 'R1': r1, 'R2': r2, 'R3': r3,
                         'S1': s1, 'S2': s2, 'S3': s3}

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

    @staticmethod
    def ada_boost(df):
        n = len(df)
        X = np.asarray(df[['open', 'high', 'low', 'tick_volume']][:n-1])  # adjusted input data columns to match the data
        X = X.reshape(n-1, 4)  # adjust '4' to match the number of input columns used above
        y = np.asarray(df[['close']][1:])

        split = 0.8
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=split)
        dtree = tree.DecisionTreeRegressor(max_depth=10)
        model = ensemble.AdaBoostRegressor(n_estimators=50, learning_rate=2.0, base_estimator=dtree)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        print("AdaBoost model accuracy: ", score)
        
        prediction = model.predict(df[['open', 'high', 'low', 'tick_volume']][n-1:])
        print("AdaBoost predicted close for next candle: ", prediction)
        return prediction

    @staticmethod
    def random_forest(df):
        n = len(df)
        X = np.asarray(df[['open', 'high', 'low', 'tick_volume']][:n-1])  # adjusted input data columns to match the data
        X = X.reshape(n-1, 4)  # adjust '4' to match the number of input columns used above
        y = np.asarray(df[['close']][1:])

        split = 0.8
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=split)
        model = ensemble.RandomForestRegressor(n_estimators=10, max_depth=10)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        print("RandomForest model accuracy: ", score)

        prediction = model.predict(df[['open', 'high', 'low', 'tick_volume']][n-1:])
        print("RandomForest predicted close for next candle: ", prediction)
        return prediction

import pandas as pd
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

class AutoARIMAStrategy:
    
    def __init__(self, file, series):
        self.file = file
        self.series = series
        self.df = self.df = pd.read_csv(file)
        
    def auto_arima_research(self):
        # Prepare data
        X = self.df[self.series].values
        size = int(len(X) * 0.6)
        train, test = X[:size], X[size:]

        # Fit auto ARIMA model
        model = auto_arima(train, seasonal=False, trace=True,
                           error_action='ignore', suppress_warnings=True,
                           stepwise=True)
        
        print(f"Optimal ARIMA parameters: {model.order}")

        # Make predictions
        history = [x for x in train]
        predictions = list()
        for t in range(len(test)):
            model_fit = model.fit(history)
            output = model_fit.predict(n_periods=1)
            yhat = output[0]
            predictions.append(yhat)
            obs = test[t]
            history.append(obs)
            err = abs(yhat - obs)
            print("Predicted: %.3f, Expected: %.3f, Error: %.3f" % (yhat, obs, err))

        # Print summary and graph results
        error = mean_squared_error(test, predictions)
        print("Test MSE: %.3f" % (error))
        plt.plot(test, label='Actual Prices')
        plt.plot(predictions, label='Predicted Prices')
        plt.legend()
        plt.show()


class ARIMAStrategy:
    
    def __init__(self, file, series, p, d, q):
        self.file = file
        self.series = series
        self.p = p
        self.d = d
        self.q = q
        self.df = pd.read_csv(file)
        
    def arima_research(self):
        # Prepare data
        X = self.df[self.series].values
        size = int(len(X) * 0.6)
        train, test = X[:size], X[size:]
        history = [x for x in train]
        predictions = list()

        # Fit the model and make predictions
        for t in range(len(test)):
            model = ARIMA(history, order=(self.p, self.d, self.q))
            model_fit = model.fit()
            output = model_fit.forecast()
            yhat = output[0]
            predictions.append(yhat)
            obs = test[t]
            history.append(obs)
            err = abs(yhat - obs)
            print("Predicted: %.3f, Expected: %.3f, Error: %.3f" % (yhat, obs, err))

        # Print summary and graph results
        error = mean_squared_error(test, predictions)
        print("Test MSE: %.3f" % (error))
        plt.plot(test, label='Actual Prices')
        plt.plot(predictions, label='Predicted Prices')
        plt.legend()
        plt.show()

# Example usage:
# Initialize the strategy class with the data file and series name
#df = pd.read_csv("data.csv")
#arima_strategy = ARIMAStrategy(df, 'close')
#arima_strategy = AutoARIMAStrategy(df, 'close')

# Run the auto ARIMA research and plot results
#arima_strategy.auto_arima_research()

