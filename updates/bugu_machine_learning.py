import numpy as np
from sklearn import tree, ensemble
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import Ridge
import time
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def AdaBoost(df):
    n = len(df)
    X_close = np.asarray(df[['open', 'high', 'low', 'tick_volume']][:n-1]).reshape(n-1, 4)
    y_close = np.asarray(df[['close']][1:]).reshape(n-1, 1)
    X_high = np.asarray(df[['open', 'close', 'low', 'tick_volume']][:n-1]).reshape(n-1, 4)
    y_high = np.asarray(df[['high']][1:]).reshape(n-1, 1)
    X_low = np.asarray(df[['open', 'high', 'close', 'tick_volume']][:n-1]).reshape(n-1, 4)
    y_low = np.asarray(df[['low']][1:]).reshape(n-1, 1)
    
    split = 0.8
    X_train_close, X_test_close, y_train_close, y_test_close = train_test_split(X_close, y_close, train_size=split)
    X_train_high, X_test_high, y_train_high, y_test_high = train_test_split(X_high, y_high, train_size=split)
    X_train_low, X_test_low, y_train_low, y_test_low = train_test_split(X_low, y_low, train_size=split)
    
    dtree = tree.DecisionTreeRegressor(max_depth=100)
    
    model_close = ensemble.AdaBoostRegressor(n_estimators=100, learning_rate=2.0, estimator=dtree)
    model_close.fit(X_train_close, y_train_close.ravel())
    close_score = model_close.score(X_test_close, y_test_close)
    
    model_high = ensemble.AdaBoostRegressor(n_estimators=100, learning_rate=2.0, estimator=dtree)
    model_high.fit(X_train_high, y_train_high.ravel())
    high_score = model_high.score(X_test_high, y_test_high)
    
    model_low = ensemble.AdaBoostRegressor(n_estimators=100, learning_rate=2.0, estimator=dtree)
    model_low.fit(X_train_low, y_train_low.ravel())
    low_score = model_low.score(X_test_low, y_test_low)
    
    prediction_high = model_high.predict(df[['open', 'close', 'low', 'tick_volume']].values[-1].reshape(1, -1))
    prediction_close = model_close.predict(df[['open', 'high', 'low', 'tick_volume']].values[-1].reshape(1, -1))
    prediction_low = model_low.predict(df[['open', 'high', 'close', 'tick_volume']].values[-1].reshape(1, -1))

    

    print(f"High: accuracy {high_score*100:.2f}%, current {df['high'].values[-1]}, predicted {prediction_high}")
    print(f"Close: accuracy {close_score*100:.2f}%, current {df['close'].values[-1]}, predicted {prediction_close}")
    print(f"Low: accuracy {low_score*100:.2f}%, current {df['low'].values[-1]}, predicted {prediction_low}")

   
    return prediction_high, high_score, prediction_low, low_score, prediction_close, close_score

def LinearRegressionModel(df):
    n = len(df)
    X_close = np.asarray(df[['open', 'high', 'low', 'tick_volume']][:n-1]).reshape(n-1, 4)
    y_close = np.asarray(df[['close']][1:]).reshape(n-1, 1)
    X_high = np.asarray(df[['open', 'close', 'low', 'tick_volume']][:n-1]).reshape(n-1, 4)
    y_high = np.asarray(df[['high']][1:]).reshape(n-1, 1)
    X_low = np.asarray(df[['open', 'high', 'close', 'tick_volume']][:n-1]).reshape(n-1, 4)
    y_low = np.asarray(df[['low']][1:]).reshape(n-1, 1)
    
    split = 0.8
    X_train_close, X_test_close, y_train_close, y_test_close = train_test_split(X_close, y_close, train_size=split)
    X_train_high, X_test_high, y_train_high, y_test_high = train_test_split(X_high, y_high, train_size=split)
    X_train_low, X_test_low, y_train_low, y_test_low = train_test_split(X_low, y_low, train_size=split)

    model_close = LinearRegression()
    model_close.fit(X_train_close, y_train_close.ravel())
    close_score = model_close.score(X_test_close, y_test_close)
    
    model_high = LinearRegression()
    model_high.fit(X_train_high, y_train_high.ravel())
    high_score = model_high.score(X_test_high, y_test_high)
    
    model_low = LinearRegression()
    model_low.fit(X_train_low, y_train_low.ravel())
    low_score = model_low.score(X_test_low, y_test_low)

    prediction_high = model_high.predict(df[['open', 'close', 'low', 'tick_volume']].values[-1].reshape(1, -1))
    prediction_close = model_close.predict(df[['open', 'high', 'low', 'tick_volume']].values[-1].reshape(1, -1))
    prediction_low = model_low.predict(df[['open', 'high', 'close', 'tick_volume']].values[-1].reshape(1, -1))

    

    print(f"High: accuracy {high_score*100:.2f}%, current {df['high'].values[-1]}, predicted {prediction_high}")
    print(f"Close: accuracy {close_score*100:.2f}%, current {df['close'].values[-1]}, predicted {prediction_close}")
    print(f"Low: accuracy {low_score*100:.2f}%, current {df['low'].values[-1]}, predicted {prediction_low}")

    return prediction_high, high_score, prediction_low, low_score, prediction_close, close_score

def DecisionTree(df):
    n = len(df)
    X_close = np.asarray(df[['open', 'high', 'low', 'tick_volume']][:n-1]).reshape(n-1, 4)
    y_close = np.asarray(df[['close']][1:]).reshape(n-1, 1)
    X_high = np.asarray(df[['open', 'close', 'low', 'tick_volume']][:n-1]).reshape(n-1, 4)
    y_high = np.asarray(df[['high']][1:]).reshape(n-1, 1)
    X_low = np.asarray(df[['open', 'high', 'close', 'tick_volume']][:n-1]).reshape(n-1, 4)
    y_low = np.asarray(df[['low']][1:]).reshape(n-1, 1)
    
    split = 0.8
    X_train_close, X_test_close, y_train_close, y_test_close = train_test_split(X_close, y_close, train_size=split)
    X_train_high, X_test_high, y_train_high, y_test_high = train_test_split(X_high, y_high, train_size=split)
    X_train_low, X_test_low, y_train_low, y_test_low = train_test_split(X_low, y_low, train_size=split)

    model_close = tree.DecisionTreeRegressor(max_depth=100)
    model_close.fit(X_train_close, y_train_close.ravel())
    close_score = model_close.score(X_test_close, y_test_close)
    
    model_high = tree.DecisionTreeRegressor(max_depth=100)
    model_high.fit(X_train_high, y_train_high.ravel())
    high_score = model_high.score(X_test_high, y_test_high)
    
    model_low = tree.DecisionTreeRegressor(max_depth=100)
    model_low.fit(X_train_low, y_train_low.ravel())
    low_score = model_low.score(X_test_low, y_test_low)

    prediction_high = model_high.predict(df[['open', 'close', 'low', 'tick_volume']].values[-1].reshape(1, -1))
    prediction_close = model_close.predict(df[['open', 'high', 'low', 'tick_volume']].values[-1].reshape(1, -1))
    prediction_low = model_low.predict(df[['open', 'high', 'close', 'tick_volume']].values[-1].reshape(1, -1))

    

    print(f"High: accuracy {high_score*100:.2f}%, current {df['high'].values[-1]}, predicted {prediction_high}")
    print(f"Close: accuracy {close_score*100:.2f}%, current {df['close'].values[-1]}, predicted {prediction_close}")
    print(f"Low: accuracy {low_score*100:.2f}%, current {df['low'].values[-1]}, predicted {prediction_low}")

    return prediction_high, high_score, prediction_low, low_score, prediction_close, close_score

def Bagging(df):
    n = len(df)
    X_close = np.asarray(df[['open', 'high', 'low', 'tick_volume']][:n-1]).reshape(n-1, 4)
    y_close = np.asarray(df[['close']][1:]).reshape(n-1, 1)
    X_high = np.asarray(df[['open', 'close', 'low', 'tick_volume']][:n-1]).reshape(n-1, 4)
    y_high = np.asarray(df[['high']][1:]).reshape(n-1, 1)
    X_low = np.asarray(df[['open', 'high', 'close', 'tick_volume']][:n-1]).reshape(n-1, 4)
    y_low = np.asarray(df[['low']][1:]).reshape(n-1, 1)
    
    split = 0.8
    X_train_close, X_test_close, y_train_close, y_test_close = train_test_split(X_close, y_close, train_size=split)
    X_train_high, X_test_high, y_train_high, y_test_high = train_test_split(X_high, y_high, train_size=split)
    X_train_low, X_test_low, y_train_low, y_test_low = train_test_split(X_low, y_low, train_size=split)

    model_close = BaggingRegressor(n_estimators=100)
    model_close.fit(X_train_close, y_train_close.ravel())
    close_score = model_close.score(X_test_close, y_test_close)
    
    model_high = BaggingRegressor(n_estimators=100)
    model_high.fit(X_train_high, y_train_high.ravel())
    high_score = model_high.score(X_test_high, y_test_high)
    
    model_low = BaggingRegressor(n_estimators=100)
    model_low.fit(X_train_low, y_train_low.ravel())
    low_score = model_low.score(X_test_low, y_test_low)

    prediction_high = model_high.predict(df[['open', 'close', 'low', 'tick_volume']].values[-1].reshape(1, -1))
    prediction_close = model_close.predict(df[['open', 'high', 'low', 'tick_volume']].values[-1].reshape(1, -1))
    prediction_low = model_low.predict(df[['open', 'high', 'close', 'tick_volume']].values[-1].reshape(1, -1))

    print(f"High: accuracy {high_score*100:.2f}%, current {df['high'].values[-1]}, predicted {prediction_high}")
    print(f"Close: accuracy {close_score*100:.2f}%, current {df['close'].values[-1]}, predicted {prediction_close}")
    print(f"Low: accuracy {low_score*100:.2f}%, current {df['low'].values[-1]}, predicted {prediction_low}")

    return prediction_high, high_score, prediction_low, low_score, prediction_close, close_score

def RandomForest(df):
    n = len(df)
    X_close = np.asarray(df[['open', 'high', 'low', 'tick_volume']][:n-1]).reshape(n-1, 4)
    y_close = np.asarray(df[['close']][1:]).reshape(n-1, 1)
    X_high = np.asarray(df[['open', 'close', 'low', 'tick_volume']][:n-1]).reshape(n-1, 4)
    y_high = np.asarray(df[['high']][1:]).reshape(n-1, 1)
    X_low = np.asarray(df[['open', 'high', 'close', 'tick_volume']][:n-1]).reshape(n-1, 4)
    y_low = np.asarray(df[['low']][1:]).reshape(n-1, 1)
    
    split = 0.8
    X_train_close, X_test_close, y_train_close, y_test_close = train_test_split(X_close, y_close, train_size=split)
    X_train_high, X_test_high, y_train_high, y_test_high = train_test_split(X_high, y_high, train_size=split)
    X_train_low, X_test_low, y_train_low, y_test_low = train_test_split(X_low, y_low, train_size=split)

    model_close = ensemble.RandomForestRegressor(n_estimators=100, max_depth=100)
    model_close.fit(X_train_close, y_train_close.ravel())
    close_score = model_close.score(X_test_close, y_test_close)
    
    model_high = ensemble.RandomForestRegressor(n_estimators=100, max_depth=100)
    model_high.fit(X_train_high, y_train_high.ravel())
    high_score = model_high.score(X_test_high, y_test_high)
    
    model_low = ensemble.RandomForestRegressor(n_estimators=100, max_depth=100)
    model_low.fit(X_train_low, y_train_low.ravel())
    low_score = model_low.score(X_test_low, y_test_low)

    prediction_high = model_high.predict(df[['open', 'close', 'low', 'tick_volume']].values[-1].reshape(1, -1))
    prediction_close = model_close.predict(df[['open', 'high', 'low', 'tick_volume']].values[-1].reshape(1, -1))
    prediction_low = model_low.predict(df[['open', 'high', 'close', 'tick_volume']].values[-1].reshape(1, -1))

    print(f"High: accuracy {high_score*100:.2f}%, current {df['high'].values[-1]}, predicted {prediction_high}")
    print(f"Close: accuracy {close_score*100:.2f}%, current {df['close'].values[-1]}, predicted {prediction_close}")
    print(f"Low: accuracy {low_score*100:.2f}%, current {df['low'].values[-1]}, predicted {prediction_low}")

    return prediction_high, high_score, prediction_low, low_score, prediction_close, close_score

def RidgeModel(df):
    n = len(df)
    X_close = np.asarray(df[['open', 'high', 'low', 'tick_volume']][:n-1]).reshape(n-1, 4)
    y_close = np.asarray(df[['close']][1:]).reshape(n-1, 1)
    X_high = np.asarray(df[['open', 'close', 'low', 'tick_volume']][:n-1]).reshape(n-1, 4)
    y_high = np.asarray(df[['high']][1:]).reshape(n-1, 1)
    X_low = np.asarray(df[['open', 'high', 'close', 'tick_volume']][:n-1]).reshape(n-1, 4)
    y_low = np.asarray(df[['low']][1:]).reshape(n-1, 1)
    
    split = 0.8
    X_train_close, X_test_close, y_train_close, y_test_close = train_test_split(X_close, y_close, train_size=split)
    X_train_high, X_test_high, y_train_high, y_test_high = train_test_split(X_high, y_high, train_size=split)
    X_train_low, X_test_low, y_train_low, y_test_low = train_test_split(X_low, y_low, train_size=split)

    model_close = Ridge()
    model_close.fit(X_train_close, y_train_close.ravel())
    close_score = model_close.score(X_test_close, y_test_close)
    
    model_high = Ridge()
    model_high.fit(X_train_high, y_train_high.ravel())
    high_score = model_high.score(X_test_high, y_test_high)
    
    model_low = Ridge()
    model_low.fit(X_train_low, y_train_low.ravel())
    low_score = model_low.score(X_test_low, y_test_low)

    prediction_high = model_high.predict(df[['open', 'close', 'low', 'tick_volume']].values[-1].reshape(1, -1))
    prediction_close = model_close.predict(df[['open', 'high', 'low', 'tick_volume']].values[-1].reshape(1, -1))
    prediction_low = model_low.predict(df[['open', 'high', 'close', 'tick_volume']].values[-1].reshape(1, -1))

    print(f"High: accuracy {high_score*100:.2f}%, current {df['high'].values[-1]}, predicted {prediction_high}")
    print(f"Close: accuracy {close_score*100:.2f}%, current {df['close'].values[-1]}, predicted {prediction_close}")
    print(f"Low: accuracy {low_score*100:.2f}%, current {df['low'].values[-1]}, predicted {prediction_low}")

    return prediction_high, high_score, prediction_low, low_score, prediction_close, close_score

def ExtraTrees(df):
    n = len(df)
    X_close = np.asarray(df[['open', 'high', 'low', 'tick_volume']][:n-1]).reshape(n-1, 4)
    y_close = np.asarray(df[['close']][1:]).reshape(n-1, 1)
    X_high = np.asarray(df[['open', 'close', 'low', 'tick_volume']][:n-1]).reshape(n-1, 4)
    y_high = np.asarray(df[['high']][1:]).reshape(n-1, 1)
    X_low = np.asarray(df[['open', 'high', 'close', 'tick_volume']][:n-1]).reshape(n-1, 4)
    y_low = np.asarray(df[['low']][1:]).reshape(n-1, 1)
    
    split = 0.8
    X_train_close, X_test_close, y_train_close, y_test_close = train_test_split(X_close, y_close, train_size=split)
    X_train_high, X_test_high, y_train_high, y_test_high = train_test_split(X_high, y_high, train_size=split)
    X_train_low, X_test_low, y_train_low, y_test_low = train_test_split(X_low, y_low, train_size=split)

    model_close = ensemble.ExtraTreesRegressor(n_estimators=100, max_depth=100)
    model_close.fit(X_train_close, y_train_close.ravel())
    close_score = model_close.score(X_test_close, y_test_close)
    
    model_high = ensemble.ExtraTreesRegressor(n_estimators=100, max_depth=100)
    model_high.fit(X_train_high, y_train_high.ravel())
    high_score = model_high.score(X_test_high, y_test_high)
    
    model_low = ensemble.ExtraTreesRegressor(n_estimators=100, max_depth=100)
    model_low.fit(X_train_low, y_train_low.ravel())
    low_score = model_low.score(X_test_low, y_test_low)

    prediction_high = model_high.predict(df[['open', 'close', 'low', 'tick_volume']].values[-1].reshape(1, -1))
    prediction_close = model_close.predict(df[['open', 'high', 'low', 'tick_volume']].values[-1].reshape(1, -1))
    prediction_low = model_low.predict(df[['open', 'high', 'close', 'tick_volume']].values[-1].reshape(1, -1))

    print(f"High: accuracy {high_score*100:.2f}%, current {df['high'].values[-1]}, predicted {prediction_high}")
    print(f"Close: accuracy {close_score*100:.2f}%, current {df['close'].values[-1]}, predicted {prediction_close}")
    print(f"Low: accuracy {low_score*100:.2f}%, current {df['low'].values[-1]}, predicted {prediction_low}")

    return prediction_high, high_score, prediction_low, low_score, prediction_close, close_score

def GradientBoosting(df):
    n = len(df)
    X_close = np.asarray(df[['open', 'high', 'low', 'tick_volume']][:n-1]).reshape(n-1, 4)
    y_close = np.asarray(df[['close']][1:]).reshape(n-1, 1)
    X_high = np.asarray(df[['open', 'close', 'low', 'tick_volume']][:n-1]).reshape(n-1, 4)
    y_high = np.asarray(df[['high']][1:]).reshape(n-1, 1)
    X_low = np.asarray(df[['open', 'high', 'close', 'tick_volume']][:n-1]).reshape(n-1, 4)
    y_low = np.asarray(df[['low']][1:]).reshape(n-1, 1)
    
    split = 0.8
    X_train_close, X_test_close, y_train_close, y_test_close = train_test_split(X_close, y_close, train_size=split)
    X_train_high, X_test_high, y_train_high, y_test_high = train_test_split(X_high, y_high, train_size=split)
    X_train_low, X_test_low, y_train_low, y_test_low = train_test_split(X_low, y_low, train_size=split)

    model_close = ensemble.GradientBoostingRegressor(n_estimators=100, max_depth=100)
    model_close.fit(X_train_close, y_train_close.ravel())
    close_score = model_close.score(X_test_close, y_test_close)

    model_high = ensemble.GradientBoostingRegressor(n_estimators=100, max_depth=100)
    model_high.fit(X_train_high, y_train_high.ravel())
    high_score = model_high.score(X_test_high, y_test_high)

    model_low = ensemble.GradientBoostingRegressor(n_estimators=100, max_depth=100)
    model_low.fit(X_train_low, y_train_low.ravel())
    low_score = model_low.score(X_test_low, y_test_low)

    prediction_high = model_high.predict(df[['open', 'close', 'low', 'tick_volume']].values[-1].reshape(1, -1))
    prediction_close = model_close.predict(df[['open', 'high', 'low', 'tick_volume']].values[-1].reshape(1, -1))
    prediction_low = model_low.predict(df[['open', 'high', 'close', 'tick_volume']].values[-1].reshape(1, -1))

    print(f"High: accuracy {high_score*100:.2f}%, current {df['high'].values[-1]}, predicted {prediction_high}")
    print(f"Close: accuracy {close_score*100:.2f}%, current {df['close'].values[-1]}, predicted {prediction_close}")
    print(f"Low: accuracy {low_score*100:.2f}%, current {df['low'].values[-1]}, predicted {prediction_low}")

    return prediction_high, high_score, prediction_low, low_score, prediction_close, close_score

def Bagging_Simple(df):
    n = len(df)
    X_close = np.asarray(df[['open', 'high', 'low', 'tick_volume']][:n-1]).reshape(n-1, 4)
    y_close = np.asarray(df[['close']][1:]).reshape(n-1, 1)
    X_high = np.asarray(df[['open', 'close', 'low', 'tick_volume']][:n-1]).reshape(n-1, 4)
    y_high = np.asarray(df[['high']][1:]).reshape(n-1, 1)
    X_low = np.asarray(df[['open', 'high', 'close', 'tick_volume']][:n-1]).reshape(n-1, 4)
    y_low = np.asarray(df[['low']][1:]).reshape(n-1, 1)
    
    split = 0.8
    X_train_close, X_test_close, y_train_close, y_test_close = train_test_split(X_close, y_close, train_size=split)
    X_train_high, X_test_high, y_train_high, y_test_high = train_test_split(X_high, y_high, train_size=split)
    X_train_low, X_test_low, y_train_low, y_test_low = train_test_split(X_low, y_low, train_size=split)

    model_close = BaggingRegressor()
    model_close.fit(X_train_close, y_train_close.ravel())
    close_score = model_close.score(X_test_close, y_test_close)
    
    model_high = BaggingRegressor()
    model_high.fit(X_train_high, y_train_high.ravel())
    high_score = model_high.score(X_test_high, y_test_high)
    
    model_low = BaggingRegressor()
    model_low.fit(X_train_low, y_train_low.ravel())
    low_score = model_low.score(X_test_low, y_test_low)

    prediction_high = model_high.predict(df[['open', 'close', 'low', 'tick_volume']].values[-1].reshape(1, -1))
    prediction_close = model_close.predict(df[['open', 'high', 'low', 'tick_volume']].values[-1].reshape(1, -1))
    prediction_low = model_low.predict(df[['open', 'high', 'close', 'tick_volume']].values[-1].reshape(1, -1))

    print(f"High: accuracy {high_score*100:.2f}%, current {df['high'].values[-1]}, predicted {prediction_high}")
    print(f"Close: accuracy {close_score*100:.2f}%, current {df['close'].values[-1]}, predicted {prediction_close}")
    print(f"Low: accuracy {low_score*100:.2f}%, current {df['low'].values[-1]}, predicted {prediction_low}")

    return prediction_high, high_score, prediction_low, low_score, prediction_close, close_score

def RandomForest_Simple(df):
    n = len(df)
    X_close = np.asarray(df[['open', 'high', 'low', 'tick_volume']][:n-1]).reshape(n-1, 4)
    y_close = np.asarray(df[['close']][1:]).reshape(n-1, 1)
    X_high = np.asarray(df[['open', 'close', 'low', 'tick_volume']][:n-1]).reshape(n-1, 4)
    y_high = np.asarray(df[['high']][1:]).reshape(n-1, 1)
    X_low = np.asarray(df[['open', 'high', 'close', 'tick_volume']][:n-1]).reshape(n-1, 4)
    y_low = np.asarray(df[['low']][1:]).reshape(n-1, 1)
    
    split = 0.8
    X_train_close, X_test_close, y_train_close, y_test_close = train_test_split(X_close, y_close, train_size=split)
    X_train_high, X_test_high, y_train_high, y_test_high = train_test_split(X_high, y_high, train_size=split)
    X_train_low, X_test_low, y_train_low, y_test_low = train_test_split(X_low, y_low, train_size=split)

    model_close = ensemble.RandomForestRegressor()
    model_close.fit(X_train_close, y_train_close.ravel())
    close_score = model_close.score(X_test_close, y_test_close)
    
    model_high = ensemble.RandomForestRegressor()
    model_high.fit(X_train_high, y_train_high.ravel())
    high_score = model_high.score(X_test_high, y_test_high)
    
    model_low = ensemble.RandomForestRegressor()
    model_low.fit(X_train_low, y_train_low.ravel())
    low_score = model_low.score(X_test_low, y_test_low)

    prediction_high = model_high.predict(df[['open', 'close', 'low', 'tick_volume']].values[-1].reshape(1, -1))
    prediction_close = model_close.predict(df[['open', 'high', 'low', 'tick_volume']].values[-1].reshape(1, -1))
    prediction_low = model_low.predict(df[['open', 'high', 'close', 'tick_volume']].values[-1].reshape(1, -1))

    print(f"High: accuracy {high_score*100:.2f}%, current {df['high'].values[-1]}, predicted {prediction_high}")
    print(f"Close: accuracy {close_score*100:.2f}%, current {df['close'].values[-1]}, predicted {prediction_close}")
    print(f"Low: accuracy {low_score*100:.2f}%, current {df['low'].values[-1]}, predicted {prediction_low}")

    return prediction_high, high_score, prediction_low, low_score, prediction_close, close_score

def AdaBoost_Simple(df):
    n = len(df)
    X_close = np.asarray(df[['open', 'high', 'low', 'tick_volume']][:n-1]).reshape(n-1, 4)
    y_close = np.asarray(df[['close']][1:]).reshape(n-1, 1)
    X_high = np.asarray(df[['open', 'close', 'low', 'tick_volume']][:n-1]).reshape(n-1, 4)
    y_high = np.asarray(df[['high']][1:]).reshape(n-1, 1)
    X_low = np.asarray(df[['open', 'high', 'close', 'tick_volume']][:n-1]).reshape(n-1, 4)
    y_low = np.asarray(df[['low']][1:]).reshape(n-1, 1)
    
    split = 0.8
    X_train_close, X_test_close, y_train_close, y_test_close = train_test_split(X_close, y_close, train_size=split)
    X_train_high, X_test_high, y_train_high, y_test_high = train_test_split(X_high, y_high, train_size=split)
    X_train_low, X_test_low, y_train_low, y_test_low = train_test_split(X_low, y_low, train_size=split)
    
    dtree = tree.DecisionTreeRegressor()
    
    model_close = ensemble.AdaBoostRegressor(n_estimators=10, learning_rate=2.0, estimator=dtree)
    model_close.fit(X_train_close, y_train_close.ravel())
    close_score = model_close.score(X_test_close, y_test_close)
    
    model_high = ensemble.AdaBoostRegressor(n_estimators=10, learning_rate=2.0, estimator=dtree)
    model_high.fit(X_train_high, y_train_high.ravel())
    high_score = model_high.score(X_test_high, y_test_high)
    
    model_low = ensemble.AdaBoostRegressor(n_estimators=10, learning_rate=2.0, estimator=dtree)
    model_low.fit(X_train_low, y_train_low.ravel())
    low_score = model_low.score(X_test_low, y_test_low)
    
    prediction_high = model_high.predict(df[['open', 'close', 'low', 'tick_volume']].values[-1].reshape(1, -1))
    prediction_close = model_close.predict(df[['open', 'high', 'low', 'tick_volume']].values[-1].reshape(1, -1))
    prediction_low = model_low.predict(df[['open', 'high', 'close', 'tick_volume']].values[-1].reshape(1, -1))

    

    print(f"High: accuracy {high_score*100:.2f}%, current {df['high'].values[-1]}, predicted {prediction_high}")
    print(f"Close: accuracy {close_score*100:.2f}%, current {df['close'].values[-1]}, predicted {prediction_close}")
    print(f"Low: accuracy {low_score*100:.2f}%, current {df['low'].values[-1]}, predicted {prediction_low}")

    return prediction_high, high_score, prediction_low, low_score, prediction_close, close_score

class TensorflowNN:
    def __init__(self, df, plotting=None):
        self.df = df
        self.plotting = plotting

    def preprocess_data(self):
        # Feature Engineering
        features = self.df[['open', 'high', 'low', 'close', 'tick_volume']]
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        features_scaled = self.scaler.fit_transform(features)

        # Build X and y
        X = features_scaled[:, 1:]
        y = features_scaled[:, 0]

        # Split data into training and testing sets
        train_size = int(len(X) * 0.8)
        self.X_train = X[:train_size]
        self.y_train = y[:train_size]
        self.X_test = X[train_size:]
        self.y_test = y[train_size:]

    def build_model(self):
        n_stocks = self.X_train.shape[1]

        # Neurons
        n_neurons_1 = 1024
        n_neurons_2 = 512
        n_neurons_3 = 256
        n_neurons_4 = 128

        # Build the model
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(n_neurons_1, input_dim=n_stocks, activation='relu'),
            tf.keras.layers.Dense(n_neurons_2, activation='relu'),
            tf.keras.layers.Dense(n_neurons_3, activation='relu'),
            tf.keras.layers.Dense(n_neurons_4, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

        # Compile the model
        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', 'mse'])

    def train_model(self, epochs=100, batch_size=256):
        self.model.fit(self.X_train, self.y_train,
                       validation_data=(self.X_test, self.y_test),
                       epochs=epochs,
                       batch_size=batch_size,
                       verbose=0)  # No training loop output

        # Save the model
        self.model.save('model.h5')
        print("Model saved successfully!")

    def predict(self):
        # Predict
        self.y_pred = self.model.predict(self.X_test)

        # Inverse transform to get actual values
        self.y_test_inv = self.scaler.inverse_transform(np.concatenate((self.y_test.reshape(-1, 1), self.X_test), axis=1))[:, 0]
        self.y_pred_inv = self.scaler.inverse_transform(np.concatenate((self.y_pred, self.X_test), axis=1))[:, 0]

        # Calculate prediction accuracy (mean absolute percentage error)
        self.mape = np.mean(np.abs((self.y_test_inv - self.y_pred_inv) / self.y_test_inv)) * 100
        print(f'prediction MAPE: {self.mape:.2f}%')

    def plot_results(self):
        if self.plotting:
            # Plot the predictions
            plt.figure(figsize=(14, 7))
            plt.plot(self.y_test_inv, label='Actual Close Price')
            plt.plot(self.y_pred_inv, label='Predicted Close Price')
            plt.title('Actual vs Predicted Close Price')
            plt.xlabel('Index')
            plt.ylabel('Price')
            plt.legend()
            plt.show()

    def get_predictions(self):
        # Predicted high, low, and close prices
        predicted_high = self.scaler.inverse_transform(np.concatenate((self.y_pred, self.X_test), axis=1))[:, 1]
        predicted_low = self.scaler.inverse_transform(np.concatenate((self.y_pred, self.X_test), axis=1))[:, 2]
        predicted_close = self.y_pred_inv

        return predicted_high[-1], predicted_low[-1], predicted_close[-1]

    def run(self):
        self.preprocess_data()
        self.build_model()
        self.train_model()
        self.predict()
        self.plot_results()
        return self.get_predictions()




def ada_boost_r(df):
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
    return prediction, score

    
def random_forest_r(df):
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
    return prediction, score