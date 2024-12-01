from bugu_machine_learning import RidgeModel, LinearRegressionModel, DecisionTree, ExtraTrees, GradientBoosting, AdaBoost_Simple, Bagging_Simple, RandomForest_Simple, TensorflowNN, ada_boost_r, random_forest_r
from bugu_trend_confirmation_indicator import analyze_market_trend
from bugu_technical_indicators import dynamic_trading_with_key_level, MachineLearningSupertrend, HalfTrendRegression, BuySellSignal, HeikinAshiSignals, TargetTrend, SignalGenerator, SwingHighLowAnalyzer, FRAMAChannel, AbnormalCandleDetector, OrderBlocksDetector, PsychoSignal, PrevGood, PivotPoints
from bugu_real_trade import MT5DataFetcher
from bugu_news_sentiment import NewsSentimentAnalyzer
from bugu_volatility_model import ArchModels
from text_writer import write_output_to_file
from datetime import datetime
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

symbol = "BTCUSDm"
timeframes = ['H1','H4', 'D1']
data_fetcher = MT5DataFetcher(symbol, timeframes)
data_fetcher.initialize_mt5()
csv_files = data_fetcher.fetch_data()
output = f'''
Do a own research on {symbol.replace("m","")} market news analysis, sentiment and other fundamentals.
'''

full_text = " "
output += "\n"
# Current Data Section
print(f"{'#'*10} Latest data {'#'*10}")
output += f"{'#'*10} Latest data {'#'*10}\n"

for i in tqdm(range(len(timeframes))):
    data_fetcher = MT5DataFetcher(symbol, timeframes)
    data_fetcher.initialize_mt5()
    csv_files = data_fetcher.fetch_data()
    #i = int(input("Enter timeframe: [0: M30; 1: H1, 2: H4, 3:D1, 4: W1]"))
    data = pd.read_csv(f"{csv_files[i]}")
    # Access the data from the DataFrame, not the filename string
    current_high = data['high'].iloc[-1]
    current_close = data['close'].iloc[-1]
    current_low = data['low'].iloc[-1]
    print(f"Current High: {current_high}, Close: {current_close}, Low: {current_low}")
    output += f"Current High: {current_high}, Close: {current_close}, Low: {current_low}\n"
    
    
    output += f"{'-'*10} {csv_files[i]} {'-'*10}\n"
    #data = pd.read_csv(csv_files[0]) 
    print(f"{'-'*10} Indicators Outputs {'-'*10}")
    output += f"{'-'*10} Indicators Outputs {'-'*10}\n"
    
    print(f"{'.'*5} half_trend_regression {'.'*5}")
    output += f"{'.'*5} half_trend_regression {'.'*5}\n"
    data = pd.read_csv(f"{csv_files[i]}")
    data['date'] = pd.to_datetime(data['time'])
    ht_regression = HalfTrendRegression(data)
    df, future_trend = ht_regression.half_trend_regression(plot=plot)
    print(f"The future trend is: {future_trend}")
    df.drop(columns=['time.1', 'open', 'high', 'low', 'close', 'tick_volume', 'real_volume'], inplace=True)
    last_row_ht = df.iloc[-1]
    # Print all columns from the last row in one go for HalfTrendRegression
    print(last_row_ht)
    # Append all columns output to the `output` variable in one go for HalfTrendRegression
    output += f"{last_row_ht}"

    print(f"{'.'*5} BuySellSignal {'.'*5}")
    output += f"{'.'*5} BuySellSignal {'.'*5}\n"
    # Example usage
    data = pd.read_csv(f"{csv_files[i]}")
    analyzer = BuySellSignal(data)
    trend = analyzer.analyze_trend(plot=plot)
    print(f"The current trend is: {trend}")
    data = analyzer.generate_signals()
    data.drop(columns=['time.1', 'open', 'high', 'low', 'close', 'tick_volume', 'real_volume'], inplace=True)
    last_row_bs = data.iloc[-1]
    last_row_ht = last_row_bs
    print(f"""time: {last_row_ht['time']}, spread: {last_row_ht['spread']}, smrng1: {last_row_ht['smrng1']}, smrng2: {last_row_ht['smrng2']},
    smrng: {last_row_ht['smrng']}, sma: {last_row_ht['sma']}, filt: {last_row_ht['filt']}, upward: {last_row_ht['upward']},
    downward: {last_row_ht['downward']}, longCond: {last_row_ht['longCond']}, shortCond: {last_row_ht['shortCond']}, CondIni: {last_row_ht['CondIni']},
    long: {last_row_ht['long']}, short: {last_row_ht['short']}, bullishCandle: {last_row_ht['bullishCandle']}, bearishCandle: {last_row_ht['bearishCandle']},
    rsi: {last_row_ht['rsi']}, isRSIOB: {last_row_ht['isRSIOB']}, isRSIOS: {last_row_ht['isRSIOS']}, tradeSignal: {last_row_ht['tradeSignal']}, signal: {last_row_ht['signal']}
    """)
    # Append all columns output to the `output` variable in one go for BuySellSignal
    output += f"""time: {last_row_ht['time']}, spread: {last_row_ht['spread']}, smrng1: {last_row_ht['smrng1']}, smrng2: {last_row_ht['smrng2']},
    smrng: {last_row_ht['smrng']}, sma: {last_row_ht['sma']}, filt: {last_row_ht['filt']}, upward: {last_row_ht['upward']},
    downward: {last_row_ht['downward']}, longCond: {last_row_ht['longCond']}, shortCond: {last_row_ht['shortCond']}, CondIni: {last_row_ht['CondIni']},
    long: {last_row_ht['long']}, short: {last_row_ht['short']}, bullishCandle: {last_row_ht['bullishCandle']}, bearishCandle: {last_row_ht['bearishCandle']},
    rsi: {last_row_ht['rsi']}, isRSIOB: {last_row_ht['isRSIOB']}, isRSIOS: {last_row_ht['isRSIOS']}, tradeSignal: {last_row_ht['tradeSignal']}, signal: {last_row_ht['signal']}
    """
    print(f"{'.'*5} SignalGenerator {'.'*5}")
    output += f"{'.'*5} SignalGenerator {'.'*5}\n"
    df = pd.read_csv(f"{csv_files[i]}")
    signal_generator = SignalGenerator(df)
    trend = signal_generator.analyze_trend(plot=plot)
    print(f"The current trend is: {trend}")
    report = signal_generator.generate_signals()
    report.drop(columns=['time.1', 'open', 'high', 'low', 'close', 'tick_volume', 'real_volume'], inplace=True)
    last_row_sg = report.iloc[-1]
    # Print all columns from the last row in one go for SignalGenerator
    print(f"""
    time: {last_row_sg['time']}, spread: {last_row_sg['spread']}, hlc3: {last_row_sg['hlc3']}, rsi: {last_row_sg['rsi']},
    smoothed_rsi: {last_row_sg['smoothed_rsi']}, highest_high: {last_row_sg['highest_high']}, lowest_low: {last_row_sg['lowest_low']},
    williams_r: {last_row_sg['williams_r']}, smoothed_williams_r: {last_row_sg['smoothed_williams_r']}, volume_average: {last_row_sg['volume_average']},
    recent_volume_average: {last_row_sg['recent_volume_average']}, is_green_candle: {last_row_sg['is_green_candle']}, is_red_candle: {last_row_sg['is_red_candle']},
    is_high_volume_b1: {last_row_sg['is_high_volume_b1']}, is_high_volume_b2: {last_row_sg['is_high_volume_b2']}, is_high_volume_s1: {last_row_sg['is_high_volume_s1']},
    is_high_volume_s2: {last_row_sg['is_high_volume_s2']}, ema_short: {last_row_sg['ema_short']}, ema_long: {last_row_sg['ema_long']},
    bullish_ema_cross: {last_row_sg['bullish_ema_cross']}, bearish_ema_cross: {last_row_sg['bearish_ema_cross']}, atr: {last_row_sg['atr']},
    long_stop: {last_row_sg['long_stop']}, short_stop: {last_row_sg['short_stop']}, psar: {last_row_sg['psar']}, psar_dir: {last_row_sg['psar_dir']},
    sell_signal: {last_row_sg['sell_signal']}, bull_weight_2: {last_row_sg['bull_weight_2']}, bear_weight_2: {last_row_sg['bear_weight_2']},
    bull_signal_2: {last_row_sg['bull_signal_2']}, bear_signal_2: {last_row_sg['bear_signal_2']}, rsi_williams_r_bull: {last_row_sg['rsi_williams_r_bull']},
    rsi_williams_r_bear: {last_row_sg['rsi_williams_r_bear']}, b1_buy: {last_row_sg['b1_buy']}, b2_buy: {last_row_sg['b2_buy']},
    b2_strong_buy: {last_row_sg['b2_strong_buy']}, s1_sell: {last_row_sg['s1_sell']}, s2_sell: {last_row_sg['s2_sell']},
    s2_strong_sell: {last_row_sg['s2_strong_sell']}, signal: {last_row_sg['signal']}
    """)
    # Append all columns output to the `output` variable in one go for SignalGenerator
    output += f"""
    time: {last_row_sg['time']}, spread: {last_row_sg['spread']}, hlc3: {last_row_sg['hlc3']}, rsi: {last_row_sg['rsi']},
    smoothed_rsi: {last_row_sg['smoothed_rsi']}, highest_high: {last_row_sg['highest_high']}, lowest_low: {last_row_sg['lowest_low']},
    williams_r: {last_row_sg['williams_r']}, smoothed_williams_r: {last_row_sg['smoothed_williams_r']}, volume_average: {last_row_sg['volume_average']},
    recent_volume_average: {last_row_sg['recent_volume_average']}, is_green_candle: {last_row_sg['is_green_candle']}, is_red_candle: {last_row_sg['is_red_candle']},
    is_high_volume_b1: {last_row_sg['is_high_volume_b1']}, is_high_volume_b2: {last_row_sg['is_high_volume_b2']}, is_high_volume_s1: {last_row_sg['is_high_volume_s1']},
    is_high_volume_s2: {last_row_sg['is_high_volume_s2']}, ema_short: {last_row_sg['ema_short']}, ema_long: {last_row_sg['ema_long']},
    bullish_ema_cross: {last_row_sg['bullish_ema_cross']}, bearish_ema_cross: {last_row_sg['bearish_ema_cross']}, atr: {last_row_sg['atr']},
    long_stop: {last_row_sg['long_stop']}, short_stop: {last_row_sg['short_stop']}, psar: {last_row_sg['psar']}, psar_dir: {last_row_sg['psar_dir']},
    sell_signal: {last_row_sg['sell_signal']}, bull_weight_2: {last_row_sg['bull_weight_2']}, bear_weight_2: {last_row_sg['bear_weight_2']},
    bull_signal_2: {last_row_sg['bull_signal_2']}, bear_signal_2: {last_row_sg['bear_signal_2']}, rsi_williams_r_bull: {last_row_sg['rsi_williams_r_bull']},
    rsi_williams_r_bear: {last_row_sg['rsi_williams_r_bear']}, b1_buy: {last_row_sg['b1_buy']}, b2_buy: {last_row_sg['b2_buy']},
    b2_strong_buy: {last_row_sg['b2_strong_buy']}, s1_sell: {last_row_sg['s1_sell']}, s2_sell: {last_row_sg['s2_sell']},
    s2_strong_sell: {last_row_sg['s2_strong_sell']}, signal: {last_row_sg['signal']}
    """
   

    print(f"{'.'*5} Pivotpoints {'.'*5}")
    output += f"{'.'*5} Pivotpoints {'.'*5}\n"
    data = pd.read_csv(csv_files[i]) # Initialize the strategy class 
    pivot_points_strategy = PivotPoints() # Generate signals 
    data_with_signals = pivot_points_strategy.generate_signals(data) # Plot signals 
    #pivot_points_strategy.plot_signals(data_with_signals)
    data_with_signals.drop(columns=['time.1', 'open', 'high', 'low', 'close', 'tick_volume', 'real_volume'], inplace=True)
    print(data_with_signals.iloc[-1])
    output += f"{data_with_signals.iloc[-1]}"
    print(f"{'*'*10} Machine Learning Models {'*'*10}")
    output += f"{'*'*10} Machine Learning Models {'*'*10}\n"
    data = pd.read_csv(f"{csv_files[i]}")
    all_predictions_high = []
    all_predictions_low = []
    all_predictions_close = []
    
    # Initialize lists to collect all predictions
    all_predictions_high = []
    all_predictions_low = []
    all_predictions_close = []
    # Ridge Model
    print("RidgeModel")
    output += "RidgeModel\n"
    prediction_high, high_score, prediction_low, low_score, prediction_close, close_score = RidgeModel(data)
    # Ensure predictions are the same length as the data
    prediction_high = [prediction_high[-1]] * len(data.index)
    prediction_low = [prediction_low[-1]] * len(data.index)
    prediction_close = [prediction_close[-1]] * len(data.index)
    output += f"Prediction High: {prediction_high[-1]} with Accuracy: {high_score*100}%, Low: {prediction_low[-1]} with Accuracy: {low_score*100}%, Close: {prediction_close[-1]}with Accuracy: {close_score*100}%\n"
    all_predictions_high.append(prediction_high[-1])
    all_predictions_low.append(prediction_low[-1])
    all_predictions_close.append(prediction_close[-1])

    # Decision Tree
    print("DecisionTree")
    output += "DecisionTree\n"
    prediction_high, high_score, prediction_low, low_score, prediction_close, close_score = DecisionTree(data)
    # Ensure predictions are the same length as the data
    prediction_high = [prediction_high[-1]] * len(data.index)
    prediction_low = [prediction_low[-1]] * len(data.index)
    prediction_close = [prediction_close[-1]] * len(data.index)
    f"Prediction High: {prediction_high[-1]} with Accuracy: {high_score*100}%, Low: {prediction_low[-1]} with Accuracy: {low_score*100}%, Close: {prediction_close[-1]}with Accuracy: {close_score*100}%\n"
    all_predictions_high.append(prediction_high[-1])
    all_predictions_low.append(prediction_low[-1])
    all_predictions_close.append(prediction_close[-1])
   
    # Linear Regression Model
    print("LinearRegressionModel")
    output += "LinearRegressionModel\n"
    prediction_high, high_score, prediction_low, low_score, prediction_close, close_score = LinearRegressionModel(data)
    # Ensure predictions are the same length as the data
    prediction_high = [prediction_high[-1]] * len(data.index)
    prediction_low = [prediction_low[-1]] * len(data.index)
    prediction_close = [prediction_close[-1]] * len(data.index)
    f"Prediction High: {prediction_high[-1]} with Accuracy: {high_score*100}%, Low: {prediction_low[-1]} with Accuracy: {low_score*100}%, Close: {prediction_close[-1]}with Accuracy: {close_score*100}%\n"
    all_predictions_high.append(prediction_high[-1])
    all_predictions_low.append(prediction_low[-1])
    all_predictions_close.append(prediction_close[-1])
  
    # Bagging
    print("Bagging")
    output += "Bagging\n"
    prediction_high, high_score, prediction_low, low_score, prediction_close, close_score = Bagging_Simple(data)
    # Ensure predictions are the same length as the data
    prediction_high = [prediction_high[-1]] * len(data.index)
    prediction_low = [prediction_low[-1]] * len(data.index)
    prediction_close = [prediction_close[-1]] * len(data.index)
    f"Prediction High: {prediction_high[-1]} with Accuracy: {high_score*100}%, Low: {prediction_low[-1]} with Accuracy: {low_score*100}%, Close: {prediction_close[-1]}with Accuracy: {close_score*100}%\n"
    all_predictions_high.append(prediction_high[-1])
    all_predictions_low.append(prediction_low[-1])
    all_predictions_close.append(prediction_close[-1])
   
   
    # Aggregate predictions
    avg_prediction_high = np.mean(all_predictions_high)
    avg_prediction_low = np.mean(all_predictions_low)
    avg_prediction_close = np.mean(all_predictions_close)
    # Make a consolidated decision based on aggregated predictions
    
    print("\n********** GARCH Model Volatility Prediction **********")
    arch_models = ArchModels(data)
    arch_models.fit_models()
    low_threshold_values = np.arange(0.5, 1.0, 0.1)
    high_threshold_values = np.arange(1.0, 2.0, 0.1)
    best_model, best_low_threshold, best_high_threshold, best_cumulative_return = arch_models.find_best_model_and_thresholds(low_threshold_values, high_threshold_values)
    forecast_volatility = arch_models.garch_model_fit.forecast(horizon=5)
    forecast = forecast_volatility.variance.iloc[-1].values
    current_price = data['close'].iloc[-1]
    action, position_size = arch_models.trading_strategy_with_optimized_thresholds(forecast, current_price, best_low_threshold, best_high_threshold)
    print(f"Action: {action}, Position Size: {position_size}, Forecasted Volatility for next 1 period: {forecast[0]}")
    output += "\n********** GARCH Model Volatility Prediction **********"
    output += f"Action: {action}\n"
    output += f"Position Size: {position_size}\n"
    output += f"Forecasted Volatility for next 1 period: {forecast[0]}\n"
    # Trend Direction and Strength
    print("\n********** Trend Direction and Strength **********")
    trend, volume_strength, support_level, resistance_level = analyze_market_trend(data, lookback_period=20, atr_period=14)
    print(f"Trend: {trend}, Volume Strength: {volume_strength}, Support: {support_level}, Resistance: {resistance_level}")
    output += "\n********** Trend Direction and Strength **********\n"
    output += f"Trend: {trend}, Volume Strength: {volume_strength}, Support: {support_level}, Resistance: {resistance_level}\n"
    output += f"Current High: {current_high}, Close: {current_close}, Low: {current_low}\n"
    full_text = output
    
        
    
     # Write to file and print
    current_time = datetime.now().strftime("%d%b%H%M")
    filename = f"{symbol}_analysis_{current_time}.txt"
    print(filename)
    write_output_to_file(output, filename)
 # Write to file and print
    
filename = f"{symbol}_analysis_FULL.txt"
print(filename)
write_output_to_file(full_text, filename)