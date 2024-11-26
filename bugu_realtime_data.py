import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime

class MT5DataFetcher:
    def __init__(self, symbol, timeframes):
        self.symbol = symbol
        self.timeframes = timeframes

    def initialize_mt5(self):
        if not mt5.initialize(login=202213476, server="Exness-MT5Trial7", password="BuguPipu_01"):
            print('initialize() failed, error code =', mt5.last_error())
            quit()

    def fetch_data(self):
        csv_list = []
        timeframes_map = {
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
            "W1": mt5.TIMEFRAME_W1,
            "MN1": mt5.TIMEFRAME_MN1
        }

        for timeframe_name in self.timeframes:
            timeframe = timeframes_map.get(timeframe_name)
            if timeframe is None:
                print(f"Timeframe {timeframe_name} is not recognized")
                continue
            
            bars = mt5.copy_rates_from_pos(self.symbol, timeframe, 0, 99999)
            df = pd.DataFrame(bars)
            df.set_index(pd.to_datetime(df['time'], unit='s'), inplace=True)
            csv_file = f'{self.symbol}_{timeframe_name}_data.csv'
            df.to_csv(csv_file)
            print(f"Downloaded {timeframe_name} data for {self.symbol}")
            csv_list.append(csv_file)

        return csv_list


