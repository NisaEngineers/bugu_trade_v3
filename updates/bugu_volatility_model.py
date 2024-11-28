import numpy as np
import pandas as pd
from arch import arch_model

class ArchModels:
    def __init__(self, data):
        self.data = data
        self.data.set_index('time', inplace=True)
        self.data['log_ret'] = np.log(self.data['close']).diff() * 100  # Rescaling returns
        self.data.dropna(inplace=True)
        self.returns = self.data['log_ret']
        
    def fit_models(self):
        self.arch_model_fit = arch_model(self.returns, vol='ARCH', p=1).fit()  
        self.arch_vol = self.arch_model_fit.conditional_volatility
        
        self.garch_model_fit = arch_model(self.returns, vol='GARCH', p=1, q=1).fit() 
        self.garch_vol = self.garch_model_fit.conditional_volatility
        
        self.egarch_model_fit = arch_model(self.returns, vol='EGARCH', p=1, q=1).fit() 
        self.egarch_vol = self.egarch_model_fit.conditional_volatility
    
    def simulate_trading_strategy(self, forecast, low_threshold, high_threshold):
        if forecast < low_threshold:
            return "enter trade"
        elif forecast > high_threshold:
            return "hold/exit trade"
        else:
            return "normal trade"
    
    def backtest_strategy(self, forecast, prices, low_threshold, high_threshold):
        actions = [self.simulate_trading_strategy(f, low_threshold, high_threshold) for f in forecast]
        returns = prices.pct_change().shift(-1)
        strategy_returns = [r if a == "enter trade" else 0 for r, a in zip(returns, actions)]
        
        # Check for NaN values and handle them
        strategy_returns = [0 if np.isnan(r) else r for r in strategy_returns]
        cumulative_returns = np.cumprod(1 + np.array(strategy_returns)) - 1
        final_cumulative_return = cumulative_returns[-1] if len(cumulative_returns) > 0 else -np.inf

        return final_cumulative_return
    
    def find_best_model_and_thresholds(self, low_threshold_values, high_threshold_values):
        volatility_models = {
            "ARCH": self.arch_vol.iloc[-10:].values,
            "GARCH": self.garch_vol.iloc[-10:].values,
            "EGARCH": self.egarch_vol.iloc[-10:].values
        }
        
        prices = self.data['close'].iloc[-10:]
        
        best_cumulative_return = -np.inf
        best_low_threshold = 0.7  # Default value
        best_high_threshold = 1.0  # Default value
        best_model = "GARCH"  # Default model
        
        for model_name, forecast in volatility_models.items():
            for low_threshold in low_threshold_values:
                for high_threshold in high_threshold_values:
                    cumulative_return = self.backtest_strategy(forecast, prices, low_threshold, high_threshold)
                    if cumulative_return > best_cumulative_return:
                        best_cumulative_return = cumulative_return
                        best_low_threshold = low_threshold
                        best_high_threshold = high_threshold
                        best_model = model_name
        
        return best_model, best_low_threshold, best_high_threshold, best_cumulative_return
    
    def trading_strategy_with_optimized_thresholds(self, forecast, current_price, low_threshold, high_threshold):
        if forecast[0] < low_threshold:
            action = "enter trade"
        elif forecast[0] > high_threshold:
            action = "hold/exit trade"
        else:
            action = "normal trade"
        
        position_size = max(1 - forecast[0], 0.1)
        return action, position_size