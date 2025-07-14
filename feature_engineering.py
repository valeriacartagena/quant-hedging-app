import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class OilFeatureEngineer:
    """
    Feature engineering class for oil futures data.
    Creates technical features, term structure features, and macro features.
    """
    
    def __init__(self):
        self.features = {}
        
    def load_raw_data(self):
        """Load raw data from data_collector.py output."""
        try:
            # Load WTI futures data
            wti_data = pd.read_csv('data/raw/wti_futures.csv')
            wti_data['Date'] = pd.to_datetime(wti_data['Date'])
            wti_data.set_index('Date', inplace=True)
            
            # Load Brent futures data
            brent_data = pd.read_csv('data/raw/brent_futures.csv')
            brent_data['Date'] = pd.to_datetime(brent_data['Date'])
            brent_data.set_index('Date', inplace=True)
            
            # Load USD index data
            usd_data = pd.read_csv('data/raw/usd_index.csv')
            usd_data['Date'] = pd.to_datetime(usd_data['Date'])
            usd_data.set_index('Date', inplace=True)
            
            print("Raw data loaded successfully!")
            return wti_data, brent_data, usd_data
            
        except Exception as e:
            print(f"Error loading raw data: {e}")
            return None, None, None
    
    def clean_data(self, wti_data, brent_data, usd_data):
        """Clean and prepare data for feature engineering."""
        
        # Remove any missing values
        wti_clean = wti_data.dropna()
        brent_clean = brent_data.dropna()
        usd_clean = usd_data.dropna()
        
        # Ensure all datasets have same date range
        start_date = max(wti_clean.index.min(), brent_clean.index.min(), usd_clean.index.min())
        end_date = min(wti_clean.index.max(), brent_clean.index.max(), usd_clean.index.max())
        
        wti_clean = wti_clean[(wti_clean.index >= start_date) & (wti_clean.index <= end_date)]
        brent_clean = brent_clean[(brent_clean.index >= start_date) & (brent_clean.index <= end_date)]
        usd_clean = usd_clean[(usd_clean.index >= start_date) & (usd_clean.index <= end_date)]
        
        print(f"Data cleaned. Date range: {start_date} to {end_date}")
        return wti_clean, brent_clean, usd_clean
    
    def create_price_features(self, data, symbol):
        """Create basic price-based features."""
        features = pd.DataFrame(index=data.index)
        
        # Price levels
        features[f'{symbol}_close'] = data['Close']
        features[f'{symbol}_open'] = data['Open']
        features[f'{symbol}_high'] = data['High']
        features[f'{symbol}_low'] = data['Low']
        
        # Returns
        features[f'{symbol}_returns'] = data['Close'].pct_change()
        features[f'{symbol}_log_returns'] = np.log(data['Close'] / data['Close'].shift(1))
        
        # Price changes
        features[f'{symbol}_price_change'] = data['Close'] - data['Close'].shift(1)
        features[f'{symbol}_price_change_pct'] = (data['Close'] - data['Close'].shift(1)) / data['Close'].shift(1)
        
        # High-Low spread
        features[f'{symbol}_hl_spread'] = data['High'] - data['Low']
        features[f'{symbol}_hl_spread_pct'] = (data['High'] - data['Low']) / data['Close']
        
        # Open-Close spread
        features[f'{symbol}_oc_spread'] = data['Close'] - data['Open']
        features[f'{symbol}_oc_spread_pct'] = (data['Close'] - data['Open']) / data['Open']
        
        return features
    
    def create_technical_indicators(self, data, symbol):
        """Create technical analysis indicators."""
        features = pd.DataFrame(index=data.index)
        
        # Moving averages
        for window in [5, 10, 20, 50, 100]:
            features[f'{symbol}_sma_{window}'] = data['Close'].rolling(window=window).mean()
            features[f'{symbol}_ema_{window}'] = data['Close'].ewm(span=window).mean()
        
        # Price relative to moving averages
        for window in [20, 50, 100]:
            features[f'{symbol}_price_vs_sma_{window}'] = data['Close'] / features[f'{symbol}_sma_{window}'] - 1
            features[f'{symbol}_price_vs_ema_{window}'] = data['Close'] / features[f'{symbol}_ema_{window}'] - 1
        
        # Volatility measures
        for window in [10, 20, 50]:
            returns = data['Close'].pct_change()
            features[f'{symbol}_volatility_{window}'] = returns.rolling(window=window).std()
            features[f'{symbol}_realized_vol_{window}'] = returns.rolling(window=window).std() * np.sqrt(252)
        
        # RSI (Relative Strength Index)
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features[f'{symbol}_rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = data['Close'].ewm(span=12).mean()
        ema_26 = data['Close'].ewm(span=26).mean()
        features[f'{symbol}_macd'] = ema_12 - ema_26
        features[f'{symbol}_macd_signal'] = features[f'{symbol}_macd'].ewm(span=9).mean()
        features[f'{symbol}_macd_histogram'] = features[f'{symbol}_macd'] - features[f'{symbol}_macd_signal']
        
        # Bollinger Bands
        for window in [20]:
            sma = data['Close'].rolling(window=window).mean()
            std = data['Close'].rolling(window=window).std()
            features[f'{symbol}_bb_upper'] = sma + (std * 2)
            features[f'{symbol}_bb_lower'] = sma - (std * 2)
            features[f'{symbol}_bb_width'] = features[f'{symbol}_bb_upper'] - features[f'{symbol}_bb_lower']
            features[f'{symbol}_bb_position'] = (data['Close'] - features[f'{symbol}_bb_lower']) / features[f'{symbol}_bb_width']
        
        return features
    
    def create_spread_features(self, wti_data, brent_data):
        """Create WTI-Brent spread features."""
        features = pd.DataFrame(index=wti_data.index)
        
        # WTI-Brent spread
        features['wti_brent_spread'] = wti_data['Close'] - brent_data['Close']
        features['wti_brent_spread_pct'] = (wti_data['Close'] - brent_data['Close']) / brent_data['Close']
        
        # Spread moving averages
        for window in [5, 10, 20, 50]:
            features[f'wti_brent_spread_sma_{window}'] = features['wti_brent_spread'].rolling(window=window).mean()
        
        # Spread volatility
        for window in [10, 20, 50]:
            spread_returns = features['wti_brent_spread'].pct_change()
            features[f'wti_brent_spread_vol_{window}'] = spread_returns.rolling(window=window).std()
        
        # Spread momentum
        features['wti_brent_spread_momentum_5'] = features['wti_brent_spread'] - features['wti_brent_spread'].shift(5)
        features['wti_brent_spread_momentum_10'] = features['wti_brent_spread'] - features['wti_brent_spread'].shift(10)
        
        return features
    
    def create_macro_features(self, usd_data):
        """Create macroeconomic features from USD index."""
        features = pd.DataFrame(index=usd_data.index)
        
        # USD index levels
        features['usd_index'] = usd_data['Close']
        
        # USD returns and changes
        features['usd_returns'] = usd_data['Close'].pct_change()
        features['usd_log_returns'] = np.log(usd_data['Close'] / usd_data['Close'].shift(1))
        features['usd_change'] = usd_data['Close'] - usd_data['Close'].shift(1)
        
        # USD moving averages
        for window in [10, 20, 50, 100]:
            features[f'usd_sma_{window}'] = usd_data['Close'].rolling(window=window).mean()
            features[f'usd_price_vs_sma_{window}'] = usd_data['Close'] / features[f'usd_sma_{window}'] - 1
        
        # USD volatility
        for window in [10, 20, 50]:
            usd_returns = usd_data['Close'].pct_change()
            features[f'usd_volatility_{window}'] = usd_returns.rolling(window=window).std()
        
        # USD momentum
        features['usd_momentum_5'] = usd_data['Close'] - usd_data['Close'].shift(5)
        features['usd_momentum_10'] = usd_data['Close'] - usd_data['Close'].shift(10)
        features['usd_momentum_20'] = usd_data['Close'] - usd_data['Close'].shift(20)
        
        return features
    
    def create_lag_features(self, data, symbol, lags=[1, 2, 3, 5, 10]):
        """Create lagged features for time series modeling."""
        features = pd.DataFrame(index=data.index)
        
        # Lagged prices
        for lag in lags:
            features[f'{symbol}_close_lag_{lag}'] = data['Close'].shift(lag)
            features[f'{symbol}_returns_lag_{lag}'] = data['Close'].pct_change().shift(lag)
        
        # Lagged volumes
        for lag in lags:
            features[f'{symbol}_volume_lag_{lag}'] = data['Volume'].shift(lag)
        
        return features
    
    def create_rolling_features(self, data, symbol):
        """Create rolling window features."""
        features = pd.DataFrame(index=data.index)
        
        # Rolling statistics
        for window in [5, 10, 20, 50]:
            features[f'{symbol}_rolling_mean_{window}'] = data['Close'].rolling(window=window).mean()
            features[f'{symbol}_rolling_std_{window}'] = data['Close'].rolling(window=window).std()
            features[f'{symbol}_rolling_min_{window}'] = data['Close'].rolling(window=window).min()
            features[f'{symbol}_rolling_max_{window}'] = data['Close'].rolling(window=window).max()
            
            # Rolling percentiles
            features[f'{symbol}_rolling_25p_{window}'] = data['Close'].rolling(window=window).quantile(0.25)
            features[f'{symbol}_rolling_75p_{window}'] = data['Close'].rolling(window=window).quantile(0.75)
        
        return features
    
    def engineer_all_features(self):
        """Main function to create all features."""
        print("Starting feature engineering...")
        
        # Load and clean data
        wti_data, brent_data, usd_data = self.load_raw_data()
        if wti_data is None:
            return None
            
        wti_clean, brent_clean, usd_clean = self.clean_data(wti_data, brent_data, usd_data)
        
        # Create features for each asset
        wti_features = self.create_price_features(wti_clean, 'wti')
        wti_tech = self.create_technical_indicators(wti_clean, 'wti')
        wti_lags = self.create_lag_features(wti_clean, 'wti')
        wti_rolling = self.create_rolling_features(wti_clean, 'wti')
        
        brent_features = self.create_price_features(brent_clean, 'brent')
        brent_tech = self.create_technical_indicators(brent_clean, 'brent')
        brent_lags = self.create_lag_features(brent_clean, 'brent')
        brent_rolling = self.create_rolling_features(brent_clean, 'brent')
        
        spread_features = self.create_spread_features(wti_clean, brent_clean)
        macro_features = self.create_macro_features(usd_clean)
        
        # Combine all features
        all_features = pd.concat([
            wti_features, wti_tech, wti_lags, wti_rolling,
            brent_features, brent_tech, brent_lags, brent_rolling,
            spread_features, macro_features
        ], axis=1)
        
        # Remove any remaining NaN values
        all_features = all_features.dropna()
        
        # Save features
        all_features.to_csv('data/processed/engineered_features.csv')
        
        print(f"Feature engineering completed! Created {all_features.shape[1]} features for {all_features.shape[0]} observations")
        
        return all_features

# Main execution
if __name__ == "__main__":
    engineer = OilFeatureEngineer()
    features = engineer.engineer_all_features() 