import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from typing import Optional, List

class DataProcessor:
    """Handles advanced data cleaning, validation, and feature engineering."""
    
    @staticmethod
    def validate_data(df: pd.DataFrame, date_col: str, target_col: str):
        """Validates basic data integrity."""
        if df.empty:
            raise ValueError("Dataframe is empty")
        
        # Check for missing values
        missing = df[[date_col, target_col]].isnull().sum()
        if missing.any():
            print(f"Warning: Found missing values:\n{missing}")
            df = df.dropna(subset=[date_col, target_col])
            
        # Ensure date format
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)
        
        # Check for duplicates
        if df[date_col].duplicated().any():
            print("Warning: Duplicate timestamps found. Aggregating by sum.")
            df = df.groupby(date_col).agg({target_col: 'sum'}).reset_index()
            
        return df

    @staticmethod
    def handle_outliers(df: pd.DataFrame, target_col: str, method='isolation_forest'):
        """Detects and treats outliers."""
        if method == 'isolation_forest':
            iso = IsolationForest(contamination=0.05, random_state=42)
            outliers = iso.fit_predict(df[[target_col]])
            df = df[outliers == 1]
        elif method == 'winsorization':
            lower = df[target_col].quantile(0.05)
            upper = df[target_col].quantile(0.95)
            df[target_col] = df[target_col].clip(lower, upper)
        return df

    @staticmethod
    def feature_engineering(df: pd.DataFrame, target_col: str):
        """Generates lag features, rolling averages, and growth rates."""
        # Rolling averages
        df['rolling_mean_7'] = df[target_col].rolling(window=7).mean()
        df['rolling_mean_30'] = df[target_col].rolling(window=30).mean()
        
        # Lags
        df['lag_1'] = df[target_col].shift(1)
        df['lag_7'] = df[target_col].shift(7)
        
        # Growth Rate
        df['growth_rate'] = df[target_col].pct_change()
        
        # Time features
        df['day_of_week'] = df.index.dayofweek if isinstance(df.index, pd.DatetimeIndex) else 0
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        return df.fillna(0)

class DataGenerator:
    """Generates complex synthetic data for testing."""
    
    @staticmethod
    def generate_sales_data(periods=365, freq='D', noise_level=0.1, trend=0.05):
        dates = pd.date_range(start='2023-01-01', periods=periods, freq=freq)
        
        # Base trend
        x = np.arange(periods)
        y = 100 + trend * x
        
        # Weekly seasonality
        y += 20 * np.sin(2 * np.pi * x / 7)
        
        # Yearly seasonality
        y += 50 * np.sin(2 * np.pi * x / 365)
        
        # Noise
        y += np.random.normal(0, noise_level * np.mean(y), periods)
        
        # Add a sudden trend shift
        if periods > 200:
            y[200:] += 100 
            
        return pd.DataFrame({'ds': dates, 'y': y})
