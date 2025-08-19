import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Union, Dict, List, Optional, Tuple
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute

logger = logging.getLogger(__name__)

class AdvancedDataProcessor:
    """Advanced data processor for time series forecasting."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.scalers = {}
        self.feature_columns = []
        
    def load_data(self, file_path: str, 
                 date_col: str = 'date', 
                 value_col: str = 'value') -> pd.DataFrame:
        """Load time series data with robust error handling."""
        try:
            df = pd.read_csv(file_path, parse_dates=[date_col])
            df = df.sort_values(date_col).reset_index(drop=True)
            
            # Check for missing dates and fill if needed
            df = self._fill_missing_dates(df, date_col)
            
            logger.info(f"Data loaded successfully with shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
            
    def _fill_missing_dates(self, df: pd.DataFrame, date_col: str) -> pd.DataFrame:
        """Fill missing dates in time series data."""
        full_date_range = pd.date_range(
            start=df[date_col].min(), 
            end=df[date_col].max(), 
            freq=self.config.get('data_frequency', 'D')
        )
        
        if len(full_date_range) != len(df[date_col].unique()):
            df_full = pd.DataFrame({date_col: full_date_range})
            df = df_full.merge(df, on=date_col, how='left')
            
            # Forward fill for time series continuity
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(method='ffill')
            
            logger.info(f"Filled {len(full_date_range) - len(df)} missing dates")
            
        return df
    
    def engineer_features(self, df: pd.DataFrame, 
                         date_col: str, 
                         value_col: str) -> pd.DataFrame:
        """Create comprehensive time series features."""
        df = df.copy()
        
        # DateTime features
        df['year'] = df[date_col].dt.year
        df['month'] = df[date_col].dt.month
        df['week'] = df[date_col].dt.isocalendar().week
        df['day'] = df[date_col].dt.day
        df['dayofweek'] = df[date_col].dt.dayofweek
        df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
        df['quarter'] = df[date_col].dt.quarter
        
        # Lag features
        for lag in self.config.get('lags', [1, 7, 30, 90]):
            df[f'lag_{lag}'] = df[value_col].shift(lag)
        
        # Rolling statistics
        for window in self.config.get('rolling_windows', [7, 30, 90]):
            df[f'rolling_mean_{window}'] = df[value_col].rolling(window=window).mean()
            df[f'rolling_std_{window}'] = df[value_col].rolling(window=window).std()
            df[f'rolling_min_{window}'] = df[value_col].rolling(window=window).min()
            df[f'rolling_max_{window}'] = df[value_col].rolling(window=window).max()
        
        # Difference features
        for diff in self.config.get('differences', [1, 7, 30]):
            df[f'diff_{diff}'] = df[value_col].diff(diff)
        
        # Seasonal features
        df['seasonal_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365)
        df['seasonal_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365)
        
        # Advanced features using tsfresh
        if self.config.get('use_tsfresh', False):
            df = self._add_tsfresh_features(df, value_col)
        
        # Handle missing values from feature engineering
        df = df.fillna(method='bfill').fillna(method='ffill')
        
        self.feature_columns = [col for col in df.columns if col not in [date_col, value_col]]
        
        logger.info(f"Created {len(self.feature_columns)} features")
        return df
    
    def _add_tsfresh_features(self, df: pd.DataFrame, value_col: str) -> pd.DataFrame:
        """Add automated features using tsfresh."""
        try:
            # Extract features
            extracted_features = extract_features(
                df.rename(columns={'date': 'time'}),
                column_id='id',  # You might need to create an ID column
                column_sort='time',
                column_value=value_col,
                impute_function=impute,
                n_jobs=self.config.get('n_jobs', -1)
            )
            
            # Select relevant features
            selected_features = select_features(extracted_features, df[value_col])
            
            # Merge back with original dataframe
            df = pd.concat([df, selected_features], axis=1)
            
        except Exception as e:
            logger.warning(f"TSFresh feature extraction failed: {str(e)}")
            
        return df
    
    def scale_features(self, df: pd.DataFrame, 
                      feature_cols: List[str],
                      fit: bool = True) -> Tuple[pd.DataFrame, dict]:
        """Scale features using appropriate scalers."""
        df_scaled = df.copy()
        scaling_info = {}
        
        for col in feature_cols:
            if fit:
                if self.config.get('scaling_method', 'standard') == 'minmax':
                    scaler = MinMaxScaler()
                else:
                    scaler = StandardScaler()
                
                df_scaled[col] = scaler.fit_transform(df[[col]])
                self.scalers[col] = scaler
                scaling_info[col] = {
                    'type': scaler.__class__.__name__,
                    'params': scaler.get_params()
                }
            else:
                if col in self.scalers:
                    df_scaled[col] = self.scalers[col].transform(df[[col]])
        
        return df_scaled, scaling_info
    
    def create_sequences(self, df: pd.DataFrame, 
                       target_col: str, 
                       feature_cols: List[str],
                       seq_length: int = 30,
                       forecast_horizon: int = 7) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for deep learning models."""
        X, y = [], []
        data = df[feature_cols + [target_col]].values
        
        for i in range(len(data) - seq_length - forecast_horizon + 1):
            X.append(data[i:i+seq_length, :-1])  # Features
            y.append(data[i+seq_length:i+seq_length+forecast_horizon, -1])  # Target
        
        return np.array(X), np.array(y)
    
    def prepare_train_test_split(self, df: pd.DataFrame, 
                               test_size: float = 0.2,
                               date_col: str = 'date') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Time-based train-test split."""
        split_idx = int(len(df) * (1 - test_size))
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        
        logger.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
        return train_df, test_df
