"""
Feature Engineering Module for NeuralStock.
Generates historical lags, leakage-free rolling windows, and cyclical dates.
"""

import os
import numpy as np
import pandas as pd

CLEANED_DATA_PATH = "../data/processed/cleaned_data.csv"
FEATURED_DATA_PATH = "../data/processed/featured_data.csv"

def construct_features(input_path: str = CLEANED_DATA_PATH, output_path: str = FEATURED_DATA_PATH):
    """Calculates rolling history windows and cyclical calendar transforms from clean data."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Cleaned data not found at {input_path}. Please run preprocess.py first.")

    df = pd.read_csv(input_path, parse_dates=['date'])
    df = df.sort_values(['product_id', 'date']).reset_index(drop=True)
    
    # 1. Base Time Features
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # 2. Shifted Lags (
    grouped = df.groupby('product_id')
    df['lag_7'] = grouped['units_sold'].shift(7)
    df['lag_14'] = grouped['units_sold'].shift(14)
    
    shifted_sales = grouped['units_sold'].shift(1)
    shifted_grouped = shifted_sales.groupby(df['product_id'])
    
    df['rolling_mean_7'] = shifted_grouped.transform(lambda x: x.rolling(7, min_periods=1).mean())
    df['rolling_std_7'] = shifted_grouped.transform(lambda x: x.rolling(7, min_periods=1).std())
    df['rolling_mean_30'] = shifted_grouped.transform(lambda x: x.rolling(30, min_periods=1).mean())
    
    df['rolling_std_7'] = df.groupby('product_id')['rolling_std_7'].bfill()
    fill_cols = ['lag_7', 'lag_14', 'rolling_mean_7', 'rolling_std_7', 'rolling_mean_30']
    df[fill_cols] = df[fill_cols].fillna(0.0)
    
    # Interaction Features
    df['promo_impact'] = df['is_promotion'] * df['discount_pct']
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f" Feature Engineering complete. Saved to: {output_path}")
    print(f"   Shape: {df.shape}")

if __name__ == "__main__":
    construct_features()
