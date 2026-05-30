"""
Preprocessing Module for NeuralStock.
Cleans missing inputs, clips invalid data anomalies, and tracks category bounds.
"""

import os
import pandas as pd
import numpy as np

RAW_DATA_PATH = "../data/raw/ecommerce_inventory_demand.csv" 
PROCESSED_DATA_PATH = "../data/processed/cleaned_data.csv"

def clean_and_prepare_data(input_path: str = RAW_DATA_PATH, output_path: str = PROCESSED_DATA_PATH):
    """Loads, cleans, and handles multi-category outlier containment for real dataset."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Missing raw data file at {input_path}. Please place your downloaded CSV there.")
        
    df = pd.read_csv(input_path, parse_dates=['date'])
    df = df.sort_values(['product_id', 'date']).reset_index(drop=True)
    
    if df['is_promotion'].dtype == 'bool':
        df['is_promotion'] = df['is_promotion'].astype(int)
    
    # (Handles random missing rows)
    df['units_sold'] = df.groupby('product_id')['units_sold'].ffill().bfill().fillna(0.0)
    df['stock_on_hand'] = df.groupby('product_id')['stock_on_hand'].ffill().bfill().fillna(0.0)
    
    # Clear negative inventory anomalies 
    df['stock_on_hand'] = df['stock_on_hand'].clip(lower=0)
    
    # Segmented Outlier Capping per Category 
    for cat in df['product_category'].unique():
        mask = df['product_category'] == cat
        q1 = df.loc[mask, 'unit_price'].quantile(0.25)
        q3 = df.loc[mask, 'unit_price'].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        df.loc[mask, 'unit_price'] = df.loc[mask, 'unit_price'].clip(lower=lower_bound, upper=upper_bound)
        
    # Standardise categorical frames 
    categories_list = ['Electronics', 'Apparel', 'Home', 'Beauty', 'Sports']
    df['product_category'] = pd.Categorical(df['product_category'], categories=categories_list)
    cat_dummies = pd.get_dummies(df['product_category'], prefix='cat', dtype=int)
    
    df = pd.concat([df.drop('product_category', axis=1), cat_dummies], axis=1)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f" Preprocessed dataset saved successfully at: {output_path}")
    print(f"   Shape: {df.shape} | Products Handled: {df['product_id'].nunique()}")

if __name__ == "__main__":
    clean_and_prepare_data()
