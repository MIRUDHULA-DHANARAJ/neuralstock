"""
Data Generator Module for NeuralStock.
Generates a realistic, synthetic e-commerce inventory demand dataset
with seasonality, trends, promotional impacts, and multiple SKUs.
"""

import os
import numpy as np
import pandas as pd


def generate_synthetic_data(
    num_products: int = 20,
    days: int = 450,
    output_path: str = "../data/raw/ecommerce_inventory_demand.csv"
):
    """
    Generates synthetic time-series dataset for e-commerce demand forecasting.
    
    Args:
        num_products (int): Number of unique products (SKUs).
        days (int): Number of days to simulate.
        output_path (str): Path to save the generated CSV.
    """
    np.random.seed(42)
    
    
    categories = ['Electronics', 'Apparel', 'Home', 'Beauty', 'Sports']
    
    start_date = pd.to_datetime("2025-01-01")
    date_range = pd.date_range(start=start_date, periods=days, freq='D')
    
    all_data = []
    
    for i in range(num_products):
        product_id = f"P{i+1:03d}"
        category = np.random.choice(categories)
        
        # Product-specific base parameters
        base_demand = np.random.uniform(40, 250)
        base_price = np.random.uniform(299, 4500)
        lead_days = int(np.random.randint(3, 15))
        trend_slope = np.random.uniform(-0.0003, 0.0005)  # Slight upward/downward trend
        
        for idx, date in enumerate(date_range):
            day_of_week = date.dayofweek
            month = date.month
            day_of_year = date.dayofyear
            
            # ==================== Demand Components ====================
            # Weekly seasonality
            weekly_factor = 1.35 if day_of_week >= 5 else 0.92
            
            # Annual seasonality 
            yearly_factor = 1.0
            if month in [10, 11, 12]:      # Diwali + Year end
                yearly_factor = 1.55
            elif month in [1, 2]:
                yearly_factor = 0.75
            
            # Promotion effect
            is_promotion = 1 if np.random.rand() < 0.08 else 0
            discount_pct = np.random.uniform(0.15, 0.45) if is_promotion else 0.0
            promo_factor = 1.0 + (discount_pct * 2.2) if is_promotion else 1.0
            
            # Long-term trend
            trend_factor = 1 + trend_slope * idx
            
            # Final demand with noise
            noise = np.random.normal(0, max(8, base_demand * 0.08))
            units_sold = base_demand * weekly_factor * yearly_factor * promo_factor * trend_factor + noise
            units_sold = max(0, int(round(units_sold)))
            
            # Random missing values (for ffill practice)
            if np.random.rand() < 0.008:
                units_sold = np.nan
            
            # ==================== Other Columns ====================
            unit_price = round(base_price * (1 - discount_pct), 2)
            
            # Inject price outlier
            if idx == 80 and i % 3 == 0:
                unit_price *= 4.5
                
            # Stock simulation (Handles NaN safely)
            if pd.isna(units_sold):
                stock_on_hand = np.nan
            else:
                stock_on_hand = int(units_sold * np.random.uniform(1.1, 2.8))
                
            if idx == 120 and i % 4 == 0:        # Negative stock anomaly
                stock_on_hand = -30
                
            reorder_point = int(base_demand * 0.4 * lead_days)
            
            all_data.append({
                'date': date,
                'product_id': product_id,
                'product_category': category,
                'units_sold': units_sold,
                'unit_price': unit_price,
                'stock_on_hand': stock_on_hand,
                'reorder_point': reorder_point,
                'is_promotion': is_promotion,
                'discount_pct': round(discount_pct, 2),
                'day_of_week': day_of_week,
                'month': month,
                'supplier_lead_days': lead_days
            })
    
    df = pd.DataFrame(all_data)
    
    # Create directory 
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"   Synthetic dataset generated successfully!")
    print(f"   Shape: {df.shape}")
    print(f"   Products: {df['product_id'].nunique()}")
    print(f"   Date Range: {df['date'].min()} to {df['date'].max()}")
    print(f"   Saved to: {output_path}")


if __name__ == "__main__":
    generate_synthetic_data(num_products=20, days=450)
