"""
inference.py
Reusable inference pipeline: loads trained model and scaler,
generates demand forecasts from new input data in under 2 seconds.
"""

import pickle
import numpy as np
import pandas as pd
import torch

from model import LSTMModel
from preprocess import load_data, handle_missing, add_features, get_feature_columns

SEQ_LEN = 14
N_FEATURES = 21
MODEL_PATH = "models/lstm_model.pt"
SCALER_PATH = "models/scaler.pkl"


def load_model(model_path=MODEL_PATH, n_features=N_FEATURES):
    """
    Load trained LSTM model from disk.

    Args:
        model_path (str): Path to the .pt weights file.
        n_features (int): Number of input features.

    Returns:
        LSTMModel: Loaded model in eval mode.
    """
    model = LSTMModel(input_size=n_features, hidden_size=128, num_layers=2)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


def load_scaler(scaler_path=SCALER_PATH):
    """
    Load fitted MinMaxScaler from disk.

    Args:
        scaler_path (str): Path to the .pkl scaler file.

    Returns:
        MinMaxScaler: Fitted scaler.
    """
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    return scaler


def forecast_category(category, forecast_days=28, df_processed=None):
    """
    Generate demand forecast for a given product category.

    Args:
        category (str): Product category (e.g. 'Electronics').
        forecast_days (int): Number of days to forecast.
        df_processed (pd.DataFrame): Pre-processed dataframe. Loads from disk if None.

    Returns:
        pd.DataFrame: DataFrame with date and forecasted_demand columns.
    """
    model = load_model()
    scaler = load_scaler()

    if df_processed is None:
        df_raw = load_data()
        df_raw = handle_missing(df_raw)
        df_processed = add_features(df_raw)

    feature_cols = get_feature_columns()

    # Filter by category
    cat_df = df_processed[df_processed["product_category"] == category].copy()
    if cat_df.empty:
        raise ValueError(f"No data found for category: {category}")

    # Use last SEQ_LEN rows as seed window
    seed = cat_df[feature_cols].tail(SEQ_LEN).values.astype(np.float32)
    seed_scaled = scaler.transform(seed)

    # Generate weekly aggregated forecast
    predictions = []
    last_date = cat_df["date"].max()

    for i in range(forecast_days):
        x = torch.tensor(seed_scaled[-SEQ_LEN:]).unsqueeze(0)  # (1, SEQ_LEN, features)
        with torch.no_grad():
            pred = model(x).item()

        pred = max(0, pred)  # No negative demand
        predictions.append({
            "date": last_date + pd.Timedelta(days=i + 1),
            "forecasted_demand": round(pred, 1),
        })

        # Slide window: shift features forward (reuse last row with updated lag)
        new_row = seed_scaled[-1].copy()
        new_row[0] = seed_scaled[-7][0] if len(seed_scaled) >= 7 else seed_scaled[-1][0]  # lag_7
        new_row[1] = seed_scaled[-14][0] if len(seed_scaled) >= 14 else seed_scaled[-1][0]  # lag_14
        seed_scaled = np.vstack([seed_scaled, new_row])

    forecast_df = pd.DataFrame(predictions)
    # Aggregate to weekly
    forecast_df["week"] = forecast_df["date"].dt.to_period("W").apply(lambda r: r.start_time)
    weekly = forecast_df.groupby("week")["forecasted_demand"].sum().reset_index()
    weekly.columns = ["week_start", "weekly_demand_forecast"]
    weekly["weekly_demand_forecast"] = weekly["weekly_demand_forecast"].round(1)

    return weekly


def check_reorder_alert(category, current_stock, forecast_df):
    """
    Check if forecasted demand exceeds current stock for reorder alerts.

    Args:
        category (str): Product category name.
        current_stock (int): Current stock on hand.
        forecast_df (pd.DataFrame): Weekly forecast DataFrame.

    Returns:
        pd.DataFrame: Alert table with reorder flags.
    """
    forecast_df = forecast_df.copy()
    forecast_df["current_stock"] = current_stock
    forecast_df["reorder_alert"] = forecast_df["weekly_demand_forecast"] > current_stock
    forecast_df["category"] = category
    return forecast_df


if __name__ == "__main__":
    import time
    print("Running inference...")
    start = time.time()
    result = forecast_category("Electronics", forecast_days=28)
    elapsed = time.time() - start
    print(result)
    print(f"\nInference time: {elapsed:.2f}s")

    alerts = check_reorder_alert("Electronics", current_stock=300, forecast_df=result)
    print("\nReorder Alerts:")
    print(alerts)
