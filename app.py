import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
import matplotlib.pyplot as plt
import os

# =========================================================
# CONFIG
# =========================================================

st.set_page_config(
    page_title="NeuralStock Dashboard",
    layout="wide"
)

st.title("📦 NeuralStock - Inventory Demand Forecasting")
st.write("LSTM-based E-Commerce Demand Forecast Dashboard")

# =========================================================
# SAFE PATH HANDLING (PRODUCTION FIX)
# =========================================================

BASE_DIR = os.path.dirname(__file__)

DATA_PATH = os.path.join(BASE_DIR, "data/processed/test_engineered.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models/lstm_model.pt")
SCALER_PATH = os.path.join(BASE_DIR, "models/scaler.pkl")

# =========================================================
# LOAD DATA (SAFE)
# =========================================================

@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        st.error(f"Missing dataset: {DATA_PATH}")
        st.stop()
    df = pd.read_csv(DATA_PATH)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df

df = load_data()

# =========================================================
# LOAD SCALER (CACHED)
# =========================================================

@st.cache_resource
def load_scaler():
    with open(SCALER_PATH, "rb") as f:
        return pickle.load(f)

scaler = load_scaler()

# =========================================================
# FEATURE COLUMNS
# =========================================================

feature_cols = [
    'unit_price', 'stock_on_hand', 'reorder_point',
    'is_promotion', 'discount_pct', 'day_of_week',
    'month', 'supplier_lead_days',
    'Beauty', 'Electronics', 'Home', 'Sports', 'Apparel',
    'lag_7', 'lag_14',
    'rolling_mean_7', 'rolling_std_7',
    'rolling_mean_30', 'rolling_std_30',
    'is_weekend', 'quarter', 'quarter_sin', 'quarter_cos'
]

cols_to_scale = [
    'units_sold', 'unit_price', 'stock_on_hand',
    'reorder_point', 'discount_pct', 'supplier_lead_days',
    'lag_7', 'lag_14',
    'rolling_mean_7', 'rolling_std_7',
    'rolling_mean_30', 'rolling_std_30'
]

# =========================================================
# MODEL
# =========================================================

class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=len(feature_cols),
            hidden_size=64,
            num_layers=2,
            batch_first=True
        )
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# =========================================================
# LOAD MODEL (CACHED)
# =========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    model = LSTMModel().to(device)

    if not os.path.exists(MODEL_PATH):
        st.error(f"Missing model: {MODEL_PATH}")
        st.stop()

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device)
    )
    model.eval()
    return model

model = load_model()

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("Dashboard Controls")

categories = ["Beauty", "Electronics", "Home", "Sports", "Apparel"]

selected_category = st.sidebar.selectbox(
    "Select Product Category",
    categories
)

# =========================================================
# DATE FILTER
# =========================================================

if "date" in df.columns:
    min_date, max_date = df["date"].min(), df["date"].max()

    date_range = st.sidebar.date_input(
        "Select Date Range",
        [min_date, max_date]
    )
else:
    date_range = None

# =========================================================
# FILTER DATA
# =========================================================

filtered_df = df[df[selected_category] == 1].copy()

if date_range and len(date_range) == 2:
    filtered_df = filtered_df[
        (filtered_df["date"] >= pd.to_datetime(date_range[0])) &
        (filtered_df["date"] <= pd.to_datetime(date_range[1]))
    ]

if filtered_df.empty:
    st.error("No data for selected filters")
    st.stop()

st.success(f"Records found: {len(filtered_df)}")

# =========================================================
# INFERENCE OPTIMIZED (VECTOR-FRIENDLY)
# =========================================================

window_size = 14

X = filtered_df[feature_cols].values
y = filtered_df["units_sold"].values

actual, predicted = [], []

with torch.no_grad():
    for i in range(len(X) - window_size):

        x_seq = torch.tensor(
            X[i:i+window_size],
            dtype=torch.float32
        ).unsqueeze(0).to(device)

        pred = model(x_seq).cpu().numpy()[0][0]

        actual.append(y[i + window_size])
        predicted.append(pred)

actual = np.array(actual)
predicted = np.array(predicted)

# =========================================================
# INVERSE SCALE
# =========================================================

dummy_a = np.zeros((len(actual), len(cols_to_scale)))
dummy_p = np.zeros((len(predicted), len(cols_to_scale)))

dummy_a[:, 0] = actual
dummy_p[:, 0] = predicted

actual_real = scaler.inverse_transform(dummy_a)[:, 0]
pred_real = scaler.inverse_transform(dummy_p)[:, 0]

# =========================================================
# METRICS
# =========================================================

mae = np.mean(np.abs(actual_real - pred_real))
rmse = np.sqrt(np.mean((actual_real - pred_real) ** 2))
mape = np.mean(np.abs((actual_real - pred_real) / (actual_real + 1e-8))) * 100

st.subheader("📊 Metrics")

c1, c2, c3 = st.columns(3)
c1.metric("MAE", f"{mae:.2f}")
c2.metric("RMSE", f"{rmse:.2f}")
c3.metric("MAPE", f"{mape:.2f}%")

# =========================================================
# VISUALIZATION
# =========================================================

st.subheader("📈 Forecast vs Actual")

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(actual_real[:100], label="Actual")
ax.plot(pred_real[:100], label="Predicted")
ax.legend()
st.pyplot(fig)

# =========================================================
# REORDER ALERTS
# =========================================================

st.subheader("⚠️ Reorder Alerts")

alert_df = filtered_df.iloc[window_size:].copy()
alert_df["Predicted_Demand"] = pred_real

alerts = alert_df[
    alert_df["Predicted_Demand"] > alert_df["stock_on_hand"]
]

if not alerts.empty:
    st.dataframe(alerts[["product_id", "stock_on_hand", "Predicted_Demand"]].head(10))
else:
    st.success("No reorder alerts")

# =========================================================
# DOWNLOAD
# =========================================================

st.subheader("⬇️ Download Results")

result_df = pd.DataFrame({
    "Actual": actual_real,
    "Predicted": pred_real
})

st.download_button(
    "Download CSV",
    result_df.to_csv(index=False).encode("utf-8"),
    "forecast.csv",
    "text/csv"
)