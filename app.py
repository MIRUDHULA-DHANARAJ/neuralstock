import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
import matplotlib.pyplot as plt

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="NeuralStock Dashboard",
    layout="wide"
)

st.title("📦 NeuralStock - Inventory Demand Forecasting")
st.write("LSTM-based E-Commerce Demand Forecast Dashboard")

# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv("data/processed/test_engineered.csv")

# OPTIONAL DATE COLUMN
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"])

# LOAD SCALER
with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# =========================================================
# FEATURE COLUMNS
# =========================================================

cols_to_scale = [
    'units_sold',
    'unit_price',
    'stock_on_hand',
    'reorder_point',
    'discount_pct',
    'supplier_lead_days',
    'lag_7',
    'lag_14',
    'rolling_mean_7',
    'rolling_std_7',
    'rolling_mean_30',
    'rolling_std_30'
]

feature_cols = [
    'unit_price',
    'stock_on_hand',
    'reorder_point',
    'is_promotion',
    'discount_pct',
    'day_of_week',
    'month',
    'supplier_lead_days',
    'Beauty',
    'Electronics',
    'Home',
    'Sports',
    'Apparel',
    'lag_7',
    'lag_14',
    'rolling_mean_7',
    'rolling_std_7',
    'rolling_mean_30',
    'rolling_std_30',
    'is_weekend',
    'quarter',
    'quarter_sin',
    'quarter_cos'
]

# =========================================================
# LSTM MODEL
# =========================================================

class LSTMModel(nn.Module):

    def __init__(self):

        super(LSTMModel, self).__init__()

        self.lstm = nn.LSTM(
            input_size=23,
            hidden_size=64,
            num_layers=2,
            batch_first=True
        )

        self.fc = nn.Linear(64, 1)

    def forward(self, x):

        lstm_out, _ = self.lstm(x)

        last_out = lstm_out[:, -1, :]

        out = self.fc(last_out)

        return out

# =========================================================
# LOAD MODEL
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = LSTMModel().to(device)

model.load_state_dict(
    torch.load(
        "models/lstm_model.pt",
        map_location=device
    )
)

model.eval()

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("Dashboard Controls")

categories = [
    "Beauty",
    "Electronics",
    "Home",
    "Sports",
    "Apparel"
]

selected_category = st.sidebar.selectbox(
    "Select Product Category",
    categories
)

# =========================================================
# DATE RANGE
# =========================================================

if "date" in df.columns:

    min_date = df["date"].min()
    max_date = df["date"].max()

    date_range = st.sidebar.date_input(
        "Select Date Range",
        [min_date, max_date]
    )

else:

    date_range = None

# =========================================================
# CATEGORY COUNTS
# =========================================================

st.subheader("Available Records")

counts = df[
    ["Beauty", "Electronics", "Home", "Sports", "Apparel"]
].sum()

st.dataframe(counts)

# =========================================================
# FILTER CATEGORY
# =========================================================

filtered_df = df[
    df[selected_category] == 1
].copy()

# =========================================================
# FILTER DATE
# =========================================================

if date_range is not None and len(date_range) == 2:

    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[1])

    filtered_df = filtered_df[
        (filtered_df["date"] >= start_date) &
        (filtered_df["date"] <= end_date)
    ]

# =========================================================
# EMPTY CHECK
# =========================================================

if filtered_df.empty:

    st.error(
        f"No data available for {selected_category}"
    )

    st.stop()

st.success(
    f"Total records found: {len(filtered_df)}"
)

# =========================================================
# CREATE SEQUENCES
# =========================================================

window_size = 14

X = filtered_df[feature_cols]
y = filtered_df["units_sold"]

actual = []
predicted = []

# =========================================================
# INFERENCE
# =========================================================

with torch.no_grad():

    for i in range(len(X) - window_size):

        x_seq = X.iloc[
            i:i+window_size
        ].values

        y_true = y.iloc[
            i+window_size
        ]

        x_tensor = torch.FloatTensor(
            x_seq
        ).unsqueeze(0).to(device)

        pred = model(x_tensor)

        actual.append(y_true)

        predicted.append(
            pred.cpu().numpy()[0][0]
        )

# =========================================================
# CONVERT TO NUMPY
# =========================================================

actual = np.array(actual)
predicted = np.array(predicted)

# =========================================================
# INVERSE TRANSFORM
# =========================================================

dummy_actual = np.zeros(
    (len(actual), len(cols_to_scale))
)

dummy_pred = np.zeros(
    (len(predicted), len(cols_to_scale))
)

dummy_actual[:, 0] = actual
dummy_pred[:, 0] = predicted

actual_real = scaler.inverse_transform(
    dummy_actual
)[:, 0]

pred_real = scaler.inverse_transform(
    dummy_pred
)[:, 0]

# =========================================================
# METRICS
# =========================================================

mae = np.mean(
    np.abs(actual_real - pred_real)
)

rmse = np.sqrt(
    np.mean(
        (actual_real - pred_real) ** 2
    )
)

mape = np.mean(
    np.abs(
        (actual_real - pred_real)
        /
        (actual_real + 1e-8)
    )
) * 100

# =========================================================
# SHOW METRICS
# =========================================================

st.subheader("📊 Evaluation Metrics")

col1, col2, col3 = st.columns(3)

col1.metric(
    "MAE",
    f"{mae:.2f}"
)

col2.metric(
    "RMSE",
    f"{rmse:.2f}"
)

col3.metric(
    "MAPE",
    f"{mape:.2f}%"
)

# =========================================================
# ACTUAL VS PREDICTED GRAPH
# =========================================================

st.subheader("📈 Actual vs Predicted Demand")

fig, ax = plt.subplots(
    figsize=(14, 6)
)

ax.plot(
    actual_real[:100],
    label="Actual Demand"
)

ax.plot(
    pred_real[:100],
    label="Predicted Demand"
)

ax.set_title(
    f"{selected_category} Demand Forecast"
)

ax.set_xlabel("Samples")

ax.set_ylabel("Units Sold")

ax.legend()

st.pyplot(fig)

# =========================================================
# REORDER ALERTS
# =========================================================

st.subheader("⚠️ Reorder Alerts")

alert_df = filtered_df.iloc[
    window_size:
].copy()

alert_df["Predicted_Demand"] = pred_real

alerts = alert_df[
    alert_df["Predicted_Demand"]
    >
    alert_df["stock_on_hand"]
]

if len(alerts) > 0:

    st.dataframe(
        alerts[
            [
                "product_id",
                "stock_on_hand",
                "Predicted_Demand"
            ]
        ].head(10)
    )

else:

    st.success(
        "No reorder alerts"
    )

# =========================================================
# DOWNLOAD CSV
# =========================================================

st.subheader("⬇️ Download Forecast Results")

result_df = pd.DataFrame({

    "Actual_Demand": actual_real,
    "Predicted_Demand": pred_real

})

csv = result_df.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="Download CSV",
    data=csv,
    file_name="forecast_results.csv",
    mime="text/csv"
)