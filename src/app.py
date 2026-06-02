import os
import pickle
import numpy as np
import pandas as pd
import torch
import streamlit as st
import matplotlib.pyplot as plt
from model import LSTMModel

# ====================== APP CONFIG & STYLING ======================
st.set_page_config(
    page_title="NeuralStock Enterprise Portal", 
    layout="wide", 
    page_icon="⚡",
    initial_sidebar_state="expanded"
)

st.html("""
    <style>
        .main-header { font-size: 2.2rem !important; font-weight: 700 !important; color: #0F172A; margin-bottom: 0.2rem; }
        .sub-header { font-size: 1.05rem !important; color: #64748B; margin-bottom: 2rem; }
        .metric-card { background-color: #F8FAFC; border: 1px solid #E2E8F0; padding: 1.25rem; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
        div[data-testid="stMetricValue"] { font-size: 1.8rem !important; font-weight: 700 !important; }
        .stTabs [data-baseweb="tab-list"] { gap: 12px; }
        .stTabs [data-baseweb="tab"] { background-color: #F1F5F9; border-radius: 8px 8px 0px 0px; padding: 10px 20px; font-weight: 600; color: #475569; }
        .stTabs [aria-selected="true"] { background-color: #1E293B !important; color: white !important; }
    </style>
""")

# ====================== UTILITY SERIALIZATION ASSET LOADERS ======================
@st.cache_resource
def load_production_assets():
    try:
        if os.path.exists('models/feature_columns.pkl'):
            base_path = 'models'
            data_path = 'data/processed/featured_data.csv'
        elif os.path.exists('../models/feature_columns.pkl'):
            base_path = '../models'
            data_path = '../data/processed/featured_data.csv'
        else:
            base_path = '/mount/src/neuralstock/models'
            data_path = '/mount/src/neuralstock/data/processed/featured_data.csv'

        with open(os.path.join(base_path, 'feature_columns.pkl'), 'rb') as f:
            feature_cols = pickle.load(f)
        with open(os.path.join(base_path, 'scalers_dict.pkl'), 'rb') as f:
            scalers = pickle.load(f)
            
        model = LSTMModel(input_size=len(feature_cols))
        model.load_state_dict(torch.load(os.path.join(base_path, 'lstm_model.pt'), map_location=torch.device('cpu')))
        model.eval()
        
        if os.path.exists(data_path):
            df = pd.read_csv(data_path, parse_dates=['date'])
        else:
            st.warning("⚠️ Accessing absolute runtime array fallback node...")
            df = pd.DataFrame(columns=['date', 'product_id', 'units_sold', 'stock_on_hand', 'reorder_point', 'is_promotion', 'discount_pct'] + feature_cols)
            
        return model, feature_cols, scalers, df
    except Exception as e:
        st.error(f"⚠️ App initialization assets missing. Error: {e}")
        return None, None, None, None

model, feature_cols, scalers, df_feat = load_production_assets()

# ====================== INTERACTIVE SIDEBAR CONTROL PANELS ======================
st.sidebar.markdown("### 🖥️ Core Control Unit")

if df_feat is not None:
    available_categories = ['Electronics', 'Apparel', 'Home', 'Beauty', 'Sports']
    selected_category = st.sidebar.selectbox("🎯 Select Product Category", available_categories, index=0)
    
    cat_prefix = f"cat_{selected_category}"
    cat_filtered_df = df_feat[df_feat[cat_prefix] == 1].sort_values('date').reset_index(drop=True)

# ====================== APPLICATION PORTAL BODY ======================
st.markdown("<div class='main-header'>NeuralStock Supply Chain Core</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Enterprise Deep Learning Engine for Inventory Optimization & Predictive Procurement Monitoring</div>", unsafe_allow_html=True)

if df_feat is None:
    st.stop()

# ====================== MIDDLE MAIN BODY DATE PARAMETERS ======================
st.markdown("### 📅 Forecast Date Parameters")

dataset_min_date = cat_filtered_df['date'].min()
dataset_max_date = cat_filtered_df['date'].max()

default_start = dataset_max_date + pd.Timedelta(days=1)
default_end = default_start + pd.Timedelta(weeks=4)

date_range = st.date_input(
    "Select Forecast Date Horizon Range",
    value=(default_start, default_end),
    min_value=dataset_min_date
)

# Parse date horizons dynamically
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_selection, end_selection = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    days_delta = (end_selection - start_selection).days
    forecast_weeks = max(1, int(np.ceil(days_delta / 7)))
else:
    forecast_weeks = 4

# ====================== FORECAST INFERENCE WRAPPER ======================
future_predictions = []
current_stock = 0
reorder_point = 0
total_demand_horizon = 0

if cat_filtered_df.empty:
    st.error(f"⚠️ No active data inventory records found for category: '{selected_category}'")
    forecast_df = pd.DataFrame(columns=['Delivery Horizon Target', 'Category Projected Sales (Units)'])
else:
    # 1. Isolate historical data for an active item track
    sample_product_id = cat_filtered_df['product_id'].iloc[0]
    prod_df = cat_filtered_df[cat_filtered_df['product_id'] == sample_product_id].sort_values('date').reset_index(drop=True)

    lookback_seq = 14
    latest_history = prod_df.tail(lookback_seq).copy()
    
    # Fill structural columns if they are missing from raw frames
    if 'prev_day_sales' not in latest_history.columns:
        latest_history['prev_day_sales'] = latest_history['units_sold'].shift(1).bfill()
    
    p_scale = scalers.get(sample_product_id, {'min': 0, 'denom': 100})
    current_history_df = latest_history.copy()

    # 2. Sequential autoregressive loop
    with torch.no_grad():
        for w in range(forecast_weeks):
            
            scaled_window_df = current_history_df.copy()
            for col in feature_cols:
                if col in scaled_window_df.columns and col in ['prev_day_sales', 'lag_7', 'lag_14', 'rolling_mean_7', 'rolling_mean_30']:
                    scaled_window_df[col] = (scaled_window_df[col] - p_scale['min']) / p_scale['denom']
            
            # Handle empty/missing features safely
            for col in feature_cols:
                if col not in scaled_window_df.columns:
                    scaled_window_df[col] = 0.0
                    
            X_input = scaled_window_df[feature_cols].values[-lookback_seq:] 
            X_tensor = torch.FloatTensor(X_input).unsqueeze(0)
            
            # Run model prediction step
            pred_scaled = model(X_tensor).item()
            pred_actual = (pred_scaled * p_scale['denom']) + p_scale['min']
            
            # Calculate total weekly values using formula match rules
            weekly_volume = max(0, int(pred_actual * 7 * 3.5)) 
            future_predictions.append(weekly_volume)
            
            new_row = current_history_df.iloc[-1].copy()
            new_row['date'] = current_history_df['date'].iloc[-1] + pd.Timedelta(weeks=1)
            
            
            new_row['units_sold'] = pred_actual
            new_row['prev_day_sales'] = pred_actual
            new_row['lag_7'] = current_history_df['units_sold'].iloc[-7] if len(current_history_df) >= 7 else pred_actual
            new_row['lag_14'] = current_history_df['units_sold'].iloc[-14] if len(current_history_df) >= 14 else pred_actual
            
            all_sales_so_far = current_history_df['units_sold'].tolist() + [pred_actual]
            new_row['rolling_mean_7'] = np.mean(all_sales_so_far[-7:])
            new_row['rolling_mean_30'] = np.mean(all_sales_so_far[-30:]) if len(all_sales_so_far) >= 30 else np.mean(all_sales_so_far)

            # Re-merge back into the running history dataframe cleanly
            current_history_df = pd.concat([current_history_df, pd.DataFrame([new_row])], ignore_index=True)

    # 4. Hook summary cards back directly to future prediction values
    current_stock = int(cat_filtered_df.groupby('product_id')['stock_on_hand'].last().sum())
    reorder_point = int(cat_filtered_df.groupby('product_id')['reorder_point'].last().sum())
    
    total_demand_horizon = sum(future_predictions)

    future_dates = [dataset_max_date + pd.Timedelta(weeks=w+1) for w in range(len(future_predictions))]
    forecast_df = pd.DataFrame({
        'Delivery Horizon Target': [d.strftime('%Y-%m-%d') for d in future_dates],
        'Category Projected Sales (Units)': future_predictions
    })

# ====================== TABBED INTERFACE ARCHITECTURE ======================
tab_workspace, tab_logistics, tab_mlops = st.tabs([
    "🔮 Forecast Workspace", 
    "🚛 Inventory Logistics", 
    "📊 MLOps Performance Center"
])

with tab_workspace:
    st.markdown(f"### ⚡ Category Matrix Planning: {selected_category}")
    
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f"<div class='metric-card'><b>📦 Category On-Hand Stock</b><br><span style='font-size:1.8rem; font-weight:700; color:#1E293B;'>{current_stock} Units</span></div>", unsafe_allow_html=True)
    with m2:
        st.markdown(f"<div class='metric-card'><b>📊 Selected Range Demand</b><br><span style='font-size:1.8rem; font-weight:700; color:#0284C7;'>{total_demand_horizon} Units</span></div>", unsafe_allow_html=True)
    with m3:
        st.markdown(f"<div class='metric-card'><b>🚨 Combined Category ROP</b><br><span style='font-size:1.8rem; font-weight:700; color:#E11D48;'>{reorder_point} Units</span></div>", unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_plot, col_ledger = st.columns([3, 2], gap="large")
    with col_plot:
        st.markdown("##### 📈 Weekly Projected Demand vs. Available Inventory")
        if not forecast_df.empty:
            fig, ax = plt.subplots(figsize=(10, 4.5), facecolor='#FFFFFF')
            ax.set_facecolor('#F8FAFC')
            
            # 1. Plot the AI's Forecast Track as a clean Bar Chart
            # We use an alpha gradient fill to make it blend with your enterprise theme
            bars = ax.bar(forecast_df['Delivery Horizon Target'], forecast_df['Category Projected Sales (Units)'], 
                          color='#0284C7', alpha=0.85, width=0.4, edgecolor='#0369A1', linewidth=1,
                          label="LSTM Weekly Required Volume")
            
            # 2. Add structural value metrics on top of each bar for immediate scanning
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{int(height)}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8, fontweight='bold', color='#334155')
            
            # 3. Current Warehouse Stock Reference Line (Green Dashboard Accent)
            ax.axhline(y=current_stock, color='#10B981', linestyle='--', linewidth=2.5, 
                       label=f"Current Available Stock ({current_stock} Units)")
            
            # 4. Reorder Point Threshold Line (Red Alert Accent)
            ax.axhline(y=reorder_point, color='#E11D48', linestyle=':', linewidth=2, 
                       label=f"Danger ROP Floor ({reorder_point} Units)")
            
            # Aesthetics, Grids, and Label Formatting
            ax.set_title(f"Procurement Balance Matrix - {selected_category}", fontsize=11, fontweight='bold', color='#1E293B')
            ax.set_ylabel("Volume Capacity (Units)", fontsize=9, fontweight='semibold', color='#475569')
            ax.tick_params(colors='#64748B', labelsize=9)
            
            # Fix x-axis layout rotation for clear date viewing
            plt.xticks(rotation=25, ha='right')
            ax.grid(True, linestyle=':', color='#E2E8F0', alpha=0.7)
            
            # Dynamically push the chart's upper ceiling limit up by 25% so annotations don't cut off
            max_y = max(max(future_predictions) if future_predictions else 100, current_stock, reorder_point)
            ax.set_ylim(0, max_y * 1.25)
            
            # Clean positioning for the legend box
            ax.legend(loc='upper right', frameon=True, facecolor='#FFFFFF', edgecolor='#E2E8F0')
            
            # Remove harsh black borders to match clean dashboard UI
            for spine in ['top', 'right', 'left', 'bottom']:
                ax.spines[spine].set_color('#E2E8F0')
                
            plt.tight_layout()
            st.pyplot(fig)
    
    
        
    with col_ledger:
        st.markdown("##### 📥 CSV Download Button Ledger")
        st.dataframe(forecast_df, use_container_width=True, hide_index=True)
        
        csv_bin = forecast_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Procurement Order CSV",
            data=csv_bin,
            file_name=f"procurement_{selected_category.lower()}_horizon.csv",
            mime="text/csv",
            use_container_width=True
        )

# ----------------- TAB 2: INVENTORY LOGISTICS (FIXED) -----------------
with tab_logistics:
    st.markdown("### 🚦 Reorder Alert Table & Exception Tracking")
    st.markdown("Automated category analysis mapping stock parameters against generated neural timelines.")
    
    alerts = []
    for pid, group in cat_filtered_df.groupby('product_id'):
        last_row = group.sort_values('date').iloc[-1]
        stock = int(last_row['stock_on_hand'])
        rop = int(last_row['reorder_point'])
        
        # 🛡️ CLEAN LOGIC: Clear, explicit safety checks independent of the UI calendar range
        if stock == 0:
            status = "🔴 Stockout Deficit"
        elif stock <= rop:
            status = "🔴 Critical Deficit"
        elif stock <= (rop * 1.25):  # Within 25% of hitting the reorder point
            status = "🟡 Trigger Restock"
        else:
            status = "🟢 Stable"
            
        alerts.append({
            "Product SKU": pid,
            "On-Hand Stock": stock,
            "Reorder Point Threshold": rop,
            "Risk Assessment Status": status
        })
        
    if alerts:
        alert_df = pd.DataFrame(alerts)
        
        # Apply custom style mapping colors matching our clean status strings
        def style_status_rows(val):
            if "🔴" in str(val):
                return 'color: #E11D48; font-weight: bold;'
            elif "🟡" in str(val):
                return 'color: #D97706; font-weight: bold;'
            return 'color: #16A34A;'

        st.dataframe(
            alert_df.style.map(style_status_rows, subset=["Risk Assessment Status"]),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("ℹ️ No active inventory items found to analyze for this category.")

# ----------------- TAB 3: MLOPS PERFORMANCE CENTER -----------------
with tab_mlops:
    st.markdown("### 🏆 Model Performance Evaluation Dashboard")

    sm1, sm2, sm3, sm4 = st.columns(4)

    with sm1:
        st.metric(
            label="🎯 LSTM R² Score",
            value="0.8661",
            delta="Best Model"
        )

    with sm2:
        st.metric(
            label="📊 LSTM MAPE",
            value="19.48%"
        )

    with sm3:
        st.metric(
            label="📉 LSTM MAE",
            value="10.55"
        )

    with sm4:
        st.metric(
            label="📉 LSTM RMSE",
            value="14.87"
        )

    st.markdown("---")

    st.markdown("#### 📊 Model Comparison Scorecard")

    scorecard = pd.DataFrame({
        "Metric": ["MAE", "RMSE", "MAPE (%)", "R² Score"],
        "LSTM Model": [
            "10.55",
            "14.87",
            "19.48%",
            "0.8661"
        ],
        "MLP Model": [
            "11.93",
            "17.25",
            "21.79%",
            "0.8200"
        ]
    })

    st.dataframe(
        scorecard,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("#### 📝 Model Selection Summary")

    st.success(
        """
        LSTM was selected as the final production model because it achieved
        the lowest forecasting error (MAE and RMSE) and the highest R² score.
        Its ability to learn temporal patterns from historical inventory data
        made it more suitable than the baseline MLP model.
        """
    )