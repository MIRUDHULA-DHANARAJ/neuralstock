# 📦 NeuralStock: Deep Learning for E-commerce Inventory Demand Forecasting

Deep learning-based weekly demand forecasting using LSTM and MLP networks, deployed via Streamlit.

---

## 📁 Folder Structure

```
NeuralStock/
├── data/
│   ├── sales_data.csv          # Generated raw dataset
│   └── processed_data.csv      # Feature-engineered dataset
├── models/
│   ├── lstm_model.pt           # Trained LSTM weights
│   ├── mlp_model.pt            # Trained MLP weights
│   ├── scaler.pkl              # Fitted MinMaxScaler
│   └── metrics.json            # Evaluation metrics
├── runs/                       # TensorBoard logs
├── data_generator.py           # Synthetic dataset generation
├── preprocess.py               # Cleaning, feature engineering, scaling
├── model.py                    # LSTM and MLP model definitions
├── train.py                    # Training loop with TensorBoard
├── inference.py                # Inference pipeline
├── app.py                      # Streamlit dashboard
├── NeuralStock_Analysis.ipynb  # Full EDA + modelling notebook
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

```bash
# 1. Clone the repo
git clone https://github.com/your-username/NeuralStock.git
cd NeuralStock

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate dataset
python data_generator.py

# 4. Train models
python train.py

# 5. Run Streamlit app
streamlit run app.py
```

---

## 🎯 Target Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| MAPE   | ≤ 12%  | Mean Absolute Percentage Error |
| RMSE   | ≤ 15   | Root Mean Squared Error |
| MAE    | ≤ 10   | Mean Absolute Error |
| R²     | ≥ 0.85 | Coefficient of Determination |

---

## 📊 TensorBoard

```bash
tensorboard --logdir=runs
```

Open http://localhost:6006 to view training/validation loss curves.

---

## ☁️ Cloud Deployment (AWS EC2)

```bash
# On EC2 instance (Ubuntu)
pip install -r requirements.txt
python data_generator.py && python train.py
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

Access at: `http://<EC2-PUBLIC-IP>:8501`

---

## 📝 Notes

- Train/test split is strictly **chronological** (no data leakage)
- MinMaxScaler is fit **only on training data**
- All random seeds set to 42 for reproducibility
- Sensitive keys (cloud credentials) stored in `.env` — never committed
