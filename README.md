# 📦 NeuralStock: Deep Learning for E-commerce Inventory Demand Forecasting

An end-to-end deep learning pipeline that predicts weekly inventory demand per product using LSTM and MLP neural networks built with PyTorch.

---

## 🧠 Problem Statement

An online retail company is losing revenue due to chronic stock imbalances — popular products go out of stock during peak demand while slow-moving items accumulate warehouse costs. This project builds a deep learning-based demand forecasting system to predict weekly inventory requirements per product.

---

## 📁 Folder Structure

```
NeuralStock/
│
├── data/
│   ├── raw/                  # Original dataset
│   └── processed/            # Cleaned and engineered data
│
├── notebooks/
│   └── NeuralStock_Analysis.ipynb
│
├── src/
│   ├── eda.py                # Exploratory Data Analysis
│   ├── preprocess.py         # Data cleaning and encoding
│   ├── feature_engineering.py# Lag, rolling, cyclical features
│   ├── model.py              # LSTM and MLP model training
│   └── evaluate.py           # Evaluation and visualization
│
├── models/
│   ├── lstm_model.pt         # Saved LSTM weights
│   └── scaler.pkl            # Fitted MinMaxScaler
│
├── app/
│   └── app.py                # Streamlit dashboard
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/MIRUDHULA-DHANARAJ/neuralstock.git
cd neuralstock
```

### 2. Create virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the pipeline in order
```bash
python src/preprocess.py
python src/feature_engineering.py
python src/model.py
```

### 5. Launch Streamlit dashboard
```bash
streamlit run app/app.py
```

---

## 🔬 Approach

| Step | Description |
|---|---|
| EDA | Sales distributions, seasonality plots, ACF/PACF, promotion impact |
| Preprocessing | Forward-fill, IQR clipping, one-hot encoding, cyclical encoding |
| Feature Engineering | Lag-7/14, rolling mean/std (7 & 30 day), is_weekend, quarter |
| Model Building | Stacked LSTM + MLP baseline in PyTorch with sliding window dataset |
| Evaluation | MAE, RMSE, MAPE, R² on chronological 80/20 split |
| Deployment | Streamlit dashboard with reorder alerts and CSV download |

---

## 📊 Model Results

| Model | MAE | RMSE | MAPE | R² |
|---|---|---|---|---|
| LSTM | 6.95 | 10.45 | 25.56% | 0.49 |
| MLP | 4.50 | 5.68 | 18.55% | 0.85 |

> MLP outperformed LSTM on this dataset, achieving MAE of 4.50 and R² of 0.85.
> MAPE remains above target due to high variance in low-volume products in the synthetic dataset.

---

## 🖥️ Streamlit Dashboard Features

- Category selector (Apparel, Beauty, Electronics, Home, Sports)
- Date range picker
- Actual vs Predicted demand chart
- Reorder alert table
- Downloadable CSV forecast results

---

## 🛠️ Technical Stack

Python · PyTorch · LSTM · Pandas · NumPy · Scikit-learn · Streamlit · Matplotlib · Seaborn · MinMaxScaler · TensorBoard · Pickle

---

## ⚠️ Limitations

- Cold-start problem: model cannot forecast demand for new SKUs with no history
- MAPE sensitive to low-volume products
- Trained on synthetic data — real-world performance may vary

---

## 👩‍💻 Author

Mirudhula Dhanaraj
