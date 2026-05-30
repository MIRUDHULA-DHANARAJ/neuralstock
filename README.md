# ⚡ NeuralStock

Deep Learning-powered inventory demand forecasting system for e-commerce supply chains.

## Overview

NeuralStock predicts future inventory demand using a stacked LSTM network with attention pooling, helping businesses reduce stockouts, optimize procurement, and improve inventory planning.

Built as an Advanced Deep Learning Capstone Project using PyTorch and Streamlit.

### Key Highlights

* 📈 Demand forecasting using Stacked LSTM + Attention
* 🛒 Multi-category e-commerce inventory prediction
* 📊 Interactive Streamlit dashboard
* ⚠️ Automated reorder risk detection
* 📥 Procurement-ready CSV exports
* 🧠 Temporal feature engineering pipeline (39 features)

---

## Tech Stack

* Python 3.12
* PyTorch
* Streamlit
* Pandas
* NumPy
* Scikit-Learn
* Matplotlib
* TensorBoard

---

## Dataset

* 6,223 inventory records
* 50 SKUs
* 5 product categories
* 2 years of transaction history

Categories:

* Electronics
* Apparel
* Home
* Sports
* Beauty

---

## Model Architecture

### Primary Model

* 2-Layer Stacked LSTM
* Attention Pooling
* Layer Normalization
* Dropout Regularization
* Fully Connected Forecast Head

### Input Configuration

* 14-Day Lookback Window
* 39 Engineered Features
* Weekly Demand Forecasting

---

## Results

| Metric   | Score  |
| -------- | ------ |
| MAE      | 8.42   |
| RMSE     | 11.15  |
| MAPE     | 9.52%  |
| R² Score | 0.8942 |

✅ Achieved all project performance targets.

---

## Dashboard Features

* Category-wise demand forecasting
* Forecast horizon planning
* Inventory monitoring
* Reorder point alerts
* KPI analytics
* CSV export functionality

---

## Project Structure

```bash
NeuralStock/
│
├── data/
├── models/
├── src/
│   ├── app.py
│   ├── model.py
│   ├── train.py
│   └── preprocessing.py
│
├── requirements.txt
└── README.md
```

---

## Run Locally

```bash
git clone <your-repo-url>

cd NeuralStock

pip install -r requirements.txt

streamlit run src/app.py
```

---

## Live Demo

https://neuralstock-forecast-app.streamlit.app/

---

## Author

Mirudhula Dhanaraj

B.Tech Artificial Intelligence & Data Science

Advanced Deep Learning Capstone Project
