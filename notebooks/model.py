import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
from sklearn.metrics import r2_score




with open('../models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
train = pd.read_csv('../data/processed/train_engineered.csv')
test = pd.read_csv('../data/processed/test_engineered.csv')
cols_to_scale=['units_sold',
'unit_price',
'stock_on_hand',
'reorder_point',
'discount_pct',
'supplier_lead_days',
'lag_7', 'lag_14',
'rolling_mean_7', 'rolling_std_7',
'rolling_mean_30', 'rolling_std_30']
feature_cols = ['unit_price','stock_on_hand','reorder_point','is_promotion','discount_pct','day_of_week', 'month','supplier_lead_days','Beauty','Electronics','Home','Sports','lag_7','lag_14','rolling_mean_7','rolling_std_7','rolling_mean_30','rolling_std_30','is_weekend','quarter','quarter_sin','quarter_cos']

X_train = train[feature_cols]
y_train = train['units_sold']

X_test = test[feature_cols]
y_test = test['units_sold']
class InventoryDataset(Dataset):
    def __init__(self, X, y,window_size=14):
        self.X = X
        self.y = y
        self.window_size = window_size

    def __len__(self):
        return len(self.X) - self.window_size

    def __getitem__(self, idx):
        x = self.X.iloc[idx:idx+self.window_size].values
        y = self.y.iloc[idx+self.window_size]
            

       

        return torch.FloatTensor(x), torch.FloatTensor([y])

        
train_dataset = InventoryDataset(X_train, y_train)
test_dataset = InventoryDataset(X_test, y_test)


train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=22, hidden_size=64, num_layers=2, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        out = self.fc(last_out)
        return out
model = LSTMModel()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 50

for epoch in range(epochs):

    model.train()

    epoch_loss = 0

    for X_batch, y_batch in train_loader:

        # Move data to GPU/CPU
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # Forward pass
        predictions = model(X_batch)

        # Calculate loss
        loss = criterion(predictions, y_batch)

        # Clear old gradients
        optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        # Store batch loss
        epoch_loss += loss.item()

    # Average loss for entire epoch
    avg_loss = epoch_loss / len(train_loader)

    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
model.eval()

test_loss = 0

with torch.no_grad():

    for X_batch, y_batch in test_loader:

        # Move data to GPU/CPU
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # Predictions
        predictions = model(X_batch)

        # Calculate loss
        loss = criterion(predictions, y_batch)

        # Add batch loss
        test_loss += loss.item()

# Average test loss
avg_test_loss = test_loss / len(test_loader)

print(f"Test Loss: {avg_test_loss:.4f}")
torch.save(model.state_dict(), '../models/lstm_model.pt')

print("Model Saved Successfully!")
class MLPModel(nn.Module):

    def __init__(self):

        super(MLPModel, self).__init__()

        self.fc1 = nn.Linear(22, 128)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):

        x = self.fc1(x)
        x = self.relu1(x)

        x = self.fc2(x)
        x = self.relu2(x)

        x = self.fc3(x)

        return x
class MLPDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.values
        self.y = y.values

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.X[idx]),
            torch.FloatTensor([self.y[idx]])
        )
lstm_test_dataset = InventoryDataset(X_test, y_test)
lstm_test_loader = DataLoader(lstm_test_dataset, batch_size=64, shuffle=False)
mlp_train_dataset = MLPDataset(X_train, y_train)
mlp_test_dataset = MLPDataset(X_test, y_test)
mlp_test_loader = DataLoader(mlp_test_dataset, batch_size=64, shuffle=False)
mlp_loader = DataLoader(mlp_train_dataset, batch_size=64, shuffle=False)

mlp_model = MLPModel().to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(mlp_model.parameters(), lr=0.001)

epochs = 50

for epoch in range(epochs):

    mlp_model.train()
    epoch_loss = 0

    for X_batch, y_batch in mlp_loader:

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        predictions = mlp_model(X_batch)

        loss = criterion(predictions, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(mlp_loader)

    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
def evaluate(model, loader, is_lstm=False):

    model.eval()

    y_true, y_pred = [], []

    with torch.no_grad():

        for X_batch, y_batch in loader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            if is_lstm:
                preds = model(X_batch)
            else:
                if len(X_batch.shape) == 3:
                    X_batch = X_batch[:, -1, :]
                preds = model(X_batch)

            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    n_features = len(cols_to_scale)
    target_idx = cols_to_scale.index("units_sold")

    dummy_true = np.zeros((len(y_true), n_features))
    dummy_pred = np.zeros((len(y_pred), n_features))

    dummy_true[:, target_idx] = y_true
    dummy_pred[:, target_idx] = y_pred

    y_true_real = scaler.inverse_transform(dummy_true)[:, target_idx]
    y_pred_real = scaler.inverse_transform(dummy_pred)[:, target_idx]

    mae = mean_absolute_error(y_true_real, y_pred_real)
    rmse = np.sqrt(mean_squared_error(y_true_real, y_pred_real))
    mape = np.mean(np.abs((y_true_real - y_pred_real) / (y_true_real + 1e-8))) * 100
    r2 = r2_score(y_true_real, y_pred_real)

    return mae, rmse, mape, r2
lstm_mae, lstm_rmse, lstm_mape, lstm_r2 = evaluate(model, lstm_test_loader, is_lstm=True)

mlp_mae, mlp_rmse, mlp_mape, mlp_r2 = evaluate(mlp_model, mlp_test_loader, is_lstm=False)
results = pd.DataFrame({
    "Model": ["LSTM", "MLP"],
    "MAE": [lstm_mae, mlp_mae],
    "RMSE": [lstm_rmse, mlp_rmse],
    "MAPE": [lstm_mape, mlp_mape],
    "R2": [lstm_r2, mlp_r2]
})

print(results)