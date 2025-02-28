import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader

class PriceDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

class LSTMPredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=3):
        super(LSTMPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0)) 
        out = self.fc(out[:, -1, :])
        return out

class PricePredictorPyTorch:
    def __init__(self, sequence_length=10):
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler(feature_range=(0, 1))  # Adjusted for trend awareness
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def create_sequences(self, data):
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])
            y.append(data[i + self.sequence_length])
        return np.array(X), np.array(y)
    
    def prepare_data(self, df, price_column, batch_size=32):
        price_data = df[price_column].copy().ffill().bfill()
        price_values = price_data.values.reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(price_values)
        
        X, y = self.create_sequences(scaled_data)
        
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        train_dataset = PriceDataset(X_train, y_train)
        test_dataset = PriceDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        return train_loader, test_loader, (X_train, y_train), (X_test, y_test)
    
    def train(self, df, price_column, epochs=100, batch_size=32, learning_rate=0.0005):
        try:
            train_loader, test_loader, train_data, test_data = self.prepare_data(df, price_column, batch_size)
            
            self.model = LSTMPredictor(input_size=1, hidden_size=64, num_layers=3).to(self.device)
            
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
            
            best_val_loss = float('inf')
            patience = 10
            patience_counter = 0
            
            for epoch in range(epochs):
                self.model.train()
                train_losses = []
                
                for sequences, targets in train_loader:
                    sequences, targets = sequences.to(self.device), targets.to(self.device)
                    optimizer.zero_grad()
                    outputs = self.model(sequences)
                    loss = criterion(outputs, targets)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    loss.backward()
                    optimizer.step()
                    train_losses.append(loss.item())
                
                self.model.eval()
                val_losses = []
                with torch.no_grad():
                    for sequences, targets in test_loader:
                        sequences, targets = sequences.to(self.device), targets.to(self.device)
                        outputs = self.model(sequences)
                        val_loss = criterion(outputs, targets)
                        val_losses.append(val_loss.item())
                
                avg_train_loss = np.mean(train_losses)
                avg_val_loss = np.mean(val_losses)

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            return {
                'train_rmse': np.sqrt(avg_train_loss),
                'test_rmse': np.sqrt(avg_val_loss),
                'status': 'success'
            }
            
        except Exception as e:
            return {'status': 'failed', 'error': str(e)}
    
    def predict_future(self, df, price_column, days_ahead=30):
        try:
            self.model.eval()
            price_data = df[price_column].copy().ffill().bfill()
            scaled_data = self.scaler.transform(price_data.values.reshape(-1, 1))
            last_sequence = scaled_data[-self.sequence_length:]
            current_sequence = torch.FloatTensor(last_sequence).unsqueeze(0).to(self.device)
            future_predictions = []
            
            with torch.no_grad():
                for _ in range(days_ahead):
                    prediction = self.model(current_sequence).cpu().numpy()[0]
                    prediction = np.clip(prediction, 0, 1)  # Keep within scaling limits
                    future_predictions.append(prediction)
                    current_sequence = torch.cat([current_sequence[:, 1:, :], 
                                                  torch.FloatTensor([[prediction]]).to(self.device)], dim=1)
            
            future_predictions = np.array(future_predictions).reshape(-1, 1)
            future_prices = self.scaler.inverse_transform(future_predictions)
            
            # ✅ Fix: Apply moving average for downward trend
            window_size = 7
            smoothed_prices = np.convolve(future_prices.flatten(), np.ones(window_size)/window_size, mode='valid')
            smoothed_prices = np.clip(smoothed_prices, np.min(price_data) * 0.9, np.max(price_data) * 1.1)
            
            return smoothed_prices
        
        except Exception as e:
            return None

def main():
    df = pd.read_csv('Price_Data/iphone12_all_models_prices.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    predictor = PricePredictorPyTorch(sequence_length=10)
    price_columns = [col for col in df.columns if 'price' in col]
    
    for price_col in price_columns:
        print(f"\nProcessing {price_col}")
        valid_data_points = df[price_col].notna().sum()
        
        if valid_data_points >= 10:
            print("Training model...")
            results = predictor.train(df, price_col)
            if results['status'] == 'success':
                print(f"Training RMSE: ${results['train_rmse']:.2f}")
                print(f"Testing RMSE: ${results['test_rmse']:.2f}")
                future_prices = predictor.predict_future(df, price_col, days_ahead=30)
                print("\nNext 30 days price predictions:", future_prices)

if __name__ == "__main__":
    main()
