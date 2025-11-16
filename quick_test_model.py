import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class QuickImprovedStockPredictor(nn.Module):
    def __init__(self, input_size=20, hidden_size=256, num_layers=2, dropout=0.3):
        super().__init__()
        # Упрощенная LSTM для быстрого тестирования
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout, bidirectional=True)
        self.dropout = nn.Dropout(dropout)

        # Улучшенный механизм внимания
        self.attention_weights = nn.Linear(hidden_size * 2, 1)
        self.attention_softmax = nn.Softmax(dim=1)

        self.linear1 = nn.Linear(hidden_size * 2, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, hidden_size // 2)
        self.linear3 = nn.Linear(hidden_size // 2, 1)

    def forward(self, x):
        lstm_out, (hidden, _) = self.lstm(x)

        # Attention mechanism
        attention_scores = self.attention_weights(lstm_out)
        attention_weights = self.attention_softmax(attention_scores)
        attended_lstm_out = lstm_out * attention_weights
        context_vector = torch.sum(attended_lstm_out, dim=1)

        output = self.dropout(context_vector)
        output = self.linear1(output)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.linear2(output)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.linear3(output)
        return output

def prepare_simple_data(symbol='AAPL', period='2y', seq_length=30):
    """Упрощенная подготовка данных для быстрого теста"""
    data = yf.download(symbol, period=period)

    # Базовые фичи
    data['Returns'] = data['High'].pct_change()
    data['EMA_10'] = data['High'].ewm(span=10).mean()
    data['EMA_50'] = data['High'].ewm(span=50).mean()
    data['Volatility'] = data['Returns'].rolling(20).std()
    data['Volume_EMA'] = data['Volume'].ewm(span=20).mean()
    data['RSI'] = compute_rsi(data['High'])
    data['MACD'] = compute_macd(data['High'])

    # Упрощенные признаки
    data['High_Low_Pct'] = (data['High'] - data['Low']) / data['High']
    data['Price_Change'] = (data['High'] - data['Open']) / data['Open']
    data['Volume_Change'] = data['Volume'].pct_change()
    data['Momentum'] = data['High'] - data['High'].shift(10)
    data['SMA_20'] = data['High'].rolling(window=20).mean()
    sma_50 = data['High'].rolling(window=50).mean()
    ma_ratio = data['SMA_20'] / sma_50
    data = data.copy()  # Создаем копию чтобы избежать проблем с индексацией
    data['MA_Ratio'] = ma_ratio

    # Новые признаки
    data['Adaptive_Volatility'] = data['Returns'].rolling(20).std() / data['Returns'].rolling(50).std()
    data['Trend_Strength'] = (data['High'] - data['Low']).rolling(10).mean() / data['High'].rolling(50).mean()
    data['Price_Strength'] = (data['High'] - data['Low'].rolling(20).min()) / (data['High'].rolling(20).max() - data['Low'].rolling(20).min())
    vwap = (data['High'] * data['Volume']).rolling(20).sum() / data['Volume'].rolling(20).sum()
    data['VWAP'] = vwap
    data['Price_to_VWAP'] = data['High'] / vwap
    data['Volatility_Momentum'] = data['Volatility'] - data['Volatility'].shift(5)

    data = data.dropna()

    features = ['High', 'Volume', 'Returns', 'EMA_10', 'EMA_50',
                'Volatility', 'Volume_EMA', 'RSI', 'MACD',
                'High_Low_Pct', 'Price_Change', 'Volume_Change', 'Momentum', 'MA_Ratio',
                'Adaptive_Volatility', 'Trend_Strength', 'Price_Strength', 'Price_to_VWAP', 'Volatility_Momentum']

    # Разделение данных
    total_size = len(data)
    train_size = int(total_size * 0.8)
    test_size = total_size - train_size

    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]

    # Масштабирование
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train_data[features])
    scaled_test = scaler.transform(test_data[features])

    def create_sequences(scaled_data, original_data, seq_length):
        X, y = [], []
        for i in range(seq_length, len(scaled_data)-1):
            X.append(scaled_data[i-seq_length:i])
            price_change = (original_data['High'].iloc[i+1] - original_data['High'].iloc[i]) / original_data['High'].iloc[i]
            y.append(price_change)
        return torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(y))

    X_train, y_train = create_sequences(scaled_train, train_data, seq_length)
    X_test, y_test = create_sequences(scaled_test, test_data, seq_length)

    return X_train, X_test, y_train, y_test, scaler, test_data

def compute_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(prices, fast=12, slow=26, signal=9):
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal).mean()
    return macd - signal_line

def calculate_directional_accuracy(model, X_test, y_test):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    with torch.no_grad():
        predicted_changes = model(X_test.to(device)).cpu().numpy().flatten()

    real_changes = y_test.numpy().flatten()
    true_directions = np.sign(real_changes)
    pred_directions = np.sign(predicted_changes)

    mask = (true_directions != 0)
    if np.sum(mask) > 0:
        accuracy = np.mean(true_directions[mask] == pred_directions[mask])
        return accuracy
    return 0

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Устройство: {device}")

    print("Загрузка данных...")
    X_train, X_test, y_train, y_test, scaler, test_data = prepare_simple_data('AAPL', period='2y')

    print(f"Размеры данных:")
    print(f"  Обучающие: {X_train.shape}")
    print(f"  Тестовые: {X_test.shape}")

    # Создание модели
    model = QuickImprovedStockPredictor(input_size=20, hidden_size=256, num_layers=2, dropout=0.3).to(device)
    print(f"\nМодель создана: {sum(p.numel() for p in model.parameters()):,} параметров")

    # Обучение
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    print("\nНачинаем обучение...")
    for epoch in range(50):  # Короткое обучение для быстрого теста
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train.to(device))
        loss = criterion(outputs.squeeze(), y_train.to(device))
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1:2d}/50] | Loss: {loss.item():.6f}')

    # Оценка
    da = calculate_directional_accuracy(model, X_test, y_test)
    print(f"\n=== РЕЗУЛЬТАТЫ ===")
    print(f"Directional Accuracy: {da:.3f} ({da*100:.1f}%)")

    return da

if __name__ == "__main__":
    main()
