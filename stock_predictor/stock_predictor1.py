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

class StockPredictor(nn.Module):
    def __init__(self, input_size=10, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.linear(lstm_out[:, -1, :])

def prepare_data(symbol='AAPL', period='10y', seq_length=60):
    data = yf.download(symbol, period=period)

    # Создаем фичи
    data['Returns'] = data['Close'].pct_change()
    data['EMA_10'] = data['Close'].ewm(span=10).mean()  #span=10  5
    data['EMA_50'] = data['Close'].ewm(span=50).mean()  #span=50  15
    data['Volatility'] = data['Returns'].rolling(20).std()
    data['Volume_EMA'] = data['Volume'].ewm(span=20).mean()

    data = data.dropna()

    features = ['Open', 'High', 'Low', 'Close', 'Volume',
                'Returns', 'EMA_10', 'EMA_50', 'Volatility', 'Volume_EMA']

    # РАЗДЕЛЯЕМ ДАННЫЕ ДО МАСШТАБИРОВАНИЯ
    total_size = len(data)
    train_size = int(total_size * 0.7)
    val_size = int(total_size * 0.15)

    train_data = data.iloc[:train_size]
    val_data = data.iloc[train_size:train_size+val_size]
    test_data = data.iloc[train_size+val_size:]

    # МАСШТАБИРУЕМ КАЖДЫЙ НАБОР ОТДЕЛЬНО
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train_data[features])
    scaled_val = scaler.transform(val_data[features])
    scaled_test = scaler.transform(test_data[features])

    # СОЗДАЕМ ПОСЛЕДОВАТЕЛЬНОСТИ ДЛЯ КАЖДОГО НАБОРА
    def create_sequences(scaled_data, dates_data, seq_length):
        X, y, dates = [], [], []
        for i in range(seq_length, len(scaled_data)-1):
            X.append(scaled_data[i-seq_length:i])
            y.append(scaled_data[i+1, 3])  # Close цену
            dates.append(dates_data.index[i+1])
        return torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(y)), dates

    X_train, y_train, dates_train = create_sequences(scaled_train, train_data, seq_length)
    X_val, y_val, dates_val = create_sequences(scaled_val, val_data, seq_length)
    X_test, y_test, dates_test = create_sequences(scaled_test, test_data, seq_length)

    return (X_train, X_val, X_test, y_train, y_val, y_test,
            dates_train, dates_val, dates_test, scaler, data)

def split_data(X, y, dates, train_ratio=0.7, val_ratio=0.15):
    """Разделяем данные 70:15:15"""
    total_size = len(X)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)

    X_train, X_val, X_test = X[:train_size], X[train_size:train_size+val_size], X[train_size+val_size:]
    y_train, y_val, y_test = y[:train_size], y[train_size:train_size+val_size], y[train_size+val_size:]
    dates_train, dates_val, dates_test = dates[:train_size], dates[train_size:train_size+val_size], dates[train_size+val_size:]

    print(f"Разделение данных:")
    print(f"  Обучающие: {len(X_train)} ({len(X_train)/total_size*100:.1f}%)")
    print(f"  Валидационные: {len(X_val)} ({len(X_val)/total_size*100:.1f}%)")
    print(f"  Тестовые: {len(X_test)} ({len(X_test)/total_size*100:.1f}%)")

    return (X_train, X_val, X_test, y_train, y_val, y_test, dates_train, dates_val, dates_test)

def train_model(model, X_train, y_train, X_val, y_val, epochs=100):
    """Обучаем модель с выводом после каждой эпохи"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Перемещаем данные на GPU
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_val, y_val = X_val.to(device), y_val.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.MSELoss()

    train_losses, val_losses = [], []

    print("\nНачинаем обучение...")
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs.squeeze(), y_train)
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs.squeeze(), y_val)

        train_losses.append(loss.item())
        val_losses.append(val_loss.item())

        # Выводим после каждой эпохи
        print(f'Epoch [{epoch+1:3d}/{epochs}] | Train Loss: {loss.item():.6f} | Val Loss: {val_loss.item():.6f}')

    return train_losses, val_losses

def plot_results(model, X_test, y_test, dates_test, scaler, original_data, features):
    """Строим график на тестовых данных с правильным масштабированием"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    with torch.no_grad():
        predictions = model(X_test.to(device)).cpu().numpy()

    # Восстанавливаем масштаб ПРАВИЛЬНО
    # Нужно восстановить всю последовательность, а не только Close
    real_prices = []
    pred_prices = []

    for i in range(len(X_test)):
        # Берем последний элемент последовательности как основу
        last_sequence = X_test[i][-1:].numpy()  # [1, 10]

        # Заменяем Close цену на предсказанное значение
        last_sequence[0, 3] = predictions[i]  # Close цена

        # Обратное преобразование
        restored = scaler.inverse_transform(last_sequence)
        pred_prices.append(restored[0, 3])  # Восстановленная Close цена

        # Реальная цена (уже знаем)
        real_seq = X_test[i][-1:].numpy()
        real_seq[0, 3] = y_test[i].item()
        real_restored = scaler.inverse_transform(real_seq)
        real_prices.append(real_restored[0, 3])

    # Строим график
    plt.figure(figsize=(15, 8))

    plt.plot(dates_test, real_prices, label='Реальная цена', linewidth=2, alpha=0.7)
    plt.plot(dates_test, pred_prices, label='Предсказание', linewidth=1.5, alpha=0.9)

    plt.title('AAPL: Реальная цена vs Предсказание (Тестовые данные)', fontsize=14)
    plt.xlabel('Дата')
    plt.ylabel('Цена ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('test_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Вычисляем метрики
    real_prices = np.array(real_prices)
    pred_prices = np.array(pred_prices)

    mse = np.mean((real_prices - pred_prices) ** 2)
    mape = np.mean(np.abs((real_prices - pred_prices) / real_prices)) * 100

    print(f"\n=== РЕЗУЛЬТАТЫ НА ТЕСТОВЫХ ДАННЫХ ===")
    print(f"Среднеквадратичная ошибка (MSE): ${mse:.2f}")
    print(f"Средняя абсолютная процентная ошибка (MAPE): {mape:.2f}%")
    print(f"Пример реальных цен: {real_prices[:5]}")
    print(f"Пример предсказаний: {pred_prices[:5]}")
    print(f"График сохранен как 'test_predictions.png'")

# Основной код
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Устройство: {device}")

    # Загружаем данные за 10 лет
    # Разделяем данные 70:15:15
    (X_train, X_val, X_test, y_train, y_val, y_test, dates_train, dates_val, dates_test, scaler, data) = prepare_data('AAPL', period='10y')

    print(f"\nОбщий размер данных X_train: {X_train.shape}")
    print(f"\nОбщий размер данных X_test: {X_test.shape}")
    print(f"Период данных: {data.index[0].strftime('%Y-%m-%d')} - {data.index[-1].strftime('%Y-%m-%d')}")



    # Создаем и обучаем модель
    model = StockPredictor(input_size=10, hidden_size=128).to(device)

    # Обучаем с выводом после каждой эпохи
    train_losses, val_losses = train_model(model, X_train, y_train, X_val, y_val, epochs=100)

    # Строим график на тестовых данных

    features_list = ['Open', 'High', 'Low', 'Close', 'Volume',
                'Returns', 'EMA_10', 'EMA_50', 'Volatility', 'Volume_EMA']

    plot_results(model, X_test, y_test, dates_test, scaler, data, features_list)

    print("\n🎯 Обучение завершено!")
