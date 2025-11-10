import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
import yfinance as yf
import ssl

# Отключаем SSL проверку (временно)

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

def prepare_data(symbol='AAPL', period='5y', seq_length=60):
    data = yf.download(symbol, period=period)

    # Создаем фичи
    data['Returns'] = data['Close'].pct_change()
    data['EMA_10'] = data['Close'].ewm(span=10).mean()
    data['EMA_50'] = data['Close'].ewm(span=50).mean()
    data['Volatility'] = data['Returns'].rolling(20).std()
    data['Volume_EMA'] = data['Volume'].ewm(span=20).mean()

    # Добавляем технические индикаторы
    data['RSI'] = compute_rsi(data['Close'])
    data['MACD'] = compute_macd(data['Close'])
    data['BB_Upper'], data['BB_Lower'] = compute_bollinger_bands(data['Close'])

    data['Price_Range'] = (data['High'] - data['Low']) / data['Close']
    data['Gap'] = (data['Open'] - data['Close'].shift(1)) / data['Close'].shift(1)

    data = data.dropna()

    features = ['Close', 'Volume', 'Returns', 'EMA_10', 'EMA_50',
                'Volatility', 'Volume_EMA', 'RSI', 'MACD', 'Price_Range']

    # Разделяем данные
    total_size = len(data)
    train_size = int(total_size * 0.7)
    val_size = int(total_size * 0.15)

    train_data = data.iloc[:train_size]
    val_data = data.iloc[train_size:train_size+val_size]
    test_data = data.iloc[train_size+val_size:]

    # Масштабируем
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train_data[features])
    scaled_val = scaler.transform(val_data[features])
    scaled_test = scaler.transform(test_data[features])

    def create_sequences(scaled_data, original_data, dates_data, seq_length):
        X, y, dates, base_prices = [], [], [], []
        for i in range(seq_length, len(scaled_data)-1):
            X.append(scaled_data[i-seq_length:i])
            # Предсказываем изменение цены (нормализованное)
            price_change = (original_data['Close'].iloc[i+1] - original_data['Close'].iloc[i]) / original_data['Close'].iloc[i]
            y.append(price_change)
            dates.append(dates_data.index[i+1])
            base_prices.append(original_data['Close'].iloc[i])
        return (torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(y)),
                dates, base_prices)

    X_train, y_train, dates_train, base_train = create_sequences(scaled_train, train_data, train_data, seq_length)
    X_val, y_val, dates_val, base_val = create_sequences(scaled_val, val_data, val_data, seq_length)
    X_test, y_test, dates_test, base_test = create_sequences(scaled_test, test_data, test_data, seq_length)

    return (X_train, X_val, X_test, y_train, y_val, y_test,
            dates_train, dates_val, dates_test, scaler, data,
            base_train, base_val, base_test)

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
    return macd

def compute_bollinger_bands(prices, window=20):
    sma = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()
    upper_band = sma + (std * 2)
    lower_band = sma - (std * 2)
    return upper_band, lower_band

def train_model_with_early_stopping(model, X_train, y_train, X_val, y_val, epochs=100, patience=15):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X_train, y_train = X_train.to(device), y_train.to(device)
    X_val, y_val = X_val.to(device), y_val.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5, verbose=True)
    criterion = nn.MSELoss()

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    print("\nНачинаем обучение с ранней остановкой...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs.squeeze(), y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs.squeeze(), y_val)

        scheduler.step(val_loss)

        train_losses.append(loss.item())
        val_losses.append(val_loss.item())

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            best_epoch = epoch + 1
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch [{epoch+1:3d}/{epochs}] | Train Loss: {loss.item():.6f} | Val Loss: {val_loss.item():.6f} | Patience: {patience_counter}/{patience}')

        if patience_counter >= patience:
            print(f"\nРанняя остановка на эпохе {epoch+1}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Восстановлены веса с эпохи {best_epoch} (val_loss: {best_val_loss:.6f})")

    return train_losses, val_losses

def plot_results_with_changes(model, X_test, y_test, dates_test, base_prices_test, original_data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    with torch.no_grad():
        predicted_changes = model(X_test.to(device)).cpu().numpy().flatten()

    # Восстанавливаем цены
    real_prices = []
    pred_prices = []
    valid_dates = []

    for i, (base_price, pred_change, date) in enumerate(zip(base_prices_test, predicted_changes, dates_test)):
        # Предсказанная цена = базовая цена * (1 + предсказанное изменение)
        pred_price = base_price * (1 + pred_change)
        pred_prices.append(pred_price)

        # Находим реальную цену для этой даты
        if date in original_data.index:
            real_price = original_data.loc[date, 'Close']
            real_prices.append(real_price)
            valid_dates.append(date)

    # Убедимся, что все массивы одинаковой длины
    real_prices = np.array(real_prices)
    pred_prices = np.array(pred_prices)

    print(f"Данные для графика: {len(real_prices)} точек")

    # Строим график
    plt.figure(figsize=(15, 10))

    # График 1: Цены
    plt.subplot(2, 1, 1)
    plt.plot(valid_dates, real_prices, label='Реальная цена', linewidth=2, alpha=0.7, color='blue')
    plt.plot(valid_dates, pred_prices, label='Предсказание', linewidth=1.5, alpha=0.9, color='red')
    plt.title('AAPL: Реальная цена vs Предсказание', fontsize=14)
    plt.ylabel('Цена ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    # График 2: Ошибки
    plt.subplot(2, 1, 2)
    errors = real_prices - pred_prices
    plt.plot(valid_dates, errors, label='Ошибка', linewidth=1, color='green', alpha=0.7)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.title('Ошибки предсказания')
    plt.ylabel('Ошибка ($)')
    plt.xlabel('Дата')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig('stock_predictions_fixed.png', dpi=300, bbox_inches='tight')
    plt.close()  # Закрываем график чтобы не занимал память
    print("График сохранен как 'stock_predictions_fixed.png'")

    # Метрики
    mse = np.mean(errors ** 2)
    mae = np.mean(np.abs(errors))
    mape = np.mean(np.abs(errors / real_prices)) * 100
    mean_error = np.mean(errors)
    std_error = np.std(errors)

    print(f"\n=== РЕЗУЛЬТАТЫ НА ТЕСТОВЫХ ДАННЫХ ===")
    print(f"Количество точек: {len(real_prices)}")
    print(f"MSE: ${mse:.2f}")
    print(f"MAE: ${mae:.2f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"Средняя ошибка: ${mean_error:.2f}")
    print(f"Стандартное отклонение ошибки: ${std_error:.2f}")

    # Анализ смещения
    if mean_error > 0:
        print(f"СМЕЩЕНИЕ: Модель занижает предсказания на ${mean_error:.2f} в среднем")
        print(f"Процент заниженных предсказаний: {np.mean(errors > 0) * 100:.1f}%")
    else:
        print(f"СМЕЩЕНИЕ: Модель завышает предсказания на ${-mean_error:.2f} в среднем")
        print(f"Процент завышенных предсказаний: {np.mean(errors < 0) * 100:.1f}%")

    print(f"\nПримеры предсказаний (первые 5):")
    for i in range(min(5, len(real_prices))):
        print(f"  {valid_dates[i].strftime('%Y-%m-%d')}: Реальная ${real_prices[i].item():.2f}, Предсказание ${pred_prices[i].item():.2f}, Ошибка ${errors[i].item():.2f}")

    return real_prices, pred_prices

# Основной код
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Устройство: {device}")
    ssl._create_default_https_context = ssl._create_unverified_context

    # Загружаем данные
    print("Загрузка данных...")
    (X_train, X_val, X_test, y_train, y_val, y_test,
     dates_train, dates_val, dates_test, scaler, data,
     base_train, base_val, base_test) = prepare_data('AAPL', period='5y')

    print(f"\nРазмеры данных:")
    print(f"  Обучающие: {X_train.shape} ({len(dates_train)} дат)")
    print(f"  Валидационные: {X_val.shape} ({len(dates_val)} дат)")
    print(f"  Тестовые: {X_test.shape} ({len(dates_test)} дат)")
    print(f"  Период данных: {data.index[0].strftime('%Y-%m-%d')} - {data.index[-1].strftime('%Y-%m-%d')}")

    # Создаем модель
    model = StockPredictor(input_size=10, hidden_size=128, num_layers=2, dropout=0.2).to(device)
    print(f"\nМодель создана: {sum(p.numel() for p in model.parameters()):,} параметров")

    # Обучаем
    train_losses, val_losses = train_model_with_early_stopping(
        model, X_train, y_train, X_val, y_val, epochs=100, patience=15
    )

    # Тестируем
    print("\n" + "="*50)
    print("ТЕСТИРОВАНИЕ НА ТЕСТОВЫХ ДАННЫХ")
    print("="*50)

    real_prices, pred_prices = plot_results_with_changes(
        model, X_test, y_test, dates_test, base_test, data
    )

    print("\n✅ Обучение и тестирование завершено!")
