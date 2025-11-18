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

class ImprovedStockPredictor(nn.Module):
    def __init__(self, input_size=15, hidden_size=768, num_layers=3, dropout=0.2):
        super().__init__()
        # Используем bidirectional LSTM с увеличенными параметрами
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout, bidirectional=True)
        # Увеличиваем размерность из-за bidirectional (hidden_size * 2)
        self.dropout = nn.Dropout(dropout)

        # Улучшенный механизм многоголового внимания
        self.multi_head_attention = nn.MultiheadAttention(
            hidden_size * 2, num_heads=8, dropout=dropout, batch_first=True
        )
        self.layer_norm1 = nn.LayerNorm(hidden_size * 2)

        # Дополнительные слои для лучшей репрезентации
        self.enhanced_layers = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Оригинальный механизм внимания как дополнительный
        self.attention_weights = nn.Linear(hidden_size // 2, 1)
        self.attention_softmax = nn.Softmax(dim=1)

        self.linear1 = nn.Linear(hidden_size // 2, hidden_size // 2)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.linear3 = nn.Linear(hidden_size // 4, 1)

    def forward(self, x):
        lstm_out, (hidden, _) = self.lstm(x)

        # Многоголовое внимание
        attended_out, attention_weights = self.multi_head_attention(lstm_out, lstm_out, lstm_out)
        # Residual connection и layer normalization
        lstm_out = self.layer_norm1(lstm_out + attended_out)

        # Улучшенные слои репрезентации
        enhanced_features = self.enhanced_layers(lstm_out)

        # Оригинальный механизм внимания на улучшенных признаках
        attention_scores = self.attention_weights(enhanced_features)
        attention_weights = self.attention_softmax(attention_scores)
        attended_lstm_out = enhanced_features * attention_weights
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

    def forward(self, x, return_attention=False):
        # Убедимся, что x имеет правильную размерность
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Добавляем размерность batch если она отсутствует

        lstm_out, (hidden, _) = self.lstm(x)

        # Многоголовое внимание
        attended_out, multi_attention_weights = self.multi_head_attention(lstm_out, lstm_out, lstm_out)
        lstm_out = self.layer_norm1(lstm_out + attended_out)

        # Улучшенные слои репрезентации
        enhanced_features = self.enhanced_layers(lstm_out)

        # Оригинальный механизм внимания на улучшенных признаках
        attention_scores = self.attention_weights(enhanced_features)
        attention_weights = self.attention_softmax(attention_scores)
        attended_lstm_out = enhanced_features * attention_weights
        context_vector = torch.sum(attended_lstm_out, dim=1)

        output = self.dropout(context_vector)
        output = self.linear1(output)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.linear2(output)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.linear3(output)

        if return_attention:
            return output, {
                'multi_head_attention': multi_attention_weights,
                'feature_attention': attention_weights,
                'enhanced_features': enhanced_features
            }
        return output

def prepare_data(symbol='AAPL', period='5y', seq_length=60):
    data = yf.download(symbol, period=period)

    # Создаем фичи
    data['Returns'] = data['High'].pct_change()
    data['EMA_10'] = data['High'].ewm(span=10).mean()
    data['EMA_50'] = data['High'].ewm(span=50).mean()
    data['Volatility'] = data['Returns'].rolling(20).std()
    data['Volume_EMA'] = data['Volume'].ewm(span=20).mean()

    # Добавляем технические индикаторы
    data['RSI'] = compute_rsi(data['High'])
    data['MACD'] = compute_macd(data['High'])
    bb_upper, bb_lower = compute_bollinger_bands(data['High'])
    data['BB_Upper'] = bb_upper
    data['BB_Lower'] = bb_lower

    # Дополнительные признаки
    data['High_Low_Pct'] = (data['High'] - data['Low']) / data['High']
    data['Price_Change'] = (data['High'] - data['Open']) / data['Open']
    data['Volume_Change'] = data['Volume'].pct_change()
    data['Momentum'] = data['High'] - data['High'].shift(10) # 10-дневный моментум

    # Средние скользящие как дополнительные фичи
    data['SMA_20'] = data['High'].rolling(window=20).mean()
    data['SMA_200'] = data['High'].rolling(window=200).mean()
    data['MA_Ratio'] = data['SMA_20'] / data['SMA_200']  # Отношение краткосрочной к долгосрочной MA

    # Позиция внутри Bollinger Bands (избегаем деления на 0)
    bb_range = data['BB_Upper'] - data['BB_Lower']
    bb_range = bb_range.replace(0, np.nan)  # Заменяем 0 на NaN, чтобы избежать деления на 0
    bb_position = (data['High'] - data['BB_Lower']) / bb_range
    bb_position = bb_position.fillna(0.5)  # Заполняем NaN значением 0.5 (середина диапазона)
    # Убедимся, что bb_position - это одномерная серия, а не DataFrame
    if isinstance(bb_position, pd.DataFrame):
        bb_position = bb_position.iloc[:, 0]  # Берем первый столбец
    data['BB_Position'] = bb_position

    # НОВЫЕ признаки для улучшения DA
    # Адаптивная волатильность
    data['Adaptive_Volatility'] = data['Returns'].rolling(20).std() / data['Returns'].rolling(50).std()

    # Сила тренда
    data['Trend_Strength'] = (data['High'] - data['Low']).rolling(10).mean() / data['High'].rolling(50).mean()

    # Относительная сила цены
    data['Price_Strength'] = (data['High'] - data['Low'].rolling(20).min()) / (data['High'].rolling(20).max() - data['Low'].rolling(20).min())

    # Объемно-взвешенная цена
    vwap = (data['High'] * data['Volume']).rolling(20).sum() / data['Volume'].rolling(20).sum()
    data['VWAP'] = vwap
    data['Price_to_VWAP'] = data['High'] / vwap

    # Моментум волатильности
    data['Volatility_Momentum'] = data['Volatility'] - data['Volatility'].shift(5)

    # Удаляем строки с NaN значениями только после всех вычислений
    data = data.dropna()

    features = ['High', 'Volume', 'Returns', 'EMA_10', 'EMA_50',
                'Volatility', 'Volume_EMA', 'RSI', 'MACD', 'BB_Position',
                'High_Low_Pct', 'Price_Change', 'Volume_Change', 'Momentum', 'MA_Ratio',
                'Adaptive_Volatility', 'Trend_Strength', 'Price_Strength', 'Price_to_VWAP', 'Volatility_Momentum']

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
            price_change = (original_data['High'].iloc[i+1] - original_data['High'].iloc[i]) / original_data['High'].iloc[i]
            y.append(price_change)
            dates.append(dates_data.index[i+1])
            base_prices.append(original_data['High'].iloc[i])
        return (torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(y)),
                dates, base_prices)

    X_train, y_train, dates_train, base_train = create_sequences(scaled_train, train_data, train_data, seq_length)
    X_val, y_val, dates_val, base_val = create_sequences(scaled_val, val_data, val_data, seq_length)
    X_test, y_test, dates_test, base_test = create_sequences(scaled_test, test_data, test_data, seq_length)

    # Анализ распределения изменений
    train_changes = y_train.numpy()
    print(f"\nРаспределение изменений в обучающих данных:")
    print(f"  Рост: {(train_changes > 0.001).mean():.1%}")
    print(f"  Падение: {(train_changes < -0.001).mean():.1%}")
    print(f" Без изменений: {(np.abs(train_changes) <= 0.001).mean():.1%}")

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
    signal_line = macd.ewm(span=signal).mean()
    return macd - signal_line  # MACD гистограмма

def compute_bollinger_bands(prices, window=20):
    sma = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()
    upper_band = sma + (std * 2)
    lower_band = sma - (std * 2)
    # Убедимся, что возвращаемые значения являются одномерными Series
    return pd.Series(upper_band.values.flatten(), index=upper_band.index), pd.Series(lower_band.values.flatten(), index=lower_band.index)

class DirectionalAccuracyLoss(nn.Module):
    def __init__(self, mse_weight=0.2, da_weight=0.7, confidence_weight=0.1):
        super().__init__()
        self.mse = nn.MSELoss()
        self.mse_weight = mse_weight
        self.da_weight = da_weight
        self.confidence_weight = confidence_weight

    def forward(self, pred, target):
        # MSE для предсказания величины изменения
        mse_loss = self.mse(pred.squeeze(), target)

        # Усиленная направленная потеря
        pred_direction = torch.sign(pred.squeeze())
        target_direction = torch.sign(target)

        # Динамический вес в зависимости от величины изменения
        magnitude_weight = torch.abs(target) + 0.1  # Избегаем нулевого веса
        directional_penalty = torch.mean(
            torch.relu(-pred_direction * target_direction) * magnitude_weight
        )

        # Штраф за неуверенные предсказания
        confidence_penalty = torch.mean(torch.abs(pred.squeeze() - target))

        # Дополнительный штраф за экстремальные ошибки
        extreme_errors = torch.abs(pred.squeeze() - target) > 0.05
        extreme_penalty = torch.mean(
            torch.relu(torch.abs(pred.squeeze() - target) - 0.05) * extreme_errors.float()
        ) * 2.0

        return (self.mse_weight * mse_loss +
                self.da_weight * directional_penalty +
                self.confidence_weight * confidence_penalty +
                0.05 * extreme_penalty)

def train_model_with_early_stopping(model, X_train, y_train, X_val, y_val, epochs=150, patience=30):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X_train, y_train = X_train.to(device), y_train.to(device)
    X_val, y_val = X_val.to(device), y_val.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=0.0008, weight_decay=1e-4)  # AdamW с улучшенными параметрами
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.7, min_lr=1e-6)  # Более агрессивное снижение
    criterion = DirectionalAccuracyLoss(mse_weight=0.2, da_weight=0.7, confidence_weight=0.1)  # Улучшенная функция потерь

    train_losses, val_losses, val_das = [], [], []
    best_val_loss = float('inf')
    best_da = 0
    patience_counter = 0
    best_model_state = None

    print("\nНачинаем обучение с ранней остановкой по DA...")
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

            # Вычисляем Directional Accuracy
            val_pred_changes = val_outputs.cpu().numpy().flatten()
            val_true_changes = y_val.cpu().numpy().flatten()

            true_directions = np.sign(val_true_changes)
            pred_directions = np.sign(val_pred_changes)

            # Исключаем нулевые изменения
            mask = (true_directions != 0)
            if np.sum(mask) > 0:
                current_da = np.mean(true_directions[mask] == pred_directions[mask])
            else:
                current_da = 0

            # Анализ распределения предсказаний
            pred_up = np.sum(val_pred_changes > 0.001) / len(val_pred_changes)
            pred_down = np.sum(val_pred_changes < -0.001) / len(val_pred_changes)
            pred_flat = 1.0 - pred_up - pred_down

        scheduler.step(val_loss)

        train_losses.append(loss.item())
        val_losses.append(val_loss.item())
        val_das.append(current_da)

        # Улучшенный критерий early stopping
        da_improved = current_da > best_da + 0.01  # минимальное улучшение DA
        loss_improved = val_loss < best_val_loss - 1e-6  # минимальное улучшение loss
        balanced_improvement = (current_da >= best_da - 0.02 and loss_improved)  # тот же DA + лучший loss

        improvement = da_improved or balanced_improvement

        if improvement:
            best_val_loss = val_loss
            best_da = current_da
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            best_epoch = epoch + 1
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch [{epoch+1:3d}/{epochs}] | Train Loss: {loss.item():.6f} | Val Loss: {val_loss.item():.6f} | Val DA: {current_da:.3f} | Patience: {patience_counter}/{patience}')
            print(f'  Pred Distribution: ↑{pred_up:.1%} ↓{pred_down:.1%} →{pred_flat:.1%}')

        if patience_counter >= patience:
            print(f"\nРанняя остановка на эпохе {epoch+1}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Восстановлены веса с эпохи {best_epoch}")
        print(f"Лучший Val Loss: {best_val_loss:.6f} (эпоха {best_epoch})")
        print(f"Лучшая Val DA: {best_da:.3f} (эпоха {best_epoch})")

    return train_losses, val_losses, val_das

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
            real_price = original_data.loc[date, 'High']
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
    plt.title('AAPL: Реальная цена High vs Предсказание', fontsize=14)
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
    plt.savefig('stock_predictions_hi.png', dpi=300, bbox_inches='tight')
    plt.show()  # Показываем график в Colab
    print("График сохранен как 'stock_predictions_hi.png'")

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

def calculate_directional_accuracy(model, X_test, y_test, base_prices_test):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    with torch.no_grad():
        predicted_changes = model(X_test.to(device)).cpu().numpy().flatten()

    real_changes = y_test.numpy().flatten()

    # Направления изменений
    true_directions = np.sign(real_changes)
    pred_directions = np.sign(predicted_changes)

    # Исключаем нулевые изменения (когда цена не изменилась)
    mask = (true_directions != 0)

    if np.sum(mask) > 0:
        accuracy = np.mean(true_directions[mask] == pred_directions[mask])
        n_correct = np.sum(true_directions[mask] == pred_directions[mask])
        n_total = np.sum(mask)

        print(f"\n=== DIRECTIONAL ACCURACY ===")
        print(f"Правильно предсказано направлений: {n_correct}/{n_total}")
        print(f"Directional Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")

        # Детализация по направлениям
        up_mask = (true_directions == 1)
        down_mask = (true_directions == -1)

        if np.sum(up_mask) > 0:
            up_accuracy = np.mean(pred_directions[up_mask] == 1)
            print(f"  Точность для роста: {up_accuracy:.3f} ({np.sum(up_mask)} случаев)")

        if np.sum(down_mask) > 0:
            down_accuracy = np.mean(pred_directions[down_mask] == -1)
            print(f" Точность для падения: {down_accuracy:.3f} ({np.sum(down_mask)} случаев)")

        return accuracy
    else:
        print("Нет данных для расчета Directional Accuracy")
        return 0

# Основной код
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Устройство: {device}")

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

    # Создаем улучшенную модель с обновленным количеством признаков
    model = ImprovedStockPredictor(input_size=20, hidden_size=768, num_layers=3, dropout=0.2).to(device)
    print(f"\nМодель создана: {sum(p.numel() for p in model.parameters()):,} параметров")

    # Обучаем
    train_losses, val_losses, val_das = train_model_with_early_stopping(
        model, X_train, y_train, X_val, y_val, epochs=150, patience=30
    )

    # Тестируем
    print("\n" + "="*50)
    print("ТЕСТИРОВАНИЕ НА ТЕСТОВЫХ ДАННЫХ")
    print("="*50)

    real_prices, pred_prices = plot_results_with_changes(
        model, X_test, y_test, dates_test, base_test, data
    )

    da = calculate_directional_accuracy(model, X_test, y_test, base_test)

    print("\n✅ Обучение и тестирование завершено!")
