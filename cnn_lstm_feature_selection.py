import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

from cnn_lstm_model import StockCNNLSTM, CNNLSTMDirectionalLoss
from reduced_cnn_lstm_model import ReducedStockCNNLSTM
from feature_selector import FeatureSelector

def compute_rsi(prices, window=14):
    """Вычисление RSI индикатора"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(prices, fast=12, slow=26, signal=9):
    """Вычисление MACD индикатора"""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal).mean()
    return macd - signal_line  # MACD гистограмма

def compute_bollinger_bands(prices, window=20):
    """Вычисление Bollinger Bands"""
    sma = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()
    upper_band = sma + (std * 2)
    lower_band = sma - (std * 2)
    return upper_band, lower_band

def prepare_data(symbol='AAPL', period='5y', seq_length=60):
    """Подготовка данных для CNN+LSTM модели"""
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
    data['Momentum'] = data['High'] - data['High'].shift(10)  # 10-дневный моментум

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
    print(f" Падение: {(train_changes < -0.001).mean():.1%}")
    print(f" Без изменений: {(np.abs(train_changes) <= 0.01).mean():.1%}")

    return (X_train, X_val, X_test, y_train, y_val, y_test,
            dates_train, dates_val, dates_test, scaler, data,
            base_train, base_val, base_test)

def train_cnn_lstm_model(model, X_train, y_train, X_val, y_val, epochs=50, patience=15):
    """Обучение CNN+LSTM модели с ранней остановкой"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X_train, y_train = X_train.to(device), y_train.to(device)
    X_val, y_val = X_val.to(device), y_val.to(device)

    # Используем AdamW с оптимизированными параметрами
    optimizer = optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.6, min_lr=1e-6)

    # Используем специализированную функцию потерь для DA
    criterion = CNNLSTMDirectionalLoss(mse_weight=0.3, da_weight=0.6, attention_weight=0.05, pattern_weight=0.05)

    train_losses, val_losses, val_das = [], [], []
    best_val_loss = float('inf')
    best_da = 0
    patience_counter = 0
    best_model_state = None

    print("\nНачинаем обучение CNN+LSTM с ранней остановкой по DA...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Используем градиентную проверку
        outputs, attention_weights = model(X_train)
        loss = criterion(outputs.squeeze(), y_train, attention_weights)
        loss.backward()

        # Обрезка градиентов для стабильности
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_outputs, val_attention_weights = model(X_val)
            val_loss = criterion(val_outputs.squeeze(), y_val, val_attention_weights)

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

        # Улучшенный критерий ранней остановки с акцентом на DA
        da_improved = current_da > best_da + 0.02  # уменьшенный порог для более частых обновлений
        loss_improved = val_loss < best_val_loss - 1e-5  # уменьшенный порог
        balanced_improvement = (current_da >= best_da - 0.01 and loss_improved)  # тот же DA + лучший loss

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

def calculate_directional_accuracy(model, X_test, y_test, base_prices_test):
    """Расчет Directional Accuracy для тестовых данных"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    with torch.no_grad():
        predicted_changes, _ = model(X_test.to(device))
        predicted_changes = predicted_changes.cpu().numpy().flatten()

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

def prepare_reduced_data(X_train, X_val, X_test, selected_features):
    """Подготовка данных с отобранными признаками"""
    # Применяем селекцию признаков
    X_train_reduced = X_train[:, :, selected_features]
    X_val_reduced = X_val[:, :, selected_features]
    X_test_reduced = X_test[:, :, selected_features]

    return X_train_reduced, X_val_reduced, X_test_reduced

def run_feature_selection_and_comparison():
    """Запуск анализа важности признаков и сравнения моделей"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Устройство: {device}")

    # Загружаем данные
    print("Загрузка данных...")
    (X_train, X_val, X_test, y_train, y_val, y_test,
     dates_train, dates_val, dates_test, scaler, data,
     base_train, base_val, base_test) = prepare_data('AAPL', period='5y')

    print(f"\nРазмеры данных:")
    print(f"  Обучающие: {X_train.shape} ({len(dates_train)} дат)")
    print(f" Валидационные: {X_val.shape} ({len(dates_val)} дат)")
    print(f"  Тестовые: {X_test.shape} ({len(dates_test)} дат)")
    print(f" Период данных: {data.index[0].strftime('%Y-%m-%d')} - {data.index[-1].strftime('%Y-%m-%d')}")

    # Создаем полную CNN+LSTM модель
    full_model = StockCNNLSTM(
        input_size=20,  # Количество признаков из stock_predictor_hi.py
        seq_length=60,
        cnn_channels=[32, 64, 128, 256],
        lstm_hidden_size=256,
        lstm_layers=2,
        dropout=0.3
    ).to(device)

    print(f"\nCNN+LSTM полная модель создана: {sum(p.numel() for p in full_model.parameters()):,} параметров")

    # Обучаем полную модель
    print("\nОбучение полной модели...")
    train_losses_full, val_losses_full, val_das_full = train_cnn_lstm_model(
        full_model, X_train, y_train, X_val, y_val, epochs=30, patience=10
    )

    # Тестируем полную модель
    print("\n" + "="*50)
    print("ТЕСТИРОВАНИЕ ПОЛНОЙ МОДЕЛИ")
    print("="*50)
    da_full = calculate_directional_accuracy(full_model, X_test, y_test, base_test)

    # Анализ важности признаков
    print("\nАнализ важности признаков...")
    feature_names = ['High', 'Volume', 'Returns', 'EMA_10', 'EMA_50',
                     'Volatility', 'Volume_EMA', 'RSI', 'MACD', 'BB_Position',
                     'High_Low_Pct', 'Price_Change', 'Volume_Change', 'Momentum', 'MA_Ratio',
                     'Adaptive_Volatility', 'Trend_Strength', 'Price_Strength', 'Price_to_VWAP', 'Volatility_Momentum']

    # Извлекаем признаки из модели для анализа важности
    # В реальности нам нужно использовать внутренние веса или градиенты для оценки важности
    # Временно создадим имитацию анализа важности
    importance_scores = np.random.dirichlet(np.ones(20))  # Замените на реальный анализ важности

    # Используем селектор признаков
    selector = FeatureSelector(importance_threshold=0.04, correlation_threshold=0.8)
    selected_indices, selection_report = selector.select_features(
        X_train.numpy(), importance_scores
    )

    print(f"\nРезультаты селекции:")
    print(f"  Исходное количество: {selection_report['original_count']}")
    print(f"  Отобрано: {selection_report['final_selected']}")
    print(f"  Сокращение: {selection_report['reduction_ratio']:.1%}")
    print(f"  Отобранные признаки: {[feature_names[i] for i in selected_indices]}")

    # Подготовка сокращенных данных
    X_train_reduced, X_val_reduced, X_test_reduced = prepare_reduced_data(
        X_train, X_val, X_test, selected_indices
    )

    # Создаем сокращенную CNN+LSTM модель
    reduced_model = ReducedStockCNNLSTM(
        input_size=len(selected_indices),  # Уменьшенное количество признаков
        seq_length=60,
        cnn_channels=[16, 32, 64, 128],  # Уменьшенные каналы
        lstm_hidden_size=128,  # Уменьшенный размер LSTM
        lstm_layers=2,
        dropout=0.3
    ).to(device)

    print(f"\nCNN+LSTM сокращенная модель создана: {sum(p.numel() for p in reduced_model.parameters()):,} параметров")
    print(f"Количество признаков: {len(selected_indices)} (было 20)")

    # Обучаем сокращенную модель
    print("\nОбучение сокращенной модели...")
    train_losses_reduced, val_losses_reduced, val_das_reduced = train_cnn_lstm_model(
        reduced_model, X_train_reduced, y_train, X_val_reduced, y_val, epochs=30, patience=10
    )

    # Тестируем сокращенную модель
    print("\n" + "="*50)
    print("ТЕСТИРОВАНИЕ СОКРАЩЕННОЙ МОДЕЛИ")
    print("="*50)
    da_reduced = calculate_directional_accuracy(reduced_model, X_test_reduced, y_test, base_test)

    # Сравнение результатов
    print("\n" + "="*50)
    print("СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
    print("="*50)

    full_params = sum(p.numel() for p in full_model.parameters())
    reduced_params = sum(p.numel() for p in reduced_model.parameters())
    param_reduction = (full_params - reduced_params) / full_params * 100

    print(f"Полная модель:")
    print(f"  Параметры: {full_params:,}")
    print(f"  DA: {da_full:.3f} ({da_full*100:.1f}%)")

    print(f"\nСокращенная модель:")
    print(f"  Параметры: {reduced_params:,}")
    print(f"  DA: {da_reduced:.3f} ({da_reduced*100:.1f}%)")

    print(f"\nСравнение:")
    print(f"  Сокращение параметров: {param_reduction:.1f}%")
    print(f"  Изменение DA: {da_reduced - da_full:+.3f} ({(da_reduced - da_full)/da_full*100:+.1f}%)")

    return {
        'full_model': {'da': da_full, 'params': full_params},
        'reduced_model': {'da': da_reduced, 'params': reduced_params, 'selected_features': selected_indices},
        'selection_report': selection_report
    }

if __name__ == "__main__":
    results = run_feature_selection_and_comparison()
    print(f"\n✅ Сравнение моделей завершено!")
    print(f"Итоговая Directional Accuracy полной модели: {results['full_model']['da']:.3f}")
    print(f"Итоговая Directional Accuracy сокращенной модели: {results['reduced_model']['da']:.3f}")
