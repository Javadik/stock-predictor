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

class ReducedStockPredictor(nn.Module):
    """
    Сокращенная версия модели с адаптированной архитектурой
    """
    def __init__(self, input_size, hidden_size=512, num_layers=2, dropout=0.2):
        super().__init__()
        # Уменьшаем размерность скрытых слоев пропорционально сокращению признаков
        reduction_factor = input_size / 20  # Отношение новых признаков к исходным
        adjusted_hidden_size = int(hidden_size * reduction_factor)

        # Bidirectional LSTM с адаптированными параметрами
        self.lstm = nn.LSTM(input_size, adjusted_hidden_size, num_layers,
                           batch_first=True, dropout=dropout, bidirectional=True)
        self.dropout = nn.Dropout(dropout)

        # Многоголовое внимание с адаптированными параметрами
        self.multi_head_attention = nn.MultiheadAttention(
            adjusted_hidden_size * 2, num_heads=max(4, int(8 * reduction_factor)),
            dropout=dropout, batch_first=True
        )
        self.layer_norm1 = nn.LayerNorm(adjusted_hidden_size * 2)

        # Улучшенные слои репрезентации
        self.enhanced_layers = nn.Sequential(
            nn.Linear(adjusted_hidden_size * 2, adjusted_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(adjusted_hidden_size, adjusted_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Механизм внимания
        self.attention_weights = nn.Linear(adjusted_hidden_size // 2, 1)
        self.attention_softmax = nn.Softmax(dim=1)

        self.linear1 = nn.Linear(adjusted_hidden_size // 2, adjusted_hidden_size // 2)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(adjusted_hidden_size // 2, adjusted_hidden_size // 4)
        self.linear3 = nn.Linear(adjusted_hidden_size // 4, 1)

    def forward(self, x, return_attention=False):
        lstm_out, (hidden, _) = self.lstm(x)

        # Многоголовое внимание
        attended_out, multi_attention_weights = self.multi_head_attention(lstm_out, lstm_out, lstm_out)
        lstm_out = self.layer_norm1(lstm_out + attended_out)

        # Улучшенные слои репрезентации
        enhanced_features = self.enhanced_layers(lstm_out)

        # Механизм внимания
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

def prepare_reduced_data(symbol='AAPL', period='5y', seq_length=60, selected_features=None):
    """
    Подготовка данных с отобранными признаками

    Args:
        symbol: тикер акции
        period: период данных
        seq_length: длина последовательности
        selected_features: список индексов отобранных признаков

    Returns:
        Подготовленные данные с сокращенным набором признаков
    """
    # Используем оригинальную функцию подготовки данных
    from stock_predictor_hi import prepare_data, compute_rsi, compute_macd, compute_bollinger_bands

    (X_train, X_val, X_test, y_train, y_val, y_test,
     dates_train, dates_val, dates_test, scaler, data,
     base_train, base_val, base_test) = prepare_data(symbol, period, seq_length)

    if selected_features is not None:
        # Применяем селекцию признаков
        X_train = X_train[:, :, selected_features]
        X_val = X_val[:, :, selected_features]
        X_test = X_test[:, :, selected_features]

        # Создаем новый scaler только для отобранных признаков
        # Это важно для корректного масштабирования
        original_features = ['High', 'Volume', 'Returns', 'EMA_10', 'EMA_50',
                          'Volatility', 'Volume_EMA', 'RSI', 'MACD', 'BB_Position',
                          'High_Low_Pct', 'Price_Change', 'Volume_Change', 'Momentum', 'MA_Ratio',
                          'Adaptive_Volatility', 'Trend_Strength', 'Price_Strength', 'Price_to_VWAP', 'Volatility_Momentum']

        selected_feature_names = [original_features[i] for i in selected_features]

        # Пересоздаем scaler только для отобранных признаков
        train_data_reduced = data[selected_feature_names].iloc[:len(X_train)]
        val_data_reduced = data[selected_feature_names].iloc[len(X_train):len(X_train)+len(X_val)]
        test_data_reduced = data[selected_feature_names].iloc[len(X_train)+len(X_val):]

        scaler_reduced = StandardScaler()
        scaled_train_reduced = scaler_reduced.fit_transform(train_data_reduced)
        scaled_val_reduced = scaler_reduced.transform(val_data_reduced)
        scaled_test_reduced = scaler_reduced.transform(test_data_reduced)

        # Пересоздаем последовательности с новыми данными
        def create_reduced_sequences(scaled_data, original_data, dates_data, seq_length):
            X, y, dates, base_prices = [], [], [], []
            for i in range(seq_length, len(scaled_data)-1):
                X.append(scaled_data[i-seq_length:i])
                price_change = (original_data['High'].iloc[i+1] - original_data['High'].iloc[i]) / original_data['High'].iloc[i]
                y.append(price_change)
                dates.append(dates_data.index[i+1])
                base_prices.append(original_data['High'].iloc[i])
            return (torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(y)),
                    dates, base_prices)

        X_train, y_train, dates_train, base_train = create_reduced_sequences(
            scaled_train_reduced, train_data_reduced, train_data_reduced, seq_length
        )
        X_val, y_val, dates_val, base_val = create_reduced_sequences(
            scaled_val_reduced, val_data_reduced, val_data_reduced, seq_length
        )
        X_test, y_test, dates_test, base_test = create_reduced_sequences(
            scaled_test_reduced, test_data_reduced, test_data_reduced, seq_length
        )

    return (X_train, X_val, X_test, y_train, y_val, y_test,
            dates_train, dates_val, dates_test, scaler_reduced, data,
            base_train, base_val, base_test, selected_feature_names)

def train_reduced_model(model, X_train, y_train, X_val, y_val, epochs=150, patience=30):
    """
    Обучение сокращенной модели с аналогичными гиперпараметрами
    """
    from stock_predictor_hi import DirectionalAccuracyLoss

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X_train, y_train = X_train.to(device), y_train.to(device)
    X_val, y_val = X_val.to(device), y_val.to(device)

    # Адаптированные гиперпараметры для меньшей модели
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.7, min_lr=1e-6)
    criterion = DirectionalAccuracyLoss(mse_weight=0.2, da_weight=0.7, confidence_weight=0.1)

    train_losses, val_losses, val_das = [], []
    best_val_loss = float('inf')
    best_da = 0
    patience_counter = 0
    best_model_state = None

    print(f"\nОбучение сокращенной модели ({X_train.shape[2]} признаков)...")
    start_time = time.time()

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

            mask = (true_directions != 0)
            if np.sum(mask) > 0:
                current_da = np.mean(true_directions[mask] == pred_directions[mask])
            else:
                current_da = 0

            pred_up = np.sum(val_pred_changes > 0.01) / len(val_pred_changes)
            pred_down = np.sum(val_pred_changes < -0.001) / len(val_pred_changes)
            pred_flat = 1.0 - pred_up - pred_down

        scheduler.step(val_loss)

        train_losses.append(loss.item())
        val_losses.append(val_loss.item())
        val_das.append(current_da)

        # Критерии early stopping
        da_improved = current_da > best_da + 0.01
        loss_improved = val_loss < best_val_loss - 1e-6
        balanced_improvement = (current_da >= best_da - 0.02 and loss_improved)

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

    training_time = time.time() - start_time

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Восстановлены веса с эпохи {best_epoch}")
        print(f"Лучший Val Loss: {best_val_loss:.6f} (эпоха {best_epoch})")
        print(f"Лучшая Val DA: {best_da:.3f} (эпоха {best_epoch})")

    print(f"Время обучения: {training_time:.2f} секунд")

    return train_losses, val_losses, val_das, training_time
