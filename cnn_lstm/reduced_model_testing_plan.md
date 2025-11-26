# План тестирования сокращенной модели с отобранными признаками

## Обзор цели

Протестировать производительность CNN+LSTM модели после сокращения количества признаков с 20 до 8-12 наиболее важных, убедиться что Directional Accuracy не ухудшается, а время обучения сокращается.

## Этапы тестирования

### Этап 1: Подготовка сокращенного набора данных

#### 1.1 Создание функции подготовки данных с отобранными признаками

```python
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
```

### Этап 2: Создание сокращенной модели

#### 2.1 Адаптация архитектуры модели

```python
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
```

### Этап 3: Обучение и тестирование сокращенной модели

#### 3.1 Функция обучения сокращенной модели

```python
def train_reduced_model(model, X_train, y_train, X_val, y_val, epochs=150, patience=30):
    """
    Обучение сокращенной модели с аналогичными гиперпараметрами
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X_train, y_train = X_train.to(device), y_train.to(device)
    X_val, y_val = X_val.to(device), y_val.to(device)

    # Адаптированные гиперпараметры для меньшей модели
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.7, min_lr=1e-6)
    criterion = DirectionalAccuracyLoss(mse_weight=0.2, da_weight=0.7, confidence_weight=0.1)

    train_losses, val_losses, val_das = [], [], []
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

            pred_up = np.sum(val_pred_changes > 0.001) / len(val_pred_changes)
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
```

### Этап 4: Сравнительное тестирование

#### 4.1 Комплексное сравнение моделей

```python
def comprehensive_model_comparison():
    """
    Комплексное сравнение полной и сокращенной моделей
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = {}

    # 1. Тестирование полной модели
    print("=" * 60)
    print("ТЕСТИРОВАНИЕ ПОЛНОЙ МОДЕЛИ (20 признаков)")
    print("=" * 60)

    # Загрузка данных для полной модели
    (X_train_full, X_val_full, X_test_full, y_train_full, y_val_full, y_test_full,
     dates_train_full, dates_val_full, dates_test_full, scaler_full, data_full,
     base_train_full, base_val_full, base_test_full) = prepare_data('AAPL', period='5y')

    # Создание и обучение полной модели
    full_model = ImprovedStockPredictor(input_size=20, hidden_size=768, num_layers=3, dropout=0.2).to(device)

    start_time = time.time()
    train_losses_full, val_losses_full, val_das_full, training_time_full = train_reduced_model(
        full_model, X_train_full, y_train_full, X_val_full, y_val_full, epochs=150, patience=30
    )

    # Тестирование полной модели
    da_full = calculate_directional_accuracy(full_model, X_test_full, y_test_full, base_test_full)

    results['full_model'] = {
        'parameters': sum(p.numel() for p in full_model.parameters()),
        'training_time': training_time_full,
        'directional_accuracy': da_full,
        'features': 20,
        'train_losses': train_losses_full,
        'val_losses': val_losses_full,
        'val_das': val_das_full
    }

    # 2. Анализ важности признаков
    print("\n" + "=" * 60)
    print("АНАЛИЗ ВАЖНОСТИ ПРИЗНАКОВ")
    print("=" * 60)

    feature_names = ['High', 'Volume', 'Returns', 'EMA_10', 'EMA_50',
                    'Volatility', 'Volume_EMA', 'RSI', 'MACD', 'BB_Position',
                    'High_Low_Pct', 'Price_Change', 'Volume_Change', 'Momentum', 'MA_Ratio',
                    'Adaptive_Volatility', 'Trend_Strength', 'Price_Strength', 'Price_to_VWAP', 'Volatility_Momentum']

    # Создание DataLoader для анализа
    train_dataset = TensorDataset(X_train_full, y_train_full)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

    # Анализ важности
    analyzer = ComprehensiveFeatureAnalyzer(full_model, feature_names, device)
    importance_results = analyzer.analyze_all_methods(train_loader, directional_accuracy_metric)

    # Селекция признаков
    selector = FeatureSelector(importance_threshold=0.04, correlation_threshold=0.8)
    selected_indices, selection_report = selector.select_features(
        X_train_full.numpy(), importance_results['combined']
    )

    print(f"\nРезультаты селекции:")
    print(f"  Отобрано признаков: {len(selected_indices)} из 20")
    print(f"  Сокращение: {selection_report['reduction_ratio']:.1%}")

    # 3. Тестирование сокращенной модели
    print("\n" + "=" * 60)
    print("ТЕСТИРОВАНИЕ СОКРАЩЕННОЙ МОДЕЛИ")
    print("=" * 60)

    # Подготовка данных с отобранными признаками
    (X_train_reduced, X_val_reduced, X_test_reduced, y_train_reduced, y_val_reduced, y_test_reduced,
     dates_train_reduced, dates_val_reduced, dates_test_reduced, scaler_reduced, data_reduced,
     base_train_reduced, base_val_reduced, base_test_reduced, selected_feature_names) = prepare_reduced_data(
        'AAPL', '5y', 60, selected_indices
    )

    # Создание и обучение сокращенной модели
    reduced_model = ReducedStockPredictor(
        input_size=len(selected_indices),
        hidden_size=768,
        num_layers=3,
        dropout=0.2
    ).to(device)

    train_losses_reduced, val_losses_reduced, val_das_reduced, training_time_reduced = train_reduced_model(
        reduced_model, X_train_reduced, y_train_reduced, X_val_reduced, y_val_reduced, epochs=150, patience=30
    )

    # Тестирование сокращенной модели
    da_reduced = calculate_directional_accuracy(reduced_model, X_test_reduced, y_test_reduced, base_test_reduced)

    results['reduced_model'] = {
        'parameters': sum(p.numel() for p in reduced_model.parameters()),
        'training_time': training_time_reduced,
        'directional_accuracy': da_reduced,
        'features': len(selected_indices),
        'train_losses': train_losses_reduced,
        'val_losses': val_losses_reduced,
        'val_das': val_das_reduced,
        'selected_features': selected_feature_names
    }

    # 4. Сравнительный анализ
    print("\n" + "=" * 60)
    print("СРАВНИТЕЛЬНЫЙ АНАЛИЗ")
    print("=" * 60)

    full_params = results['full_model']['parameters']
    reduced_params = results['reduced_model']['parameters']
    param_reduction = (full_params - reduced_params) / full_params * 100

    full_time = results['full_model']['training_time']
    reduced_time = results['reduced_model']['training_time']
    time_reduction = (full_time - reduced_time) / full_time * 100

    full_da = results['full_model']['directional_accuracy']
    reduced_da = results['reduced_model']['directional_accuracy']
    da_change = (reduced_da - full_da) / full_da * 100

    print(f"Параметры:")
    print(f"  Полная модель: {full_params:,}")
    print(f"  Сокращенная модель: {reduced_params:,}")
    print(f"  Сокращение параметров: {param_reduction:.1f}%")

    print(f"\nВремя обучения:")
    print(f"  Полная модель: {full_time:.2f} сек")
    print(f"  Сокращенная модель: {reduced_time:.2f} сек")
    print(f"  Ускорение обучения: {time_reduction:.1f}%")

    print(f"\nDirectional Accuracy:")
    print(f"  Полная модель: {full_da:.3f} ({full_da*100:.1f}%)")
    print(f"  Сокращенная модель: {reduced_da:.3f} ({reduced_da*100:.1f}%)")
    print(f"  Изменение DA: {da_change:+.1f}%")

    # 5. Визуализация результатов
    create_comparison_plots(results, importance_results, selected_indices, feature_names)

    return results, importance_results, selected_indices
```

#### 4.2 Визуализация сравнения

```python
def create_comparison_plots(results, importance_results, selected_indices, feature_names):
    """
    Создание графиков сравнения моделей
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Сравнение потерь
    axes[0, 0].plot(results['full_model']['train_losses'], label='Полная модель (train)', alpha=0.7)
    axes[0, 0].plot(results['full_model']['val_losses'], label='Полная модель (val)', alpha=0.7)
    axes[0, 0].plot(results['reduced_model']['train_losses'], label='Сокращенная модель (train)', alpha=0.7)
    axes[0, 0].plot(results['reduced_model']['val_losses'], label='Сокращенная модель (val)', alpha=0.7)
    axes[0, 0].set_title('Сравнение потерь')
    axes[0, 0].set_xlabel('Эпоха')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Сравнение Directional Accuracy
    axes[0, 1].plot(results['full_model']['val_das'], label='Полная модель', alpha=0.7)
    axes[0, 1].plot(results['reduced_model']['val_das'], label='Сокращенная модель', alpha=0.7)
    axes[0, 1].set_title('Сравнение Directional Accuracy')
    axes[0, 1].set_xlabel('Эпоха')
    axes[0, 1].set_ylabel('DA')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Важность признаков
    importance_percentages = importance_results['combined'] * 100
    sorted_idx = np.argsort(importance_percentages)[::-1]
    sorted_names = [feature_names[i] for i in sorted_idx]
    sorted_values = importance_percentages[sorted_idx]

    bars = axes[1, 0].barh(range(len(sorted_names)), sorted_values, color='skyblue')
    axes[1, 0].set_yticks(range(len(sorted_names)))
    axes[1, 0].set_yticklabels(sorted_names)
    axes[1, 0].set_xlabel('Важность (%)')
    axes[1, 0].set_title('Важность признаков')
    axes[1, 0].invert_yaxis()

    # 4. Сравнительные метрики
    metrics = ['Параметры', 'Время обучения', 'DA']
    full_values = [
        results['full_model']['parameters'] / 1000,  # в тысячах
        results['full_model']['training_time'],
        results['full_model']['directional_accuracy'] * 100
    ]
    reduced_values = [
        results['reduced_model']['parameters'] / 1000,
        results['reduced_model']['training_time'],
        results['reduced_model']['directional_accuracy'] * 100
    ]

    x = np.arange(len(metrics))
    width = 0.35

    axes[1, 1].bar(x - width/2, full_values, width, label='Полная модель', alpha=0.7)
    axes[1, 1].bar(x + width/2, reduced_values, width, label='Сокращенная модель', alpha=0.7)
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(metrics)
    axes[1, 1].set_title('Сравнительные метрики')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
```

## Ожидаемые результаты

### Количественные метрики:
- **Сокращение параметров**: 30-50%
- **Ускорение обучения**: 20-40%
- **Directional Accuracy**: не ниже исходной модели (допустимо снижение до 2%)
- **Количество признаков**: 8-12 из 20

### Качественные преимущества:
- Улучшенная интерпретируемость модели
- Снижение риска переобучения
- Более быстрая инференция
- Упрощение деплоймента

## Критерии успеха

1. **DA не ухудшается**: Directional Accuracy сокращенной модели не более чем на 2% ниже полной
2. **Значительное ускорение**: Время обучения сокращается минимум на 20%
3. **Существенное сокращение**: Количество признаков сокращается минимум на 40%
4. **Стабильность**: Результаты воспроизводимы на разных временных периодах

Этот план обеспечивает систематический подход к тестированию сокращенной модели с объективными критериями оценки и всесторонним сравнением с исходной моделью.
