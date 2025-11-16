# Архитектура CNN+LSTM для предсказания акций

## Обзор архитектуры

Предлагаемая гибридная архитектура CNN+LSTM сочетает в себе сверточные нейронные сети для извлечения локальных паттернов и LSTM для моделирования временных зависимостей, специально адаптированная для улучшения Directional Accuracy в финансовых временных рядах.

## Основные компоненты

### 1. Входной слой и предварительная обработка
```python
class CNNLSTMStockPredictor(nn.Module):
    def __init__(self, input_size=15, seq_length=60,
                 cnn_channels=[32, 64, 128],
                 lstm_hidden_size=256,
                 lstm_layers=2,
                 dropout=0.3):
        super().__init__()
        self.input_size = input_size
        self.seq_length = seq_length

        # Входной слой для преобразования признаков
        self.input_projection = nn.Conv1d(input_size, cnn_channels[0],
                                         kernel_size=1, padding=0)
```

### 2. Многошкальная CNN часть
```python
class MultiScaleCNN(nn.Module):
    def __init__(self, in_channels, out_channels_list):
        super().__init__()
        self.convs = nn.ModuleList()

        # Разные размеры ядер для извлечения паттернов разных масштабов
        for kernel_size in [3, 5, 7, 9]:
            self.convs.append(nn.Sequential(
                nn.Conv1d(in_channels, out_channels_list[0],
                         kernel_size=kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(out_channels_list[0]),
                nn.ReLU(),
                nn.Dropout(0.2)
            ))

    def forward(self, x):
        # Параллельное применение сверток разных масштабов
        outputs = [conv(x) for conv in self.convs]
        return torch.cat(outputs, dim=1)  # Конкатенация результатов
```

### 3. Иерархические CNN слои
```python
class HierarchicalCNN(nn.Module):
    def __init__(self, input_size, channel_sizes=[32, 64, 128]):
        super().__init__()
        self.layers = nn.ModuleList()

        in_channels = input_size
        for out_channels in channel_sizes:
            self.layers.append(nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2),  # Пулинг для уменьшения размерности
                nn.Dropout(0.2)
            ))
            in_channels = out_channels

    def forward(self, x):
        feature_maps = []
        for layer in self.layers:
            x = layer(x)
            feature_maps.append(x)
        return feature_maps  # Возвращаем карты признаков с разных уровней
```

### 4. LSTM часть с вниманием
```python
class AttentionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout, bidirectional=True)

        # Механизм внимания
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch_size, seq_len, features)
        lstm_out, (hidden, _) = self.lstm(x)

        # Вычисление весов внимания
        attention_scores = self.attention(lstm_out)  # (batch_size, seq_len, 1)
        attention_weights = F.softmax(attention_scores, dim=1)

        # Применение весов внимания
        attended_output = torch.sum(lstm_out * attention_weights, dim=1)

        return attended_output, attention_weights
```

## Специализированные компоненты для финансовых данных

### 1. Извлечение технических паттернов
```python
class TechnicalPatternExtractor(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        # Свёртки для извлечения свечных паттернов
        self.candlestick_patterns = nn.Sequential(
            nn.Conv1d(num_features, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU()
        )

        # Свёртки для извлечения трендовых паттернов
        self.trend_patterns = nn.Sequential(
            nn.Conv1d(num_features, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=9, padding=4),
            nn.ReLU()
        )

        # Свёртки для извлечения волатильности
        self.volatility_patterns = nn.Sequential(
            nn.Conv1d(num_features, 32, kernel_size=11, padding=5),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=13, padding=6),
            nn.ReLU()
        )

    def forward(self, x):
        candlestick = self.candlestick_patterns(x)
        trend = self.trend_patterns(x)
        volatility = self.volatility_patterns(x)

        # Объединение всех паттернов
        return torch.cat([candlestick, trend, volatility], dim=1)
```

### 2. Адаптивное объединение признаков
```python
class AdaptiveFeatureFusion(nn.Module):
    def __init__(self, feature_sizes):
        super().__init__()
        self.feature_sizes = feature_sizes
        total_features = sum(feature_sizes)

        # Адаптивные веса для разных типов признаков
        self.feature_weights = nn.Parameter(torch.ones(len(feature_sizes)))
        self.fusion_layer = nn.Sequential(
            nn.Linear(total_features, total_features // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(total_features // 2, total_features // 4),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, features):
        # Нормализация весов
        weights = F.softmax(self.feature_weights, dim=0)

        # Взвешенное объединение признаков
        weighted_features = []
        for i, feature in enumerate(features):
            weighted_features.append(feature * weights[i])

        concatenated = torch.cat(weighted_features, dim=1)
        return self.fusion_layer(concatenated)
```

## Полная архитектура модели

### 1. Основная архитектура
```python
class StockCNNLSTM(nn.Module):
    def __init__(self, input_size=15, seq_length=60, num_classes=1):
        super().__init__()

        # Многошкальное извлечение признаков
        self.multiscale_cnn = MultiScaleCNN(input_size, [32, 64, 128, 256])

        # Иерархические CNN слои
        self.hierarchical_cnn = HierarchicalCNN(input_size, [64, 128, 256])

        # Извлечение технических паттернов
        self.pattern_extractor = TechnicalPatternExtractor(input_size)

        # Вычисление общего количества признаков после CNN
        cnn_output_size = self._calculate_cnn_output_size(input_size, seq_length)

        # LSTM с вниманием
        self.attention_lstm = AttentionLSTM(cnn_output_size, 256, num_layers=2)

        # Адаптивное объединение признаков
        feature_sizes = [256, 192, 192]  # Размеры от разных CNN компонентов
        self.feature_fusion = AdaptiveFeatureFusion(feature_sizes)

        # Выходной слой
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        batch_size, seq_len, features = x.size()

        # Транспонирование для Conv1d: (batch_size, features, seq_length)
        x = x.transpose(1, 2)

        # Извлечение признаков разными методами
        multiscale_features = self.multiscale_cnn(x)
        hierarchical_features = self.hierarchical_cnn(x)[-1]  # Берем последний уровень
        pattern_features = self.pattern_extractor(x)

        # Адаптация размерностей
        multiscale_features = F.adaptive_avg_pool1d(multiscale_features, seq_len)
        hierarchical_features = F.adaptive_avg_pool1d(hierarchical_features, seq_len)
        pattern_features = F.adaptive_avg_pool1d(pattern_features, seq_len)

        # Транспонирование обратно для LSTM
        multiscale_features = multiscale_features.transpose(1, 2)
        hierarchical_features = hierarchical_features.transpose(1, 2)
        pattern_features = pattern_features.transpose(1, 2)

        # Объединение признаков
        combined_features = torch.cat([multiscale_features,
                                      hierarchical_features,
                                      pattern_features], dim=2)

        # Обработка LSTM с вниманием
        lstm_output, attention_weights = self.attention_lstm(combined_features)

        # Классификация
        output = self.classifier(lstm_output)

        return output, attention_weights
```

## Оптимизация для Directional Accuracy

### 1. Специализированная функция потерь
```python
class CNNLSTMDirectionalLoss(nn.Module):
    def __init__(self, mse_weight=0.3, da_weight=0.7, attention_weight=0.1):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.mse_weight = mse_weight
        self.da_weight = da_weight
        self.attention_weight = attention_weight

    def forward(self, pred, target, attention_weights=None):
        # Стандартная MSE потеря
        mse_loss = self.mse_loss(pred.squeeze(), target)

        # Направленная потеря
        pred_direction = torch.sign(pred.squeeze())
        target_direction = torch.sign(target)

        # Усиленный штраф за неправильное направление
        directional_penalty = torch.mean(
            torch.relu(-pred_direction * target_direction) *
            (1 + torch.abs(pred.squeeze()))
        )

        total_loss = self.mse_weight * mse_loss + self.da_weight * directional_penalty

        # Регуляризация внимания (если предоставлены веса внимания)
        if attention_weights is not None:
            # Штраф за слишком концентрированное внимание
            attention_entropy = -torch.mean(
                torch.sum(attention_weights * torch.log(attention_weights + 1e-8), dim=1)
            )
            total_loss += self.attention_weight * attention_entropy

        return total_loss
```

## Гиперпараметры

### CNN параметры
- **kernel_sizes**: [3, 5, 7, 9] для многошкального анализа
- **channels**: [32, 64, 128, 256] для разных уровней абстракции
- **pooling**: MaxPool1d с kernel_size=2
- **dropout**: 0.2-0.3 для регуляризации

### LSTM параметры
- **hidden_size**: 256 (двунаправленная: 512)
- **num_layers**: 2-3
- **dropout**: 0.2-0.4
- **bidirectional**: True

### Общие параметры
- **sequence_length**: 60-120
- **batch_size**: 32-64
- **learning_rate**: 0.001 с адаптацией
- **optimizer**: Adam с weight_decay=1e-5

## Преимущества для DA

### 1. Улучшенное распознавание паттернов
- **Локальные паттерны**: CNN эффективно извлекает свечные и краткосрочные паттерны
- **Многошкальный анализ**: Одновременный анализ паттернов разных временных масштабов
- **Технические индикаторы**: Автоматическое извлечение паттернов технического анализа

### 2. Временной контекст
- **Долгосрочная память**: LSTM сохраняет информацию о долгосрочных трендах
- **Внимание**: Фокусировка на наиболее важных временных периодах
- **Адаптивность**: Динамическая адаптация к изменяющимся рыночным условиям

### 3. Снижение запаздывания
- **Быстрое обнаружение**: CNN быстро обнаруживает формирующиеся паттерны
- **Своевременные сигналы**: LSTM обеспечивает своевременные предсказания на основе этих паттернов
- **Баланс**: Оптимальный баланс между скоростью обнаружения и точностью предсказания

Эта архитектура CNN+LSTM специально разработана для улучшения Directional Accuracy, сочетая мощное извлечение локальных паттернов с моделированием долгосрочных временных зависимостей, что особенно важно для предсказания направления движения финансовых рынков.
