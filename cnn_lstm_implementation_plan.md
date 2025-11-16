# Детальный план реализации CNN+LSTM архитектуры (Этап 3)

## Обзор цели

**Цель**: Реализовать гибридную CNN+LSTM архитектуру для улучшения Directional Accuracy с текущих 43-59% до целевых 48-68% (улучшение на +5-9%).

## Анализ преимуществ CNN+LSTM для нашей задачи

### Ключевые преимущества
1. **Извлечение локальных паттернов**: CNN эффективно выявляет свечные модели и краткосрочные паттерны
2. **Моделирование временных зависимостей**: LSTM сохраняет долгосрочный контекст и тренды
3. **Синергия архитектур**: CNN уменьшает размерность данных, LSTM работает с очищенными признаками
4. **Адаптивность к рыночным условиям**: Лучшая производительность в периоды изменчивой волатильности

### Сравнение с другими архитектурами
- **Преимущество над LSTM**: Лучшее извлечение локальных паттернов через свертки
- **Преимущество над Transformer**: Более быстрая обработка и меньше параметров
- **Идеальный баланс**: Оптимальное соотношение скорости и точности

## Детальная архитектура CNN+LSTM

### 1. Основная архитектура
```python
class StockCNNLSTM(nn.Module):
    def __init__(self, input_size=20, seq_length=60,
                 cnn_channels=[32, 64, 128, 256],
                 lstm_hidden_size=256,
                 lstm_layers=2,
                 dropout=0.3):
        super().__init__()

        self.input_size = input_size
        self.seq_length = seq_length

        # 1. Многошкальная CNN часть для извлечения локальных паттернов
        self.multiscale_cnn = MultiScaleCNN(input_size, cnn_channels)

        # 2. Иерархические CNN слои для извлечения паттернов разных уровней
        self.hierarchical_cnn = HierarchicalCNN(input_size, [64, 128, 256])

        # 3. Извлечение технических паттернов (свечные, трендовые, волатильность)
        self.pattern_extractor = TechnicalPatternExtractor(input_size)

        # 4. Вычисление размерности после CNN
        cnn_output_size = self._calculate_cnn_output_size(input_size, seq_length)

        # 5. LSTM с механизмом внимания для временного моделирования
        self.attention_lstm = AttentionLSTM(cnn_output_size, lstm_hidden_size, lstm_layers, dropout)

        # 6. Адаптивное объединение признаков из разных CNN ветвей
        feature_sizes = [256, 192, 192]  # Размеры от разных CNN компонентов
        self.feature_fusion = AdaptiveFeatureFusion(feature_sizes)

        # 7. Выходные слои для предсказания
        self.output_layers = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, 128),  # *2 из-за bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1)
        )
```

### 2. Многошкальная CNN для извлечения паттернов
```python
class MultiScaleCNN(nn.Module):
    def __init__(self, in_channels, out_channels_list):
        super().__init__()
        self.convs = nn.ModuleList()

        # Разные размеры ядер для извлечения паттернов разных масштабов
        kernel_sizes = [3, 5, 7, 9]  # Краткосрочные, среднесрочные паттерны

        for i, kernel_size in enumerate(kernel_sizes):
            self.convs.append(nn.Sequential(
                # Первая свертка
                nn.Conv1d(in_channels, out_channels_list[i],
                         kernel_size=kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(out_channels_list[i]),
                nn.ReLU(),
                nn.Dropout(0.2),

                # Вторая свертка для углубления
                nn.Conv1d(out_channels_list[i], out_channels_list[i],
                         kernel_size=kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(out_channels_list[i]),
                nn.ReLU(),
                nn.Dropout(0.2)
            ))

    def forward(self, x):
        # x shape: (batch_size, features, seq_length)
        outputs = [conv(x) for conv in self.convs]
        return torch.cat(outputs, dim=1)  # Конкатенация по каналам
```

### 3. Иерархические CNN слои
```python
class HierarchicalCNN(nn.Module):
    def __init__(self, input_size, channel_sizes=[64, 128, 256]):
        super().__init__()
        self.layers = nn.ModuleList()

        in_channels = input_size
        for i, out_channels in enumerate(channel_sizes):
            # Увеличиваем размер ядра для более глубоких слоев
            kernel_size = 3 + i * 2

            self.layers.append(nn.Sequential(
                nn.Conv1d(in_channels, out_channels,
                         kernel_size=kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),

                # Пулинг для уменьшения размерности
                nn.MaxPool1d(kernel_size=2, stride=2) if i < len(channel_sizes) - 1 else nn.Identity(),
                nn.Dropout(0.2 + i * 0.05)  # Увеличиваем dropout для глубоких слоев
            ))
            in_channels = out_channels

    def forward(self, x):
        feature_maps = []
        for layer in self.layers:
            x = layer(x)
            feature_maps.append(x)
        return feature_maps  # Возвращаем карты признаков с разных уровней
```

### 4. Извлечение технических паттернов
```python
class TechnicalPatternExtractor(nn.Module):
    def __init__(self, num_features):
        super().__init__()

        # Свёртки для извлечения свечных паттернов (краткосрочные)
        self.candlestick_patterns = nn.Sequential(
            nn.Conv1d(num_features, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Свёртки для извлечения трендовых паттернов (среднесрочные)
        self.trend_patterns = nn.Sequential(
            nn.Conv1d(num_features, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Свёртки для извлечения волатильности (долгосрочные)
        self.volatility_patterns = nn.Sequential(
            nn.Conv1d(num_features, 32, kernel_size=11, padding=5),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=13, padding=6),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Адаптивные веса для разных типов паттернов
        self.pattern_weights = nn.Parameter(torch.ones(3))

    def forward(self, x):
        # x shape: (batch_size, features, seq_length)
        candlestick = self.candlestick_patterns(x)
        trend = self.trend_patterns(x)
        volatility = self.volatility_patterns(x)

        # Взвешенное объединение паттернов
        weights = F.softmax(self.pattern_weights, dim=0)

        weighted_candlestick = candlestick * weights[0]
        weighted_trend = trend * weights[1]
        weighted_volatility = volatility * weights[2]

        # Объединение всех паттернов
        combined = torch.cat([weighted_candlestick, weighted_trend, weighted_volatility], dim=1)
        return combined
```

### 5. LSTM с механизмом внимания
```python
class AttentionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.3):
        super().__init__()

        # Двунаправленный LSTM для захвата прошлого и будущего контекста
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout, bidirectional=True)

        # Механизм внимания для фокусировки на важных временных шагах
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),  # *2 из-за bidirectional
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )

        # Дополнительный слой для улучшения репрезентации
        self.enhancement = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch_size, seq_len, features)
        lstm_out, (hidden, cell) = self.lstm(x)

        # Улучшение признаков
        enhanced_out = self.enhancement(lstm_out)

        # Вычисление весов внимания
        attention_scores = self.attention(enhanced_out)  # (batch_size, seq_len, 1)
        attention_weights = F.softmax(attention_scores, dim=1)

        # Применение весов внимания
        attended_output = torch.sum(lstm_out * attention_weights, dim=1)

        return attended_output, attention_weights
```

### 6. Адаптивное объединение признаков
```python
class AdaptiveFeatureFusion(nn.Module):
    def __init__(self, feature_sizes):
        super().__init__()
        self.feature_sizes = feature_sizes
        total_features = sum(feature_sizes)

        # Обучаемые веса для разных типов признаков
        self.feature_weights = nn.Parameter(torch.ones(len(feature_sizes)))

        # Фьюжн слои для объединения признаков
        self.fusion_layers = nn.Sequential(
            nn.Linear(total_features, total_features // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(total_features // 2, total_features // 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(total_features // 4, total_features // 8)
        )

        # Нормализация для стабилизации обучения
        self.layer_norm = nn.LayerNorm(total_features // 8)

    def forward(self, features):
        # Адаптивная нормализация весов
        weights = F.softmax(self.feature_weights, dim=0)

        # Взвешенное объединение признаков
        weighted_features = []
        for i, feature in enumerate(features):
            # Глобальное среднее для приведения к размеру (batch_size, feature_size)
            if feature.dim() == 3:  # (batch_size, seq_len, feature_size)
                feature = torch.mean(feature, dim=1)
            weighted_features.append(feature * weights[i])

        # Конкатенация и фьюжн
        concatenated = torch.cat(weighted_features, dim=1)
        fused = self.fusion_layers(concatenated)

        return self.layer_norm(fused)
```

### 7. Специализированная функция потерь для DA
```python
class CNNLSTMDirectionalLoss(nn.Module):
    def __init__(self, mse_weight=0.3, da_weight=0.6,
                 attention_weight=0.05, pattern_weight=0.05):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.mse_weight = mse_weight
        self.da_weight = da_weight
        self.attention_weight = attention_weight
        self.pattern_weight = pattern_weight

    def forward(self, pred, target, attention_weights=None, pattern_features=None):
        pred = pred.squeeze()
        target = target.squeeze()

        # 1. Стандартная MSE потеря
        mse_loss = self.mse_loss(pred, target)

        # 2. Направленная потеря с адаптивными весами
        pred_direction = torch.sign(pred)
        target_direction = torch.sign(target)

        # Усиленный штраф за неправильное направление
        directional_penalty = torch.mean(
            torch.relu(-pred_direction * target_direction) *
            (1 + torch.abs(target))  # Вес зависит от величины изменения
        )

        total_loss = self.mse_weight * mse_loss + self.da_weight * directional_penalty

        # 3. Регуляризация внимания (для предотвращения перефокусировки)
        if attention_weights is not None:
            # Поощряем распределенное внимание
            attention_entropy = -torch.mean(
                torch.sum(attention_weights * torch.log(attention_weights + 1e-8), dim=1)
            )
            total_loss += self.attention_weight * attention_entropy

        # 4. Регуляризация паттернов (для предотвращения переобучения на паттерны)
        if pattern_features is not None:
            # L2 регуляризация для паттерн признаков
            pattern_regularization = torch.mean(torch.sum(pattern_features ** 2, dim=1))
            total_loss += self.pattern_weight * pattern_regularization

        return total_loss
```

## План реализации

### Фаза 1: Базовая реализация (Дни 1-3)

#### День 1: CNN компоненты
- [ ] Создать файл `cnn_lstm_model.py`
- [ ] Реализовать `MultiScaleCNN` для извлечения паттернов разных масштабов
- [ ] Реализовать `HierarchicalCNN` для иерархического анализа
- [ ] Тестирование CNN компонентов с синтетическими данными

#### День 2: LSTM и внимание
- [ ] Реализовать `AttentionLSTM` с механизмом внимания
- [ ] Реализовать `TechnicalPatternExtractor` для финансовых паттернов
- [ ] Реализовать `AdaptiveFeatureFusion` для объединения признаков
- [ ] Тестирование LSTM компонентов

#### День 3: Основная архитектура
- [ ] Реализовать основной класс `StockCNNLSTM`
- [ ] Интегрировать все компоненты в единую архитектуру
- [ ] Реализовать `CNNLSTMDirectionalLoss` функцию потерь
- [ ] Базовое тестирование полной архитектуры

### Фаза 2: Интеграция с существующим кодом (Дни 4-5)

#### День 4: Адаптация данных
- [ ] Модифицировать подготовку данных для CNN+LSTM
- [ ] Адаптировать существующие 20 признаков для новой архитектуры
- [ ] Создать эффективный DataLoader с учетом требований CNN
- [ ] Оптимизировать размерность данных для CNN слоев

#### День 5: Интеграция в основной код
- [ ] Интегрировать CNN+LSTM в `stock_predictor_hi.py`
- [ ] Адаптировать существующий цикл обучения для гибридной модели
- [ ] Настроить оптимизатор и scheduler для CNN+LSTM
- [ ] Сохранить совместимость с текущими метриками

### Фаза 3: Оптимизация и тестирование (Дни 6-8)

#### День 6: Базовое тестирование
- [ ] Запустить обучение на базовых гиперпараметрах
- [ ] Проверить сходимость и базовую производительность
- [ ] Диагностировать и исправить проблемы с размерностями
- [ ] Визуализировать attention weights для интерпретации

#### День 7: Оптимизация гиперпараметров
- [ ] Настроить CNN параметры (kernel_sizes, channels, dropout)
- [ ] Оптимизировать LSTM параметры (hidden_size, num_layers)
- [ ] Тестировать разные комбинации fusion стратегий
- [ ] Настроить веса функции потерь

#### День 8: Углубленное тестирование
- [ ] Полное обучение на датасете AAPL 5y
- [ ] Оценка DA на тестовых данных
- [ ] Сравнение с улучшенной LSTM и Transformer моделями
- [ ] Анализ производительности на разных рыночных режимах

## Гиперпараметры для тестирования

### CNN параметры
```python
cnn_configs = [
    # Конфигурация 1: Легкая модель
    {
        'cnn_channels': [32, 64, 128],
        'kernel_sizes': [3, 5, 7],
        'cnn_dropout': 0.2,
        'pooling_strategy': 'max'
    },

    # Конфигурация 2: Сбалансированная модель
    {
        'cnn_channels': [32, 64, 128, 256],
        'kernel_sizes': [3, 5, 7, 9],
        'cnn_dropout': 0.25,
        'pooling_strategy': 'adaptive'
    },

    # Конфигурация 3: Тяжелая модель
    {
        'cnn_channels': [64, 128, 256, 512],
        'kernel_sizes': [3, 5, 7, 9, 11],
        'cnn_dropout': 0.3,
        'pooling_strategy': 'hybrid'
    }
]
```

### LSTM параметры
```python
lstm_configs = [
    # Конфигурация 1: Компактная
    {
        'lstm_hidden_size': 128,
        'lstm_layers': 2,
        'lstm_dropout': 0.2,
        'bidirectional': True
    },

    # Конфигурация 2: Сбалансированная
    {
        'lstm_hidden_size': 256,
        'lstm_layers': 2,
        'lstm_dropout': 0.3,
        'bidirectional': True
    },

    # Конфигурация 3: Большая
    {
        'lstm_hidden_size': 384,
        'lstm_layers': 3,
        'lstm_dropout': 0.35,
        'bidirectional': True
    }
]
```

### Параметры обучения
```python
training_params = {
    'epochs': 150,
    'batch_size': 32,
    'early_stopping_patience': 30,
    'lr_scheduler_patience': 10,
    'lr_scheduler_factor': 0.7,
    'gradient_clipping': 1.0,
    'weight_decay': 1e-5,
    'learning_rate': 0.001
}
```

## Ожидаемые результаты

### Количественные цели
- **Улучшение DA**: +5-9% (с 43-59% до 48-68%)
- **Снижение MSE**: на 10-20%
- **Улучшение MAE**: на 8-15%
- **Время обучения**: +40% по сравнению с LSTM, но -20% по сравнению с Transformer

### Качественные цели
- **Лучшее распознавание свечных паттернов**
- **Улучшенная адаптация к волатильности**
- **Более быстрая обработка по сравнению с Transformer**
- **Хорошая интерпретируемость через attention и feature weights**

## Риски и митигация

### Технические риски
1. **Сложность интеграции** CNN и LSTM компонентов
   - Митигация: пошаговая интеграция, тщательное тестирование размерностей

2. **Переобучение на паттерны**
   - Митигация: усиленный dropout, регуляризация паттернов, аугментация данных

3. **Вычислительная сложность**
   - Митигация: оптимизация CNN слоев, градиентная аккумуляция, смешанная точность

### Архитектурные риски
1. **Несоответствие временных масштабов**
   - Митигация: многошкальный CNN, адаптивное объединение признаков

2. **Проблемы с attention механизмом**
   - Митигация: визуализация attention, регуляризация энтропии

## Критерии успеха

### Основные метрики
- **DA > 62%** на тестовых данных
- **Улучшение DA на +5%** по сравнению с текущей LSTM
- **Стабильная сходимость** в течение 120 эпох

### Вторичные метрики
- **Время инференса < 80мс** на GPU (быстрее Transformer)
- **Стабильность производительности** на разных рыночных режимах
- **Интерпретируемость** через attention weights и feature importance

## План оценки

### Сравнительное тестирование
1. **Прямое сравнение** с LSTM и Transformer на тех же данных
2. **Анализ паттернов** которые CNN улавливает лучше других моделей
3. **Стресс-тестирование** на периоды высокой волатильности

### Анализ результатов
1. **Матрица ошибок** для направлений с детализацией по паттернам
2. **Attention визуализация** для интерпретации временных зависимостей
3. **Feature importance** для понимания вклада разных CNN компонентов

## Следующие шаги после Этапа 3

1. **Сравнительный анализ** всех трех архитектур (LSTM, Transformer, CNN+LSTM)
2. **Создание ансамбля** лучших моделей для максимальной производительности
3. **Оптимизация гиперпараметров** для финальной модели
4. **Развертывание и мониторинг** в production среде
5. **Постоянное улучшение** на основе новых данных и обратной связи

---

**Статус**: Готов к реализации
**Преимущество**: Оптимальный баланс между производительностью и скоростью
**Следующее действие**: Переключиться в Code режим для начала реализации
