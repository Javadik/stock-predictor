# Детальный план реализации Transformer архитектуры (Этап 2)

## Обзор цели

**Цель**: Реализовать Transformer архитектуру для улучшения Directional Accuracy с текущих 43-59% до целевых 51-71% (улучшение на +8-12%).

## Анализ текущей ситуации

### Результаты Этапа 1
- Улучшенная LSTM модель с 20 признаками
- Увеличена размерность до 768 скрытых нейронов
- Добавлен многоголовый механизм внимания
- Улучшенная функция потерь с фокусом на DA
- **Ожидаемое улучшение DA**: +3-5% (до 46-64%)

### Преимущества Transformer для нашей задачи
1. **Параллельная обработка**: Одновременный анализ всех временных шагов
2. **Многоголовое внимание**: Фокус на разных временных масштабах
3. **Позиционное кодирование**: Сохранение временной информации
4. **Глобальный контекст**: Улучшенное понимание долгосрочных зависимостей

## Детальная архитектура Transformer

### 1. Основная архитектура
```python
class StockTransformer(nn.Module):
    def __init__(self, input_size=20, d_model=512, nhead=8,
                 num_encoder_layers=6, dim_feedforward=2048,
                 dropout=0.1, max_seq_length=5000):
        super().__init__()

        # 1. Входной слой - проекция признаков
        self.input_projection = nn.Linear(input_size, d_model)

        # 2. Позиционное кодирование для временных рядов
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length, dropout)

        # 3. Энкодер Transformer с каузальным маскированием
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )

        # 4. Многошкальный анализ временных паттернов
        self.temporal_fusion = TemporalFusionLayer(d_model)

        # 5. Выходные слои для предсказания
        self.output_layers = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1)
        )

        self.d_model = d_model
        self.dropout = dropout
```

### 2. Позиционное кодирование для финансовых данных
```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Адаптированное позиционное кодирование для финансовых данных
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Дополнительная кодировка для недельных/месячных паттернов
        weekly_pattern = torch.sin(2 * math.pi * position / 5)  # 5 торговых дней
        monthly_pattern = torch.sin(2 * math.pi * position / 21)  # ~21 торговый день

        pe[:, :d_model//4] += weekly_pattern[:max_len, :d_model//4]
        pe[:, d_model//4:d_model//2] += monthly_pattern[:max_len, :d_model//4-d_model%4]

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
```

### 3. Временная фьюжн-слоя для многошкального анализа
```python
class TemporalFusionLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        # Внимание для разных временных горизонтов
        self.short_term_attention = nn.MultiheadAttention(d_model, 4, batch_first=True)
        self.medium_term_attention = nn.MultiheadAttention(d_model, 4, batch_first=True)
        self.long_term_attention = nn.MultiheadAttention(d_model, 4, batch_first=True)

        # Фьюжн слой
        self.fusion = nn.Linear(d_model * 3, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape

        # Краткосрочное внимание (последние 10 дней)
        short_x = x[:, -10:, :]
        short_out, _ = self.short_term_attention(short_x, short_x, short_x)
        short_pooled = torch.mean(short_out, dim=1)  # [batch_size, d_model]

        # Среднесрочное внимание (последние 30 дней)
        medium_x = x[:, -30:, :]
        medium_out, _ = self.medium_term_attention(medium_x, medium_x, medium_x)
        medium_pooled = torch.mean(medium_out, dim=1)

        # Долгосрочное внимание (вся последовательность)
        long_out, _ = self.long_term_attention(x, x, x)
        long_pooled = torch.mean(long_out, dim=1)

        # Объединение временных масштабов
        combined = torch.cat([short_pooled, medium_pooled, long_pooled], dim=-1)
        fused = self.fusion(combined)

        return self.norm(fused)
```

### 4. Специализированная функция потерь для DA
```python
class TransformerDirectionalLoss(nn.Module):
    def __init__(self, mse_weight=0.2, da_weight=0.6,
                 volatility_weight=0.1, confidence_weight=0.1):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.mse_weight = mse_weight
        self.da_weight = da_weight
        self.volatility_weight = volatility_weight
        self.confidence_weight = confidence_weight

    def forward(self, pred, target, volatility=None):
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

        # 3. Волатильно-адаптивная потеря
        if volatility is not None:
            # Больший штраф в периоды высокой волатильности
            volatility_adjusted_penalty = directional_penalty * (1 + volatility)
        else:
            volatility_adjusted_penalty = directional_penalty

        # 4. Штраф за неуверенные предсказания
        confidence_penalty = torch.mean(torch.abs(pred - target))

        total_loss = (
            self.mse_weight * mse_loss +
            self.da_weight * volatility_adjusted_penalty +
            self.confidence_weight * confidence_penalty
        )

        return total_loss
```

## План реализации

### Фаза 1: Базовая реализация (Дни 1-3)

#### День 1: Основная архитектура
- [ ] Создать файл `transformer_model.py`
- [ ] Реализовать базовый класс `StockTransformer`
- [ ] Реализовать `PositionalEncoding` для финансовых данных
- [ ] Тестирование базовой архитектуры с простыми данными

#### День 2: Внимание и фьюжн-слои
- [ ] Реализовать `TemporalFusionLayer` для многошкального анализа
- [ ] Добавить каузальное маскирование для предотвращения "заглядывания в будущее"
- [ ] Интегрировать механизм внимания в основную архитектуру

#### День 3: Функция потерь и оптимизатор
- [ ] Реализовать `TransformerDirectionalLoss`
- [ ] Настроить AdamW оптимизатор с адаптивным learning rate
- [ ] Добавить learning rate scheduler для стабильного обучения

### Фаза 2: Интеграция с существующим кодом (Дни 4-5)

#### День 4: Адаптация данных
- [ ] Модифицировать подготовку данных для Transformer
- [ ] Адаптировать существующие 20 признаков для новой архитектуры
- [ ] Создать DataLoader с учетом требований Transformer

#### День 5: Интеграция в основной код
- [ ] Интегрировать Transformer в `stock_predictor_hi.py`
- [ ] Адаптировать существующий цикл обучения
- [ ] Сохранить совместимость с текущими метриками

### Фаза 3: Оптимизация и тестирование (Дни 6-8)

#### День 6: Базовое тестирование
- [ ] Запустить обучение на базовых гиперпараметрах
- [ ] Проверить сходимость и базовую производительность
- [ ] Диагностировать и исправить проблемы

#### День 7: Оптимизация гиперпараметров
- [ ] Настроить d_model, nhead, num_layers
- [ ] Оптимизировать dropout и learning rate
- [ ] Тестирование различных комбинаций

#### День 8: Углубленное тестирование
- [ ] Полное обучение на датасете AAPL 5y
- [ ] Оценка DA на тестовых данных
- [ ] Сравнение с улучшенной LSTM моделью

## Гиперпараметры для тестирования

### Основные параметры
```python
transformer_configs = [
    # Конфигурация 1: Компактная модель
    {
        'd_model': 256,
        'nhead': 8,
        'num_encoder_layers': 4,
        'dim_feedforward': 1024,
        'dropout': 0.2,
        'learning_rate': 0.001
    },

    # Конфигурация 2: Сбалансированная модель
    {
        'd_model': 512,
        'nhead': 8,
        'num_encoder_layers': 6,
        'dim_feedforward': 2048,
        'dropout': 0.1,
        'learning_rate': 0.0005
    },

    # Конфигурация 3: Большая модель
    {
        'd_model': 768,
        'nhead': 12,
        'num_encoder_layers': 8,
        'dim_feedforward': 3072,
        'dropout': 0.15,
        'learning_rate': 0.0003
    }
]
```

### Параметры обучения
```python
training_params = {
    'epochs': 150,
    'batch_size': 32,
    'early_stopping_patience': 30,
    'lr_scheduler_patience': 8,
    'lr_scheduler_factor': 0.7,
    'gradient_clipping': 1.0,
    'weight_decay': 1e-5
}
```

## Ожидаемые результаты

### Количественные цели
- **Улучшение DA**: +8-12% (с 43-59% до 51-71%)
- **Снижение MSE**: на 15-25%
- **Улучшение MAE**: на 10-20%
- **Время обучения**: +60% по сравнению с LSTM

### Качественные цели
- **Лучшая адаптация к волатильности**
- **Улучшенное распознавание трендов**
- **Более стабильные предсказания**

## Риски и митигация

### Технические риски
1. **Переобучение** из-за большого количества параметров
   - Митигация: усиленный dropout, early stopping, weight decay

2. **Вычислительные ресурсы**
   - Митигация: градиентная аккумуляция, смешанная точность (mixed precision)

3. **Нестабильное обучение**
   - Митигация: learning rate warmup, gradient clipping, layer normalization

### Архитектурные риски
1. **Несоответствие временных паттернов**
   - Митигация: адаптивное позиционное кодирование, временная фьюжн-слоя

2. **Проблемы с каузальным маскированием**
   - Митигация: тщательное тестирование маскирования, валидация на будущих данных

## Критерии успеха

### Основные метрики
- **DA > 65%** на тестовых данных
- **Улучшение DA на +8%** по сравнению с текущей LSTM
- **Стабильная сходимость** в течение 100 эпох

### Вторичные метрики
- **Время инференса < 100мс** на GPU
- **Стабильность производительности** на разных периодах
- **Интерпретируемость** через attention weights

## План оценки

### Сравнительное тестирование
1. **Прямое сравнение** с улучшенной LSTM на тех же данных
2. **Кросс-валидация** на разных временных периодах
3. **Стресс-тестирование** на периоды высокой волатильности

### Анализ результатов
1. **Матрица ошибок** для направлений
2. **Attention визуализация** для интерпретации
3. **Анализ производительности** по временным горизонтам

## Следующие шаги после Этапа 2

1. **Анализ результатов** Transformer модели
2. **Реализация CNN+LSTM** (Этап 3)
3. **Сравнительный анализ** всех трех архитектур
4. **Создание ансамбля** лучших моделей
5. **Оптимизация и развертывание** финального решения

---

**Статус**: Готов к реализации
**Следующее действие**: Переключиться в Code режим для начала реализации
