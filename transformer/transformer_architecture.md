# Архитектура Transformer для предсказания акций

## Обзор архитектуры

Предлагаемая архитектура Transformer специально адаптирована для предсказания финансовых временных рядов с фокусом на улучшение Directional Accuracy (DA).

## Основные компоненты

### 1. Входной слой и позиционное кодирование
```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # Создание матрицы позиционного кодирования
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]
```

### 2. Многоголовое внимание (Multi-Head Attention)
- **Количество голов**: 8 голов внимания для параллельного анализа различных аспектов данных
- **Размерность модели**: 512 (d_model)
- **Размерность каждой головы**: 64 (d_model // num_heads)

### 3. Энкодер блоки
```python
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()
```

### 4. Адаптированное для временных рядов внимание
- **Каузальное маскирование**: Предотвращение "заглядывания в будущее"
- **Временные веса**: Специальные веса для учета важности различных временных периодов

## Архитектурные особенности для финансовых данных

### 1. Многошкальный анализ
```python
class MultiScaleTransformer(nn.Module):
    def __init__(self, input_size, d_model=512, nhead=8, num_layers=6):
        super().__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        # Многошкальные энкодеры для разных временных горизонтов
        self.short_term_encoder = TransformerEncoder(...)
        self.medium_term_encoder = TransformerEncoder(...)
        self.long_term_encoder = TransformerEncoder(...)

        # Фьюжн слой для объединения информации
        self.fusion_layer = nn.Linear(d_model * 3, d_model)
```

### 2. Сегментация временных рядов
- **Динамическая сегментация**: Адаптивное разделение временных рядов на сегменты
- **Перекрывающиеся окна**: Использование перекрывающихся окон для лучшего контекста

### 3. Специализированные механизмы внимания
```python
class FinancialAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead)
        self.volatility_attention = nn.MultiheadAttention(d_model, nhead//2)
        self.volume_attention = nn.MultiheadAttention(d_model, nhead//2)

    def forward(self, x, volatility_features, volume_features):
        # Основное внимание к цене
        attn_output, _ = self.attention(x, x, x)

        # Внимание к волатильности
        vol_attn, _ = self.volatility_attention(volatility_features,
                                               volatility_features,
                                               volatility_features)

        # Внимание к объемам
        vol_attn, _ = self.volume_attention(volume_features,
                                           volume_features,
                                           volume_features)

        # Комбинированный вывод
        return self.combine_attentions(attn_output, vol_attn, vol_attn)
```

## Оптимизация для Directional Accuracy

### 1. Специализированная функция потерь
```python
class DirectionalTransformerLoss(nn.Module):
    def __init__(self, mse_weight=0.3, da_weight=0.7):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.mse_weight = mse_weight
        self.da_weight = da_weight

    def forward(self, pred, target):
        # Стандартная MSE потеря
        mse_loss = self.mse_loss(pred, target)

        # Направленная потеря с учетом уверенности
        pred_direction = torch.sign(pred)
        target_direction = torch.sign(target)

        # Усиленный штраф за неправильное направление
        directional_penalty = torch.mean(
            torch.relu(-pred_direction * target_direction) *
            (1 + torch.abs(pred))  # Усиление штрафа для уверенных предсказаний
        )

        return self.mse_weight * mse_loss + self.da_weight * directional_penalty
```

### 2. Адаптивное обучение
- **Динамический learning rate**: Адаптация скорости обучения в зависимости от DA
- **Фокussed sampling**: Увеличение выборки сложных периодов для обучения

## Гиперпараметры

### Основные параметры
- **d_model**: 512 (размерность модели)
- **nhead**: 8 (количество голов внимания)
- **num_layers**: 6 (количество энкодер слоев)
- **dim_feedforward**: 2048 (размерность feedforward сети)
- **dropout**: 0.1-0.3 (регуляризация)

### Специализированные параметры
- **sequence_length**: 60-120 (длина последовательности)
- **attention_window**: 30 (окно внимания для каузального маскирования)
- **volatility_threshold**: 0.02 (порог для адаптивного внимания)

## Преимущества для DA

### 1. Улучшенное распознавание трендов
- **Долгосрочные зависимости**: Механизм внимания улавливает долгосрочные тренды
- **Контекстуальная важность**: Фокусировка на ключевых моментах разворота тренда

### 2. Адаптивность к волатильности
- **Динамическое внимание**: Адаптация к периодам высокой волатильности
- **Мультифакторный анализ**: Одновременный анализ цены, объема и волатильности

### 3. Снижение запаздывания
- **Параллельная обработка**: Одновременный анализ всех временных шагов
- **Быстрое обнаружение**: Эффективное обнаружение паттернов разворота

## Интеграция с существующим кодом

### 1. Совместимость с подготовкой данных
- Использование существующих функций подготовки данных из stock_predictor_hi.py
- Адаптация формата входных данных для Transformer

### 2. Замена модели
```python
# Замена существующей модели
model = StockTransformer(
    input_size=15,  # Количество признаков из stock_predictor_hi.py
    d_model=512,
    nhead=8,
    num_layers=6,
    dropout=0.2
).to(device)
```

### 3. Адаптация обучения
- Использование существующего цикла обучения с адаптацией для Transformer
- Сохранение существующих метрик и визуализации

Эта архитектура Transformer специально разработана для улучшения Directional Accuracy в предсказании акций, сочетая мощь механизма внимания со специализированными адаптациями для финансовых временных рядов.
