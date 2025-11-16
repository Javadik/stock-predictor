# Гиперпараметры для каждой архитектуры

## Общие параметры для всех моделей

### Параметры данных
- **sequence_length**: 60 (количество дней в последовательности)
- **input_size**: 15 (количество признаков из stock_predictor_hi.py)
- **prediction_horizon**: 1 (предсказание на 1 день вперед)
- **train_split**: 0.7 (70% данных для обучения)
- **val_split**: 0.15 (15% для валидации)
- **test_split**: 0.15 (15% для тестирования)

### Параметры обучения
- **batch_size**: 32
- **learning_rate**: 0.001 с адаптацией
- **optimizer**: Adam с weight_decay=1e-5
- **scheduler**: ReduceLROnPlateau (patience=10, factor=0.5)
- **epochs**: 150 с ранней остановкой
- **early_stopping_patience**: 30
- **gradient_clipping**: max_norm=1.0

### Параметры регуляризации
- **dropout_range**: 0.1-0.4 (зависит от архитектуры)
- **weight_decay**: 1e-5
- **batch_norm**: True (где применимо)

## 1. Текущая LSTM (stock_predictor_hi.py)

### Архитектурные параметры
```python
lstm_params = {
    'input_size': 15,
    'hidden_size': 512,        # Увеличено для лучшей производительности
    'num_layers': 2,           # Двунаправленный LSTM
    'dropout': 0.1,            # Низкий dropout из-за bidirectional
    'bidirectional': True
}
```

### Параметры внимания
```python
attention_params = {
    'attention_hidden': 512,   # Размерность слоя внимания
    'attention_dropout': 0.2
}
```

### Параметры полносвязных слоев
```python
fc_params = {
    'fc1_size': 512,           # Первый полносвязный слой
    'fc2_size': 256,           # Второй полносвязный слой
    'fc3_size': 1,             # Выходной слой
    'fc_dropout': 0.2
}
```

### Оптимальные диапазоны для поиска
- **hidden_size**: [256, 384, 512, 768]
- **num_layers**: [2, 3, 4]
- **dropout**: [0.1, 0.2, 0.3]
- **learning_rate**: [0.0005, 0.001, 0.002]

## 2. Transformer

### Архитектурные параметры
```python
transformer_params = {
    'input_size': 15,
    'd_model': 512,            # Размерность модели
    'nhead': 8,                # Количество голов внимания
    'num_encoder_layers': 6,   # Количество энкодер слоев
    'dim_feedforward': 2048,   # Размерность feedforward сети
    'dropout': 0.1,            # Dropout для трансформера
    'activation': 'relu'       # Функция активации
}
```

### Параметры позиционного кодирования
```python
positional_encoding_params = {
    'max_len': 5000,           # Максимальная длина последовательности
    'dropout': 0.1
}
```

### Параметры многошкального анализа
```python
multiscale_params = {
    'short_term_window': 5,    # Окно для краткосрочных паттернов
    'medium_term_window': 15,  # Окно для среднесрочных паттернов
    'long_term_window': 30     # Окно для долгосрочных паттернов
}
```

### Оптимальные диапазоны для поиска
- **d_model**: [256, 384, 512, 768]
- **nhead**: [4, 6, 8, 12] (должно делить d_model без остатка)
- **num_encoder_layers**: [3, 4, 6, 8]
- **dim_feedforward**: [1024, 1536, 2048, 3072]
- **dropout**: [0.1, 0.2, 0.3]
- **learning_rate**: [0.0001, 0.0005, 0.001]

## 3. CNN+LSTM

### CNN параметры
```python
cnn_params = {
    'input_channels': 15,
    'multiscale_channels': [32, 64, 128, 256],  # Каналы для разных масштабов
    'kernel_sizes': [3, 5, 7, 9],              # Размеры ядер
    'hierarchical_channels': [64, 128, 256],     # Каналы для иерархии
    'pooling_kernel': 2,                        # Ядро пулинга
    'cnn_dropout': 0.2
}
```

### LSTM параметры
```python
cnn_lstm_params = {
    'input_size': 768,          # Размерность после CNN
    'hidden_size': 256,         # Размерность скрытого состояния
    'num_layers': 2,            # Количество слоев
    'dropout': 0.3,            # Dropout для LSTM
    'bidirectional': True       # Двунаправленный LSTM
}
```

### Параметры внимания
```python
cnn_lstm_attention_params = {
    'attention_hidden': 256,    # Размерность слоя внимания
    'attention_dropout': 0.2
}
```

### Параметры объединения признаков
```python
fusion_params = {
    'feature_sizes': [256, 192, 192],  # Размеры от разных CNN компонентов
    'fusion_hidden': [384, 192],      # Скрытые слои фьюжн
    'fusion_dropout': 0.3
}
```

### Оптимальные диапазоны для поиска
- **multiscale_channels**: [[16,32,64,128], [32,64,128,256], [64,128,256,512]]
- **kernel_sizes**: [[3,5,7], [3,5,7,9], [5,7,9,11]]
- **lstm_hidden_size**: [128, 192, 256, 384]
- **lstm_num_layers**: [1, 2, 3]
- **cnn_dropout**: [0.1, 0.2, 0.3]
- **lstm_dropout**: [0.2, 0.3, 0.4]

## Стратегии оптимизации гиперпараметров

### 1. Базовая оптимизация
```python
base_optimization = {
    'method': 'grid_search',
    'parameters': ['hidden_size', 'dropout', 'learning_rate'],
    'cv_folds': 3,
    'scoring': 'directional_accuracy'
}
```

### 2. Продвинутая оптимизация
```python
advanced_optimization = {
    'method': 'bayesian_optimization',
    'parameters': ['hidden_size', 'num_layers', 'dropout', 'learning_rate'],
    'n_iterations': 50,
    'acquisition_function': 'expected_improvement'
}
```

### 3. Иерархическая оптимизация
```python
hierarchical_optimization = {
    'stage1': {
        'focus': 'architecture',
        'parameters': ['hidden_size', 'num_layers', 'nhead']
    },
    'stage2': {
        'focus': 'regularization',
        'parameters': ['dropout', 'weight_decay']
    },
    'stage3': {
        'focus': 'training',
        'parameters': ['learning_rate', 'batch_size']
    }
}
```

## Адаптивные гиперпараметры

### В зависимости от размера датасета
```python
adaptive_params = {
    'small_dataset': {
        'hidden_size': 256,
        'num_layers': 2,
        'dropout': 0.3,
        'learning_rate': 0.001
    },
    'medium_dataset': {
        'hidden_size': 512,
        'num_layers': 3,
        'dropout': 0.2,
        'learning_rate': 0.0005
    },
    'large_dataset': {
        'hidden_size': 768,
        'num_layers': 4,
        'dropout': 0.1,
        'learning_rate': 0.0001
    }
}
```

### В зависимости от волатильности рынка
```python
volatility_adaptive = {
    'low_volatility': {
        'sequence_length': 60,
        'dropout': 0.1,
        'attention_window': 30
    },
    'high_volatility': {
        'sequence_length': 30,
        'dropout': 0.3,
        'attention_window': 15
    }
}
```

## Вычислительные ограничения

### Ограничения по памяти
```python
memory_constraints = {
    'low_memory': {
        'batch_size': 16,
        'hidden_size': 256,
        'sequence_length': 30
    },
    'medium_memory': {
        'batch_size': 32,
        'hidden_size': 512,
        'sequence_length': 60
    },
    'high_memory': {
        'batch_size': 64,
        'hidden_size': 768,
        'sequence_length': 120
    }
}
```

### Ограничения по времени обучения
```python
time_constraints = {
    'fast_training': {
        'epochs': 50,
        'early_stopping_patience': 10,
        'model_complexity': 'low'
    },
    'normal_training': {
        'epochs': 150,
        'early_stopping_patience': 30,
        'model_complexity': 'medium'
    },
    'thorough_training': {
        'epochs': 300,
        'early_stopping_patience': 50,
        'model_complexity': 'high'
    }
}
```

## Рекомендации по выбору гиперпараметров

### Для улучшения Directional Accuracy
1. **Больше скрытых нейронов** для лучшего представления паттернов
2. **Высокий dropout** для предотвращения переобучения на шумных данных
3. **Низкий learning rate** для стабильной сходимости
4. **Больше эпох** с ранней остановкой для нахождения оптимальной точки

### Для балансировки скорости и точности
1. **Средний размер модели** (hidden_size: 256-512)
2. **Умеренный dropout** (0.2-0.3)
3. **Batch size 32-64** для эффективного использования GPU
4. **Bidirectional слои** для лучшего контекста

Эти гиперпараметры обеспечат основу для сравнения трех архитектур и дальнейшей оптимизации лучшей модели.
