# План интеграции новых архитектур в существующий код

## Обзор стратегии интеграции

Цель - интегрировать Transformer и CNN+LSTM архитектуры в существующий код stock_predictor_hi.py с минимальными изменениями в основной логике и максималь переиспользуемостью кода.

## 1. Модульная структура кода

### Новая структура файлов
```
stock_prediction/
├── stock_predictor_hi.py          # Основной файл (сохраняется)
├── models/
│   ├── __init__.py
│   ├── base_model.py              # Базовый класс для всех моделей
│   ├── lstm_model.py              # Текущая LSTM модель
│   ├── transformer_model.py       # Transformer модель
│   └── cnn_lstm_model.py          # CNN+LSTM модель
├── data/
│   ├── __init__.py
│   └── data_processor.py         # Обработка данных (вынесено из основного файла)
├── training/
│   ├── __init__.py
│   ├── trainer.py                # Универсальный тренер
│   └── loss_functions.py         # Функции потерь
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py                # Метрики оценки
│   └── visualizer.py             # Визуализация результатов
└── config/
    ├── __init__.py
    └── config.py                 # Конфигурация моделей
```

## 2. Базовый класс для всех моделей

### Абстрактный базовый класс
```python
from abc import ABC, abstractmethod
import torch
import torch.nn as nn

class BaseStockPredictor(ABC, nn.Module):
    """Базовый класс для всех моделей предсказания акций"""

    def __init__(self, input_size=15, seq_length=60):
        super().__init__()
        self.input_size = input_size
        self.seq_length = seq_length

    @abstractmethod
    def forward(self, x):
        """Прямой проход модели"""
        pass

    def get_model_name(self):
        """Возвращает имя модели"""
        return self.__class__.__name__

    def count_parameters(self):
        """Возвращает количество параметров модели"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
```

### Адаптер для существующей LSTM
```python
class ImprovedStockPredictor(BaseStockPredictor):
    """Адаптер для существующей LSTM модели"""

    def __init__(self, input_size=15, hidden_size=512, num_layers=2, dropout=0.1):
        super().__init__(input_size)
        # Существующий код LSTM модели
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout, bidirectional=True)
        # ... остальной код из stock_predictor_hi.py
```

## 3. Универсальный обработчик данных

### Вынос обработки данных в отдельный модуль
```python
# data/data_processor.py
class StockDataProcessor:
    """Универсальный обработчик данных для всех моделей"""

    def __init__(self, symbol='AAPL', period='5y', seq_length=60):
        self.symbol = symbol
        self.period = period
        self.seq_length = seq_length
        self.scaler = None
        self.features = [
            'High', 'Volume', 'Returns', 'EMA_10', 'EMA_50',
            'Volatility', 'Volume_EMA', 'RSI', 'MACD', 'BB_Position',
            'High_Low_Pct', 'Price_Change', 'Volume_Change', 'Momentum', 'MA_Ratio'
        ]

    def prepare_data(self):
        """Подготовка данных (код из stock_predictor_hi.py)"""
        # Существующий код подготовки данных
        pass

    def get_data_loaders(self, batch_size=32):
        """Создание DataLoader для обучения"""
        # Преобразование данных в DataLoader
        pass
```

## 4. Универсальный тренер

### Единый класс для обучения всех моделей
```python
# training/trainer.py
class ModelTrainer:
    """Универсальный тренер для всех моделей"""

    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.best_model_state = None

    def train(self, train_loader, val_loader, epochs=150, patience=30):
        """Универсальный метод обучения"""
        # Адаптивный выбор оптимизатора в зависимости от модели
        optimizer = self._get_optimizer()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.5
        )
        criterion = self._get_criterion()

        # Универсальный цикл обучения
        return self._training_loop(
            train_loader, val_loader, optimizer, scheduler,
            criterion, epochs, patience
        )

    def _get_optimizer(self):
        """Выбор оптимизатора в зависимости от модели"""
        if isinstance(self.model, TransformerModel):
            return optim.AdamW(self.model.parameters(), lr=0.0001, weight_decay=1e-5)
        else:
            return optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)

    def _get_criterion(self):
        """Выбор функции потерь в зависимости от модели"""
        if hasattr(self.model, 'get_loss_function'):
            return self.model.get_loss_function()
        else:
            return DirectionalAccuracyLoss(mse_weight=0.2, da_weight=0.8)
```

## 5. Конфигурация моделей

### Единый файл конфигурации
```python
# config/config.py
class ModelConfig:
    """Конфигурация для всех моделей"""

    # Общие параметры
    INPUT_SIZE = 15
    SEQ_LENGTH = 60
    BATCH_SIZE = 32

    # LSTM параметры
    LSTM_CONFIG = {
        'hidden_size': 512,
        'num_layers': 2,
        'dropout': 0.1
    }

    # Transformer параметры
    TRANSFORMER_CONFIG = {
        'd_model': 512,
        'nhead': 8,
        'num_encoder_layers': 6,
        'dim_feedforward': 2048,
        'dropout': 0.1
    }

    # CNN+LSTM параметры
    CNN_LSTM_CONFIG = {
        'cnn_channels': [32, 64, 128, 256],
        'lstm_hidden_size': 256,
        'lstm_num_layers': 2,
        'dropout': 0.3
    }

    @classmethod
    def get_model_config(cls, model_name):
        """Получение конфигурации для конкретной модели"""
        configs = {
            'LSTM': cls.LSTM_CONFIG,
            'Transformer': cls.TRANSFORMER_CONFIG,
            'CNN_LSTM': cls.CNN_LSTM_CONFIG
        }
        return configs.get(model_name, {})
```

## 6. Фабрика моделей

### Создание моделей через фабрику
```python
# models/__init__.py
from .lstm_model import ImprovedStockPredictor
from .transformer_model import StockTransformer
from .cnn_lstm_model import StockCNNLSTM
from config.config import ModelConfig

class ModelFactory:
    """Фабрика для создания моделей"""

    @staticmethod
    def create_model(model_type, **kwargs):
        """Создание модели указанного типа"""
        config = ModelConfig.get_model_config(model_type)
        config.update(kwargs)  # Переопределение параметров

        if model_type == 'LSTM':
            return ImprovedStockPredictor(**config)
        elif model_type == 'Transformer':
            return StockTransformer(**config)
        elif model_type == 'CNN_LSTM':
            return StockCNNLSTM(**config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    @staticmethod
    def get_available_models():
        """Получение списка доступных моделей"""
        return ['LSTM', 'Transformer', 'CNN_LSTM']
```

## 7. Адаптация основного файла

### Минимальные изменения в stock_predictor_hi.py
```python
# В начале файла добавляем импорты
from models import ModelFactory
from data.data_processor import StockDataProcessor
from training.trainer import ModelTrainer
from evaluation.metrics import calculate_directional_accuracy
from evaluation.visualizer import plot_results_with_changes

# Основная функция с выбором модели
def main(model_type='LSTM'):
    """Основная функция с выбором модели"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Устройство: {device}")

    # Подготовка данных
    print("Загрузка данных...")
    data_processor = StockDataProcessor('AAPL', '5y')
    (X_train, X_val, X_test, y_train, y_val, y_test,
     dates_train, dates_val, dates_test, scaler, data,
     base_train, base_val, base_test) = data_processor.prepare_data()

    # Создание модели через фабрику
    model = ModelFactory.create_model(model_type).to(device)
    print(f"\nМодель создана: {model.get_model_name()}")
    print(f"Параметров: {model.count_parameters():,}")

    # Обучение через универсальный тренер
    trainer = ModelTrainer(model, device)
    train_losses, val_losses, val_das = trainer.train(
        X_train, y_train, X_val, y_val, epochs=150, patience=30
    )

    # Оценка и визуализация
    print("\n" + "="*50)
    print("ТЕСТИРОВАНИЕ НА ТЕСТОВЫХ ДАННЫХ")
    print("="*50)

    real_prices, pred_prices = plot_results_with_changes(
        model, X_test, y_test, dates_test, base_test, data
    )

    da = calculate_directional_accuracy(model, X_test, y_test, base_test)

    print("\n✅ Обучение и тестирование завершено!")

    return model, da

if __name__ == "__main__":
    # Можно выбрать модель через командную строку или аргумент
    import sys
    model_type = sys.argv[1] if len(sys.argv) > 1 else 'LSTM'

    if model_type not in ModelFactory.get_available_models():
        print(f"Доступные модели: {ModelFactory.get_available_models()}")
        sys.exit(1)

    model, da = main(model_type)
```

## 8. Обратная совместимость

### Сохранение оригинального функционала
```python
# В stock_predictor_hi.py оставляем оригинальные функции для обратной совместимости

# Оригинальные функции без изменений
def compute_rsi(prices, window=14):
    # Существующий код без изменений
    pass

def compute_macd(prices, fast=12, slow=26, signal=9):
    # Существующий код без изменений
    pass

def compute_bollinger_bands(prices, window=20):
    # Существующий код без изменений
    pass

class DirectionalAccuracyLoss(nn.Module):
    # Существующий код без изменений
    pass

# Оригинальная функция main для обратной совместимости
def original_main():
    """Оригинальная функция main без изменений"""
    # Существующий код из stock_predictor_hi.py
    pass
```

## 9. Тестирование интеграции

### Unit тесты для компонентов
```python
# tests/test_integration.py
import unittest
import torch
from models import ModelFactory
from data.data_processor import StockDataProcessor

class TestIntegration(unittest.TestCase):

    def setUp(self):
        self.data_processor = StockDataProcessor('AAPL', '2y')  # Короткий период для тестов

    def test_model_creation(self):
        """Тест создания всех моделей"""
        for model_type in ModelFactory.get_available_models():
            model = ModelFactory.create_model(model_type)
            self.assertIsNotNone(model)
            self.assertEqual(model.input_size, 15)

    def test_data_preparation(self):
        """Тест подготовки данных"""
        data = self.data_processor.prepare_data()
        self.assertIsNotNone(data)
        self.assertEqual(len(data), 12)  # Проверка количества возвращаемых значений

    def test_model_forward_pass(self):
        """Тест прямого прохода для всех моделей"""
        # Создание тестовых данных
        batch_size, seq_len, features = 4, 60, 15
        x = torch.randn(batch_size, seq_len, features)

        for model_type in ModelFactory.get_available_models():
            model = ModelFactory.create_model(model_type)
            output = model(x)
            self.assertIsNotNone(output)
            self.assertEqual(output.shape[0], batch_size)

if __name__ == '__main__':
    unittest.main()
```

## 10. План миграции

### Этапы внедрения
1. **Этап 1**: Создание модульной структуры
   - Создание папок и файлов
   - Перенос существующего кода в модули

2. **Этап 2**: Реализация новых моделей
   - Реализация Transformer модели
   - Реализация CNN+LSTM модели

3. **Этап 3**: Интеграция и тестирование
   - Интеграция новых моделей в основной код
   - Тестирование обратной совместимости

4. **Этап 4**: Оптимизация и документация
   - Оптимизация производительности
   - Обновление документации

### Преимущества такого подхода
1. **Минимальные изменения** в существующем коде
2. **Обратная совместимость** с оригинальным функционалом
3. **Модульность** и переиспользуемость кода
4. **Легкость добавления** новых моделей в будущем
5. **Унифицированный интерфейс** для всех моделей

Этот план позволяет плавно интегрировать новые архитектуры, сохраняя при этом функциональность существующего кода и обеспечивая основу для будущего расширения.
