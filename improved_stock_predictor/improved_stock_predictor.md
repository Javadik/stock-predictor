# Обзор модели ImprovedStockPredictor

## Архитектура модели
Улучшенная модель представляет собой bidirectional LSTM-сеть с 2 слоями и 128 скрытыми нейронами. Архитектура включает:
- Bidirectional LSTM слой с 15 входными признаками, 128 скрытыми состояниями и 2 слоями
- Два линейных слоя с ReLU активацией
- Dropout 0.3 для регуляризации
- Всего 576,769 параметров (в 2.8 раза больше, чем в предыдущей модели)

## Используемые фичи (15 признаков)
1. [`Close`](improved_stock_predictor/improved_stock_predictor.py:76) - цена закрытия
2. [`Volume`](improved_stock_predictor/improved_stock_predictor.py:76) - объем торгов
3. [`Returns`](improved_stock_predictor/improved_stock_predictor.py:39) - процентное изменение цены
4. [`EMA_10`](improved_stock_predictor/improved_stock_predictor.py:40) - экспоненциальное скользящее среднее за 10 дней
5. [`EMA_50`](improved_stock_predictor/improved_stock_predictor.py:41) - экспоненциальное скользящее среднее за 50 дней
6. [`Volatility`](improved_stock_predictor/improved_stock_predictor.py:42) - волатильность (стандартное отклонение за 20 дней)
7. [`Volume_EMA`](improved_stock_predictor/improved_stock_predictor.py:43) - EMA объема
8. [`RSI`](improved_stock_predictor/improved_stock_predictor.py:46) - индекс относительной силы
9. [`MACD`](improved_stock_predictor/improved_stock_predictor.py:47) - схождение/расхождение скользящих средних
10. [`BB_Position`](improved_stock_predictor/improved_stock_predictor.py:77) - позиция внутри Bollinger Bands
11. [`High_Low_Pct`](improved_stock_predictor/improved_stock_predictor.py:53) - отношение (High-Low)/Close
12. [`Price_Change`](improved_stock_predictor/improved_stock_predictor.py:54) - внутридневное изменение цены
13. [`Volume_Change`](improved_stock_predictor/improved_stock_predictor.py:55) - изменение объема
14. [`Momentum`](improved_stock_predictor/improved_stock_predictor.py:56) - 10-дневный моментум
15. [`MA_Ratio`](improved_stock_predictor/improved_stock_predictor.py:61) - отношение краткосрочной к долгосрочной скользящей средней

## Результаты модели

### Метрики качества:
- MSE: $12.22
- MAE: $2.43
- MAPE: 1.01%
- Directional Accuracy: 46.9% (точность предсказания направления изменения цены)
- Точность для роста: 41.5%
- Точность для падения: 53.3%

### Данные:
- Обучающая выборка: 678 дат (70%)
- Валидационная выборка: 97 дат (15%)
- Тестовая выборка: 98 дат (15%)
- Период данных: 2021-09-14 - 2025-11-25
- Длина последовательности: 60 дней

### Особенности:
- Используется bidirectional LSTM для лучшего понимания контекста
- Специальная функция потерь (DirectionalAccuracyLoss), штрафующая за неправильное направление
- Ранняя остановка с параметром patience=30
- Обучение остановлено на 59-й эпохе, восстановлены веса с 29-й эпохи
- Модель склонна занижать предсказания (средняя ошибка - $0.74)
- 54.1% предсказаний занижены

Хотя улучшенная модель использует больше признаков и более сложную архитектуру, её точность в предсказании направления (46.9%) оказалась ниже, чем у предыдущей модели (52.0%). Это может быть связано с переобучением из-за большего количества параметров или с проблемами в функции потерь, которая слишком сильно фокусируется на направлении, жертвуя точностью предсказания величины изменений.
