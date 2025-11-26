# Стратегия автоматической селекции признаков на основе анализа важности

## Обзор подхода

После проведения анализа важности признаков с помощью различных методов (attention weights, gradient-based, permutation importance), необходимо создать механизм автоматической селекции наиболее важных признаков для сокращения размерности модели без потери производительности.

## Критерии селекции признаков

### 1. Пороговые значения важности

#### Абсолютные пороги:
- **Высокая важность**: > 8% от общей важности
- **Средняя важность**: 4-8% от общей важности
- **Низкая важность**: < 4% от общей важности

#### Относительные пороги:
- **Топ-N признаков**: Выбор N наиболее важных признаков
- **Кумулятивный порог**: Включение признаков до достижения 80-90% кумулятивной важности

### 2. Стабильность оценок

#### Временная стабильность:
- Анализ важности признаков на разных временных периодах
- Исключение признаков с высокой вариативностью важности

#### Методологическая стабильность:
- Согласованность оценок между разными методами анализа
- Приоритет признаков, которые важны по нескольким методам

### 3. Корреляционный анализ

#### Внутригрупповая корреляция:
- Удаление сильно коррелированных признаков (r > 0.8)
- Оставление наиболее важного из коррелирующей группы

#### Мультиколлинеарность:
- Расчет VIF (Variance Inflation Factor)
- Исключение признаков с VIF > 5

## Алгоритм селекции признаков

### Шаг 1: Первичный отбор по важности

```python
def primary_selection_by_importance(importance_scores, threshold=0.04):
    """
    Первичный отбор признаков по порогу важности

    Args:
        importance_scores: массив важности признаков (нормализованный)
        threshold: порог важности (по умолчанию 4%)

    Returns:
        selected_indices: индексы отобранных признаков
        selected_scores: важность отобранных признаков
    """
    selected_indices = np.where(importance_scores >= threshold)[0]
    selected_scores = importance_scores[selected_indices]

    return selected_indices, selected_scores

def cumulative_importance_selection(importance_scores, cumulative_threshold=0.85):
    """
    Отбор признаков по кумулятивной важности

    Args:
        importance_scores: массив важности признаков (отсортированный по убыванию)
        cumulative_threshold: порог кумулятивной важности (по умолчанию 85%)

    Returns:
        selected_indices: индексы отобранных признаков
    """
    sorted_indices = np.argsort(importance_scores)[::-1]
    sorted_scores = importance_scores[sorted_indices]

    cumulative_sum = np.cumsum(sorted_scores)
    n_selected = np.argmax(cumulative_sum >= cumulative_threshold) + 1

    return sorted_indices[:n_selected]
```

### Шаг 2: Корреляционный анализ

```python
def correlation_based_selection(data, selected_indices, correlation_threshold=0.8):
    """
    Удаление сильно коррелированных признаков

    Args:
        data: исходные данные (samples × features)
        selected_indices: индексы предварительно отобранных признаков
        correlation_threshold: порог корреляции

    Returns:
        final_indices: финальные индексы признаков
    """
    selected_data = data[:, selected_indices]
    correlation_matrix = np.corrcoef(selected_data.T)

    # Находим группы сильно коррелированных признаков
    groups = []
    used = set()

    for i in range(len(selected_indices)):
        if i in used:
            continue

        group = [i]
        for j in range(i+1, len(selected_indices)):
            if j not in used and abs(correlation_matrix[i, j]) > correlation_threshold:
                group.append(j)
                used.add(j)

        groups.append(group)
        used.add(i)

    # Из каждой группы оставляем наиболее важный признак
    final_indices = []
    for group in groups:
        if len(group) == 1:
            final_indices.append(selected_indices[group[0]])
        else:
            # Здесь нужна дополнительная информация о важности
            # Пока просто берем первый признак из группы
            final_indices.append(selected_indices[group[0]])

    return np.array(final_indices)
```

### Шаг 3: Стабильность оценок

```python
def stability_analysis(importance_results_list, stability_threshold=0.7):
    """
    Анализ стабильности важности признаков

    Args:
        importance_results_list: список результатов анализа важности на разных периодах
        stability_threshold: порог стабильности

    Returns:
        stable_indices: индексы стабильных признаков
    """
    # Конвертируем в numpy массив
    all_importance = np.array(importance_results_list)

    # Рассчитываем коэффициент вариации для каждого признака
    mean_importance = np.mean(all_importance, axis=0)
    std_importance = np.std(all_importance, axis=0)

    # Избегаем деления на ноль
    cv = np.where(mean_importance > 0, std_importance / mean_importance, 0)

    # Отбираем стабильные признаки (низкая вариативность + достаточная важность)
    stable_mask = (cv <= (1 - stability_threshold)) & (mean_importance >= 0.02)
    stable_indices = np.where(stable_mask)[0]

    return stable_indices
```

### Шаг 4: Комплексный алгоритм селекции

```python
class FeatureSelector:
    def __init__(self, importance_threshold=0.04, correlation_threshold=0.8,
                 cumulative_threshold=0.85, stability_threshold=0.7):
        self.importance_threshold = importance_threshold
        self.correlation_threshold = correlation_threshold
        self.cumulative_threshold = cumulative_threshold
        self.stability_threshold = stability_threshold

    def select_features(self, data, importance_scores, importance_history=None):
        """
        Комплексная селекция признаков

        Args:
            data: исходные данные
            importance_scores: текущие оценки важности
            importance_history: история оценок важности (для анализа стабильности)

        Returns:
            selected_features: информация об отобранных признаках
            selection_report: отчет о селекции
        """
        # Шаг 1: Первичный отбор по важности
        primary_indices, primary_scores = primary_selection_by_importance(
            importance_scores, self.importance_threshold
        )

        # Если слишком мало признаков, используем кумулятивный подход
        if len(primary_indices) < 8:
            primary_indices = cumulative_importance_selection(
                importance_scores, self.cumulative_threshold
            )
            primary_scores = importance_scores[primary_indices]

        # Шаг 2: Корреляционный анализ
        if len(primary_indices) > 1:
            final_indices = correlation_based_selection(
                data, primary_indices, self.correlation_threshold
            )
        else:
            final_indices = primary_indices

        # Шаг 3: Анализ стабильности (если доступна история)
        if importance_history is not None:
            stable_indices = stability_analysis(
                importance_history, self.stability_threshold
            )
            # Пересечение с текущим отбором
            final_indices = np.intersect1d(final_indices, stable_indices)

        # Создание отчета
        selection_report = {
            'original_count': len(importance_scores),
            'primary_selected': len(primary_indices),
            'final_selected': len(final_indices),
            'reduction_ratio': 1 - len(final_indices) / len(importance_scores),
            'selected_indices': final_indices,
            'selected_scores': importance_scores[final_indices]
        }

        return final_indices, selection_report
```

## Адаптивная селекция

### 1. Динамические пороги

```python
def adaptive_threshold_selection(importance_scores, target_features_range=(8, 12)):
    """
    Адаптивный подбор порога для целевого количества признаков

    Args:
        importance_scores: массив важности признаков
        target_features_range: целевой диапазон количества признаков

    Returns:
        selected_indices: индексы отобранных признаков
        adaptive_threshold: подобранный порог
    """
    sorted_scores = np.sort(importance_scores)[::-1]

    # Ищем порог, дающий нужное количество признаков
    for threshold in np.linspace(0.01, 0.1, 100):
        selected = np.sum(importance_scores >= threshold)
        if target_features_range[0] <= selected <= target_features_range[1]:
            break

    selected_indices = np.where(importance_scores >= threshold)[0]

    return selected_indices, threshold
```

### 2. Рыночно-зависимая селекция

```python
def market_regime_feature_selection(importance_scores_by_regime, current_regime):
    """
    Селекция признаков в зависимости от рыночного режима

    Args:
        importance_scores_by_regime: словарь важности признаков по режимам
        current_regime: текущий рыночный режим

    Returns:
        selected_indices: индексы отобранных признаков
    """
    current_importance = importance_scores_by_regime[current_regime]

    # Учитываем важность в других режимах с меньшим весом
    combined_importance = current_importance.copy()

    for regime, importance in importance_scores_by_regime.items():
        if regime != current_regime:
            combined_importance += 0.3 * importance  # Вес 30% для других режимов

    # Нормализуем
    combined_importance = combined_importance / np.sum(combined_importance)

    # Применяем стандартную селекцию
    return cumulative_importance_selection(combined_importance, 0.8)
```

## Валидация селекции

### 1. Кросс-валидация по времени

```python
def time_series_cross_validation(data, targets, selected_features, n_splits=5):
    """
    Валидация качества селекции признаков с помощью кросс-валидации по времени

    Args:
        data: исходные данные
        targets: целевые переменные
        selected_features: индексы отобранных признаков
        n_splits: количество фолдов

    Returns:
        cv_scores: оценки качества на фолдах
    """
    fold_size = len(data) // n_splits
    cv_scores = []

    for i in range(n_splits):
        # Разделение на train/val с учетом временной структуры
        train_end = (i + 1) * fold_size
        val_end = min((i + 2) * fold_size, len(data))

        train_data = data[:train_end, selected_features]
        train_targets = targets[:train_end]
        val_data = data[train_end:val_end, selected_features]
        val_targets = targets[train_end:val_end]

        # Обучение и оценка простой модели
        score = train_and_evaluate_simple_model(train_data, train_targets, val_data, val_targets)
        cv_scores.append(score)

    return np.array(cv_scores)
```

### 2. Сравнение с базовой моделью

```python
def compare_with_baseline(data, targets, selected_features, baseline_features=None):
    """
    Сравнение производительности модели с отобранными признаками и базовой модели

    Args:
        data: исходные данные
        targets: целевые переменные
        selected_features: индексы отобранных признаков
        baseline_features: индексы базовых признаков (если None, то все признаки)

    Returns:
        comparison_results: результаты сравнения
    """
    if baseline_features is None:
        baseline_features = np.arange(data.shape[1])

    # Оценка модели с отобранными признаками
    selected_score = evaluate_model_with_features(data, targets, selected_features)

    # Оценка базовой модели
    baseline_score = evaluate_model_with_features(data, targets, baseline_features)

    # Расчет улучшения
    improvement = (selected_score - baseline_score) / baseline_score * 100

    comparison_results = {
        'selected_score': selected_score,
        'baseline_score': baseline_score,
        'improvement_percent': improvement,
        'feature_reduction': 1 - len(selected_features) / len(baseline_features)
    }

    return comparison_results
```

## Практическое применение

### 1. Интеграция в существующий код

```python
def integrate_feature_selection():
    """
    Интеграция селекции признаков в существующий pipeline
    """
    # Загрузка данных и модели
    (X_train, X_val, X_test, y_train, y_val, y_test,
     dates_train, dates_val, dates_test, scaler, data,
     base_train, base_val, base_test) = prepare_data('AAPL', period='5y')

    feature_names = ['High', 'Volume', 'Returns', 'EMA_10', 'EMA_50',
                    'Volatility', 'Volume_EMA', 'RSI', 'MACD', 'BB_Position',
                    'High_Low_Pct', 'Price_Change', 'Volume_Change', 'Momentum', 'MA_Ratio',
                    'Adaptive_Volatility', 'Trend_Strength', 'Price_Strength', 'Price_to_VWAP', 'Volatility_Momentum']

    # Анализ важности признаков
    analyzer = ComprehensiveFeatureAnalyzer(model, feature_names, device)
    results = analyzer.analyze_all_methods(train_loader, directional_accuracy_metric)

    # Селекция признаков
    selector = FeatureSelector(
        importance_threshold=0.04,
        correlation_threshold=0.8,
        cumulative_threshold=0.85
    )

    selected_indices, selection_report = selector.select_features(
        X_train.numpy(), results['combined']
    )

    print(f"Селекция признаков завершена:")
    print(f"  Исходное количество: {selection_report['original_count']}")
    print(f"  Отобрано: {selection_report['final_selected']}")
    print(f"  Сокращение: {selection_report['reduction_ratio']:.1%}")

    # Создание новых данных с отобранными признаками
    X_train_selected = X_train[:, :, selected_indices]
    X_val_selected = X_val[:, :, selected_indices]
    X_test_selected = X_test[:, :, selected_indices]

    selected_feature_names = [feature_names[i] for i in selected_indices]

    return X_train_selected, X_val_selected, X_test_selected, selected_feature_names, selection_report
```

Эта стратегия обеспечивает систематический подход к селекции признаков, учитывающий важность, корреляции и стабильность оценок, что позволяет сократить количество признаков без потери производительности модели.
