# План тестирования и оценки производительности

## Цель тестирования

Обеспечить объективную и всестороннюю оценку производительности трех архитектур (LSTM, Transformer, CNN+LSTM) с фокусом на улучшение показателя Directional Accuracy (DA).

## 1. Метрики оценки

### Основные метрики
```python
# evaluation/metrics.py
class ModelMetrics:
    """Класс для расчета метрик производительности моделей"""

    @staticmethod
    def directional_accuracy(y_true, y_pred, threshold=0.001):
        """
        Расчет Directional Accuracy

        Args:
            y_true: реальные изменения цен
            y_pred: предсказанные изменения цен
            threshold: порог для определения значимого изменения

        Returns:
            DA: directional accuracy в процентах
        """
        true_directions = np.sign(y_true)
        pred_directions = np.sign(y_pred)

        # Исключаем незначительные изменения
        mask = (np.abs(y_true) > threshold)

        if np.sum(mask) > 0:
            accuracy = np.mean(true_directions[mask] == pred_directions[mask])
            return accuracy * 100
        return 0

    @staticmethod
    def weighted_directional_accuracy(y_true, y_pred, weights=None):
        """
        Взвешенная Directional Accuracy с учетом величины изменений
        """
        if weights is None:
            weights = np.abs(y_true)  # Вес пропорционален величине изменения

        correct_predictions = (np.sign(y_true) == np.sign(y_pred)).astype(float)
        weighted_accuracy = np.sum(correct_predictions * weights) / np.sum(weights)

        return weighted_accuracy * 100

    @staticmethod
    def precision_recall_f1(y_true, y_pred, threshold=0.001):
        """
        Расчет Precision, Recall, F1 для задачи классификации направления
        """
        true_directions = np.sign(y_true)
        pred_directions = np.sign(y_pred)

        # Преобразуем в бинарную классификацию (вверх vs вниз)
        true_up = (true_directions > 0)
        pred_up = (pred_directions > 0)

        # Метрики для предсказания роста
        precision_up = precision_score(true_up, pred_up)
        recall_up = recall_score(true_up, pred_up)
        f1_up = f1_score(true_up, pred_up)

        # Метрики для предсказания падения
        true_down = (true_directions < 0)
        pred_down = (pred_directions < 0)

        precision_down = precision_score(true_down, pred_down)
        recall_down = recall_score(true_down, pred_down)
        f1_down = f1_score(true_down, pred_down)

        return {
            'up': {'precision': precision_up, 'recall': recall_up, 'f1': f1_up},
            'down': {'precision': precision_down, 'recall': recall_down, 'f1': f1_down}
        }
```

### Дополнительные метрики
```python
class AdditionalMetrics:
    """Дополнительные метрики для комплексной оценки"""

    @staticmethod
    def sharpe_ratio(returns, risk_free_rate=0.02):
        """
        Расчет коэффициента Шарпа для торговой стратегии на основе предсказаний
        """
        excess_returns = returns - risk_free_rate / 252  # Дневная ставка
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)

    @staticmethod
    def max_drawdown(returns):
        """
        Расчет максимальной просадки
        """
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown)

    @staticmethod
    def information_ratio(portfolio_returns, benchmark_returns):
        """
        Расчет Information Ratio
        """
        excess_returns = portfolio_returns - benchmark_returns
        tracking_error = np.std(excess_returns)
        return np.mean(excess_returns) / tracking_error if tracking_error != 0 else 0
```

## 2. Структура тестирования

### Класс для организации тестов
```python
# testing/model_tester.py
class ModelTester:
    """Класс для тестирования моделей"""

    def __init__(self, models, data_processor, device='cpu'):
        self.models = models
        self.data_processor = data_processor
        self.device = device
        self.results = {}

    def run_comprehensive_test(self, num_runs=5):
        """
        Комплексное тестирование всех моделей
        """
        for model_name, model in self.models.items():
            print(f"\nТестирование модели: {model_name}")
            model_results = []

            for run in range(num_runs):
                print(f"  Запуск {run + 1}/{num_runs}")

                # Инициализация модели
                model_instance = model().to(self.device)

                # Обучение и оценка
                run_results = self._single_run_test(model_instance)
                model_results.append(run_results)

            self.results[model_name] = model_results

        return self._analyze_results()

    def _single_run_test(self, model):
        """
        Одиночный запуск теста для модели
        """
        # Подготовка данных
        train_loader, val_loader, test_loader = self.data_processor.get_data_loaders()

        # Обучение модели
        trainer = ModelTrainer(model, self.device)
        training_metrics = trainer.train(train_loader, val_loader)

        # Оценка на тестовых данных
        test_metrics = self._evaluate_model(model, test_loader)

        # Дополнительные тесты
        robustness_metrics = self._test_robustness(model, test_loader)

        return {
            'training': training_metrics,
            'test': test_metrics,
            'robustness': robustness_metrics,
            'model_params': model.count_parameters()
        }
```

### Специализированные тесты
```python
class SpecializedTests:
    """Специализированные тесты для оценки моделей"""

    @staticmethod
    def volatility_test(model, test_data, volatility_thresholds=[0.01, 0.02, 0.03]):
        """
        Тест производительности в периоды разной волатильности
        """
        results = {}

        for threshold in volatility_thresholds:
            # Фильтрация данных по волатильности
            high_vol_mask = test_data['volatility'] > threshold
            low_vol_mask = test_data['volatility'] <= threshold

            # Оценка на высоковолатильных данных
            high_vol_da = ModelMetrics.directional_accuracy(
                test_data['y_true'][high_vol_mask],
                test_data['y_pred'][high_vol_mask]
            )

            # Оценка на низковолатильных данных
            low_vol_da = ModelMetrics.directional_accuracy(
                test_data['y_true'][low_vol_mask],
                test_data['y_pred'][low_vol_mask]
            )

            results[f'vol_threshold_{threshold}'] = {
                'high_volatility_da': high_vol_da,
                'low_volatility_da': low_vol_da,
                'performance_gap': abs(high_vol_da - low_vol_da)
            }

        return results

    @staticmethod
    def trend_test(model, test_data, trend_windows=[5, 10, 20]):
        """
        Тест производительности на трендовых и боковых рынках
        """
        results = {}

        for window in trend_windows:
            # Определение тренда
            price_trend = test_data['prices'].rolling(window).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0]
            )

            # Классификация рынков
            strong_trend_mask = np.abs(price_trend) > price_trend.std()
            sideways_mask = np.abs(price_trend) <= price_trend.std()

            # Оценка производительности
            trend_da = ModelMetrics.directional_accuracy(
                test_data['y_true'][strong_trend_mask],
                test_data['y_pred'][strong_trend_mask]
            )

            sideways_da = ModelMetrics.directional_accuracy(
                test_data['y_true'][sideways_mask],
                test_data['y_pred'][sideways_mask]
            )

            results[f'trend_window_{window}'] = {
                'trend_market_da': trend_da,
                'sideways_market_da': sideways_da,
                'trend_advantage': trend_da - sideways_da
            }

        return results

    @staticmethod
    def temporal_stability_test(model, test_data, time_windows=[30, 60, 90]):
        """
        Тест стабильности производительности во времени
        """
        results = {}

        for window in time_windows:
            # Разделение данных на временные окна
            window_size = len(test_data) // window
            da_values = []

            for i in range(window):
                start_idx = i * window_size
                end_idx = (i + 1) * window_size if i < window - 1 else len(test_data)

                window_da = ModelMetrics.directional_accuracy(
                    test_data['y_true'][start_idx:end_idx],
                    test_data['y_pred'][start_idx:end_idx]
                )
                da_values.append(window_da)

            results[f'time_window_{window}'] = {
                'mean_da': np.mean(da_values),
                'std_da': np.std(da_values),
                'min_da': np.min(da_values),
                'max_da': np.max(da_values),
                'stability_score': np.mean(da_values) / (np.std(da_values) + 1e-8)
            }

        return results
```

## 3. Статистический анализ

### Статистические тесты
```python
# testing/statistical_analysis.py
class StatisticalAnalyzer:
    """Класс для статистического анализа результатов"""

    @staticmethod
    def compare_models_da(results, alpha=0.05):
        """
        Статистическое сравнение Directional Accuracy между моделями
        """
        model_names = list(results.keys())
        da_values = {name: [run['test']['da'] for run in results[name]]
                    for name in model_names}

        # Попарное сравнение
        comparisons = {}
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names[i+1:], i+1):
                # T-тест для независимых выборок
                t_stat, p_value = stats.ttest_ind(
                    da_values[model1], da_values[model2]
                )

                # Эффект размера (Cohen's d)
                pooled_std = np.sqrt(
                    ((len(da_values[model1]) - 1) * np.var(da_values[model1]) +
                     (len(da_values[model2]) - 1) * np.var(da_values[model2])) /
                    (len(da_values[model1]) + len(da_values[model2]) - 2)
                )
                cohens_d = (np.mean(da_values[model1]) - np.mean(da_values[model2])) / pooled_std

                comparisons[f'{model1}_vs_{model2}'] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < alpha,
                    'cohens_d': cohens_d,
                    'effect_size': 'small' if abs(cohens_d) < 0.2 else
                                  'medium' if abs(cohens_d) < 0.8 else 'large'
                }

        return comparisons

    @staticmethod
    def confidence_intervals(results, confidence=0.95):
        """
        Расчет доверительных интервалов для метрик
        """
        intervals = {}

        for model_name, model_results in results.items():
            da_values = [run['test']['da'] for run in model_results]

            mean_da = np.mean(da_values)
            std_da = np.std(da_values)
            n = len(da_values)

            # Расчет доверительного интервала
            t_critical = stats.t.ppf((1 + confidence) / 2, n - 1)
            margin_error = t_critical * (std_da / np.sqrt(n))

            intervals[model_name] = {
                'mean': mean_da,
                'lower_bound': mean_da - margin_error,
                'upper_bound': mean_da + margin_error,
                'margin_error': margin_error,
                'confidence_level': confidence
            }

        return intervals
```

## 4. Визуализация результатов

### Класс для визуализации
```python
# evaluation/visualizer.py
class ResultsVisualizer:
    """Класс для визуализации результатов тестирования"""

    @staticmethod
    def plot_da_comparison(results, save_path='da_comparison.png'):
        """
        Визуализация сравнения Directional Accuracy
        """
        model_names = list(results.keys())
        da_means = [np.mean([run['test']['da'] for run in results[name]])
                   for name in model_names]
        da_stds = [np.std([run['test']['da'] for run in results[name]])
                  for name in model_names]

        plt.figure(figsize=(12, 8))

        # Столбчатая диаграмма с ошибками
        bars = plt.bar(model_names, da_means, yerr=da_stds,
                      capsize=5, alpha=0.7, color=['blue', 'green', 'red'])

        # Добавление значений на столбцы
        for bar, mean in zip(bars, da_means):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + da_stds[0],
                    f'{mean:.2f}%', ha='center', va='bottom', fontsize=12)

        plt.title('Сравнение Directional Accuracy моделей', fontsize=16)
        plt.ylabel('Directional Accuracy (%)', fontsize=14)
        plt.xlabel('Модель', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, max(da_means) + max(da_stds) + 5)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_training_curves(results, save_path='training_curves.png'):
        """
        Визуализация кривых обучения
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        for model_name, model_results in results.items():
            # Берем первый запуск для визуализации
            train_losses = model_results[0]['training']['train_losses']
            val_losses = model_results[0]['training']['val_losses']
            val_das = model_results[0]['training']['val_das']

            epochs = range(1, len(train_losses) + 1)

            # Потери на обучении
            axes[0, 0].plot(epochs, train_losses, label=f'{model_name} Train')
            axes[0, 0].set_title('Потери на обучении')
            axes[0, 0].set_xlabel('Эпоха')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            # Потери на валидации
            axes[0, 1].plot(epochs, val_losses, label=f'{model_name} Val')
            axes[0, 1].set_title('Потери на валидации')
            axes[0, 1].set_xlabel('Эпоха')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

            # DA на валидации
            axes[1, 0].plot(epochs, val_das, label=f'{model_name}')
            axes[1, 0].set_title('Directional Accuracy на валидации')
            axes[1, 0].set_xlabel('Эпоха')
            axes[1, 0].set_ylabel('DA (%)')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

            # Сравнение финальных метрик
            final_da = model_results[0]['test']['da']
            axes[1, 1].bar(model_name, final_da, alpha=0.7)
            axes[1, 1].set_title('Финальная DA на тесте')
            axes[1, 1].set_ylabel('DA (%)')
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
```

## 5. Автоматизация тестирования

### Основной скрипт для запуска тестов
```python
# testing/run_tests.py
def main():
    """Основная функция для запуска всех тестов"""

    # Инициализация
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_processor = StockDataProcessor('AAPL', '5y')

    # Создание моделей
    models = {
        'LSTM': lambda: ModelFactory.create_model('LSTM'),
        'Transformer': lambda: ModelFactory.create_model('Transformer'),
        'CNN_LSTM': lambda: ModelFactory.create_model('CNN_LSTM')
    }

    # Запуск тестов
    tester = ModelTester(models, data_processor, device)
    results = tester.run_comprehensive_test(num_runs=5)

    # Статистический анализ
    analyzer = StatisticalAnalyzer()
    comparisons = analyzer.compare_models_da(results)
    intervals = analyzer.confidence_intervals(results)

    # Визуализация
    visualizer = ResultsVisualizer()
    visualizer.plot_da_comparison(results)
    visualizer.plot_training_curves(results)

    # Генерация отчета
    report_generator = ReportGenerator()
    report_generator.generate_comprehensive_report(results, comparisons, intervals)

    return results

if __name__ == "__main__":
    results = main()
```

## 6. Критерии успеха

### Количественные критерии
1. **Улучшение DA**: Минимальное улучшение на 5% по сравнению с базовой LSTM
2. **Статистическая значимость**: p < 0.05 в статистических тестах
3. **Стабильность**: Стандартное отклонение DA менее 3% между запусками

### Качественные критерии
1. **Устойчивость к волатильности**: Производительность не должна значительно падать в периоды высокой волатильности
2. **Адаптивность к трендам**: Модель должна работать лучше на трендовых рынках
3. **Временная стабильность**: Производительность должна быть стабильной во времени

Этот план обеспечивает всестороннюю оценку производительности моделей с фокусом на улучшение Directional Accuracy и предоставляет объективные критерии для выбора лучшей архитектуры.
