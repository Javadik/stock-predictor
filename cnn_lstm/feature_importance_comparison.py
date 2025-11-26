import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import time

# Импортируем необходимые компоненты
from stock_predictor_hi import ImprovedStockPredictor, prepare_data
from feature_importance_analyzer import ComprehensiveFeatureAnalyzer, directional_accuracy_metric
from feature_selector import FeatureSelector
from reduced_stock_predictor import ReducedStockPredictor, prepare_reduced_data, train_reduced_model

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
    print("Загрузка данных...")
    (X_train_full, X_val_full, X_test_full, y_train_full, y_val_full, y_test_full,
     dates_train_full, dates_val_full, dates_test_full, scaler_full, data_full,
     base_train_full, base_val_full, base_test_full) = prepare_data('AAPL', period='3y')

    # Создание и обучение полной модели
    full_model = ImprovedStockPredictor(input_size=20, hidden_size=768, num_layers=3, dropout=0.2).to(device)

    start_time = time.time()
    # Используем функцию обучения из stock_predictor_hi
    from stock_predictor_hi import train_model_with_early_stopping
    # Обучаем модель с возвращением потерь
    from stock_predictor_hi import train_model_with_early_stopping
    train_losses_full, val_losses_full, val_das_full = train_model_with_early_stopping(
        full_model, X_train_full, y_train_full, X_val_full, y_val_full, epochs=10, patience=10
    )
    training_time_full = time.time() - start_time

    # Тестирование полной модели
    from stock_predictor_hi import calculate_directional_accuracy
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

    # Создание отчета
    report = analyzer.create_comprehensive_report(importance_results)
    print("\n=== ОТЧЕТ О ВАЖНОСТИ ПРИЗНАКОВ ===")
    print(report.to_string(index=False))

    # Визуализация
    analyzer.attention_analyzer.visualize_importance(
        importance_results['combined'],
        "Комбинированная важность признаков"
    )

    # Сохранение результатов
    report.to_csv('feature_importance_report.csv', index=False)

    # Селекция признаков
    selector = FeatureSelector(importance_threshold=0.04, correlation_threshold=0.8)
    selected_indices, selection_report = selector.select_features(
        X_train_full.numpy(), importance_results['combined']
    )

    print(f"\nРезультаты селекции:")
    print(f" Исходное количество: {selection_report['original_count']}")
    print(f"  Отобрано: {selection_report['final_selected']}")
    print(f"  Сокращение: {selection_report['reduction_ratio']:.1%}")
    print(f" Отобранные признаки: {[feature_names[i] for i in selected_indices]}")

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
        reduced_model, X_train_reduced, y_train_reduced, X_val_reduced, y_val_reduced, epochs=50, patience=15
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
    print(f" Сокращение параметров: {param_reduction:.1f}%")

    print(f"\nВремя обучения:")
    print(f" Полная модель: {full_time:.2f} сек")
    print(f"  Сокращенная модель: {reduced_time:.2f} сек")
    print(f"  Ускорение обучения: {time_reduction:.1f}%")

    print(f"\nDirectional Accuracy:")
    print(f"  Полная модель: {full_da:.3f} ({full_da*100:.1f}%)")
    print(f"  Сокращенная модель: {reduced_da:.3f} ({reduced_da*100:.1f}%)")
    print(f"  Изменение DA: {da_change:+.1f}%")

    # 5. Визуализация результатов
    create_comparison_plots(results, importance_results, selected_indices, feature_names)

    return results, importance_results, selected_indices

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

def main():
    """
    Основная функция для запуска анализа
    """
    print("Запуск комплексного анализа важности признаков и тестирования сокращенной модели...")

    results, importance_results, selected_indices = comprehensive_model_comparison()

    print("\nАнализ завершен!")
    print("Созданы файлы:")
    print("- feature_importance_report.csv: отчет о важности признаков")
    print("- feature_importance.png: визуализация важности признаков")
    print("- model_comparison.png: сравнение моделей")

    return results, importance_results, selected_indices

if __name__ == "__main__":
    results, importance_results, selected_indices = main()
