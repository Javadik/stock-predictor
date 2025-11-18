import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import time

class AttentionFeatureAnalyzer:
    def __init__(self, model, feature_names, device='cpu'):
        self.model = model
        self.feature_names = feature_names
        self.device = device
        self.attention_data = []

    def extract_attention_weights(self, data_loader):
        """Извлечение attention weights из модели"""
        self.model.eval()
        all_attention_weights = []
        all_multi_attention = []
        all_enhanced_features = []

        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x = batch_x.to(self.device)

                # Получаем attention weights
                _, attention_dict = self.model(batch_x, return_attention=True)

                # Сохраняем разные типы attention
                all_attention_weights.append(attention_dict['feature_attention'].cpu().numpy())
                all_multi_attention.append(attention_dict['multi_head_attention'].cpu().numpy())
                all_enhanced_features.append(attention_dict['enhanced_features'].cpu().numpy())

        # Объединяем все батчи
        self.attention_weights = np.concatenate(all_attention_weights, axis=0)
        self.multi_attention = np.concatenate(all_multi_attention, axis=0)
        self.enhanced_features = np.concatenate(all_enhanced_features, axis=0)

        return self.attention_weights

    def calculate_feature_importance(self):
        """Расчет важности признаков на основе attention weights"""
        # Усредняем attention weights по временным шагам и батчам
        # attention_weights shape: (batch_size, seq_len, 1)

        # Важность на основе feature attention
        feature_attention_importance = np.mean(self.attention_weights, axis=(0, 1))

        # Важность на основе enhanced features
        # enhanced_features shape: (batch_size, seq_len, feature_dim)
        # Нужно проецировать на исходные признаки

        # Создаем проекционную матрицу для enhanced features
        enhanced_importance = np.mean(np.abs(self.enhanced_features), axis=(0, 1))

        # Нормализуем важность
        feature_attention_importance = feature_attention_importance / np.sum(feature_attention_importance)
        enhanced_importance = enhanced_importance / np.sum(enhanced_importance)

        # Комбинируем разные типы важности
        combined_importance = 0.7 * feature_attention_importance + 0.3 * enhanced_importance

        return {
            'attention_based': feature_attention_importance.flatten(),
            'enhanced_based': enhanced_importance,
            'combined': combined_importance.flatten()
        }

    def visualize_importance(self, importance_scores, title="Feature Importance"):
        """Визуализация важности признаков в процентах"""
        plt.figure(figsize=(12, 8))

        # Преобразуем в проценты
        importance_percentages = importance_scores * 10

        # Сортируем по важности
        sorted_idx = np.argsort(importance_percentages)[::-1]
        sorted_names = [self.feature_names[i] for i in sorted_idx]
        sorted_values = importance_percentages[sorted_idx]

        # Создаем горизонтальную столбчатую диаграмму
        bars = plt.barh(range(len(sorted_names)), sorted_values, color='skyblue')

        # Добавляем значения на столбцы
        for i, (bar, value) in enumerate(zip(bars, sorted_values)):
            plt.text(value + 0.5, bar.get_y() + bar.get_height()/2,
                    f'{value:.1f}%', va='center')

        plt.yticks(range(len(sorted_names)), sorted_names)
        plt.xlabel('Важность (%)')
        plt.title(title)
        plt.gca().invert_yaxis()  # Самые важные сверху

        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()

        return sorted_names, sorted_values

    def create_importance_dataframe(self, importance_scores):
        """Создание DataFrame с результатами"""
        importance_percentages = importance_scores * 100

        df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance_Percent': importance_percentages,
            'Rank': np.argsort(importance_percentages)[::-1] + 1
        })

        return df.sort_values('Importance_Percent', ascending=False)

class GradientFeatureAnalyzer(AttentionFeatureAnalyzer):
    def __init__(self, model, feature_names, device='cpu'):
        super().__init__(model, feature_names, device)

    def calculate_gradient_importance(self, data_loader):
        """Расчет важности на основе градиентов"""
        self.model.train()  # Переводим модель в режим обучения для корректной работы RNN
        gradient_importance = np.zeros(len(self.feature_names))

        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            batch_x.requires_grad_(True)

            # Прямой проход
            outputs = self.model(batch_x)
            if isinstance(outputs, tuple):
                output = outputs[0] # Берем только выход, игнорируем attention weights
            else:
                output = outputs
            loss = torch.mean((output.squeeze() - batch_y) ** 2)

            # Обратный проход
            self.model.zero_grad()  # Обнуляем градиенты перед обратным проходом
            loss.backward()

            # Получаем градиенты
            gradients = batch_x.grad.detach().cpu().numpy()

            # Агрегируем градиенты по признакам
            # gradients shape: (batch_size, seq_len, num_features)
            feature_gradients = np.mean(np.abs(gradients), axis=(0, 1))
            gradient_importance += feature_gradients

        # Нормализуем
        gradient_importance = gradient_importance / np.sum(gradient_importance)

        return gradient_importance

class PermutationImportanceAnalyzer(AttentionFeatureAnalyzer):
    def __init__(self, model, feature_names, device='cpu'):
        super().__init__(model, feature_names, device)

    def calculate_permutation_importance(self, data_loader, metric_fn, n_repeats=5):
        """Расчет permutation importance"""
        baseline_score = self._evaluate_model(data_loader, metric_fn)
        permutation_scores = np.zeros((len(self.feature_names), n_repeats))

        for feature_idx in range(len(self.feature_names)):
            for repeat in range(n_repeats):
                # Перемешиваем значения признака
                score = self._evaluate_with_permuted_feature(
                    data_loader, metric_fn, feature_idx
                )
                permutation_scores[feature_idx, repeat] = score

        # Важность = снижение производительности при перемешивании
        importance = baseline_score - np.mean(permutation_scores, axis=1)
        importance = np.maximum(importance, 0)  # Отрицательные значения = 0

        # Нормализуем
        if np.sum(importance) > 0:
            importance = importance / np.sum(importance)

        return importance

    def _evaluate_model(self, data_loader, metric_fn):
        """Оценка модели с заданной метрикой"""
        self.model.eval()
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x = batch_x.to(self.device)
                outputs = self.model(batch_x)
                if isinstance(outputs, tuple):
                    predictions = outputs[0] # Берем только выход, игнорируем attention weights
                else:
                    predictions = outputs
                all_predictions.extend(predictions.cpu().numpy().flatten())
                all_targets.extend(batch_y.cpu().numpy())  # Добавил .cpu() для перемещения на CPU перед конвертацией в numpy

        return metric_fn(np.array(all_predictions), np.array(all_targets))

    def _evaluate_with_permuted_feature(self, data_loader, metric_fn, feature_idx):
        """Оценка модели с перемешанным признаком"""
        self.model.eval()
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                # Копируем и перемешиваем признак
                permuted_x = batch_x.clone()
                feature_values = permuted_x[:, :, feature_idx].cpu().numpy()  # Добавил .cpu() перед .numpy()
                np.random.shuffle(feature_values.flatten())
                permuted_x[:, :, feature_idx] = torch.tensor(feature_values, device=self.device)

                permuted_x = permuted_x.to(self.device)
                outputs = self.model(permuted_x)
                if isinstance(outputs, tuple):
                    predictions = outputs[0] # Берем только выход, игнорируем attention weights
                else:
                    predictions = outputs
                all_predictions.extend(predictions.cpu().numpy().flatten())
                all_targets.extend(batch_y.cpu().numpy())

        return metric_fn(np.array(all_predictions), np.array(all_targets))

class ComprehensiveFeatureAnalyzer:
    def __init__(self, model, feature_names, device='cpu'):
        self.attention_analyzer = AttentionFeatureAnalyzer(model, feature_names, device)
        self.gradient_analyzer = GradientFeatureAnalyzer(model, feature_names, device)
        self.permutation_analyzer = PermutationImportanceAnalyzer(model, feature_names, device)

    def analyze_all_methods(self, data_loader, metric_fn=None):
        """Комплексный анализ всеми методами"""
        results = {}

        print("Анализ на основе attention weights...")
        self.attention_analyzer.extract_attention_weights(data_loader)
        attention_importance = self.attention_analyzer.calculate_feature_importance()
        results['attention'] = attention_importance['combined']

        print("Анализ на основе градиентов...")
        gradient_importance = self.gradient_analyzer.calculate_gradient_importance(data_loader)
        results['gradient'] = gradient_importance

        if metric_fn is not None:
            print("Анализ permutation importance...")
            permutation_importance = self.permutation_analyzer.calculate_permutation_importance(
                data_loader, metric_fn
            )
            results['permutation'] = permutation_importance

        # Комбинированная важность
        if len(results) == 3:  # Все три метода
            combined = 0.5 * results['attention'] + 0.3 * results['gradient'] + 0.2 * results['permutation']
        elif len(results) == 2:  # Только два метода
            combined = 0.6 * results['attention'] + 0.4 * results['gradient']
        else:
            combined = results['attention']

        results['combined'] = combined

        return results

    def create_comprehensive_report(self, results):
        """Создание комплексного отчета"""
        df = pd.DataFrame({
            'Feature': self.attention_analyzer.feature_names,
            'Attention_Importance': results.get('attention', [0]*len(self.attention_analyzer.feature_names)) * 100,
            'Gradient_Importance': results.get('gradient', [0]*len(self.attention_analyzer.feature_names)) * 100,
            'Permutation_Importance': results.get('permutation', [0]*len(self.attention_analyzer.feature_names)) * 100,
            'Combined_Importance': results['combined'] * 100
        })

        # Добавляем ранги
        for method in ['Attention', 'Gradient', 'Permutation', 'Combined']:
            df[f'{method}_Rank'] = df[f'{method}_Importance'].rank(ascending=False).astype(int)

        return df.sort_values('Combined_Importance', ascending=False)

def directional_accuracy_metric(predictions, targets):
    """Метрика Directional Accuracy для permutation importance"""
    # Убедимся, что predictions и targets одномерные массивы
    if predictions.ndim > 1:
        predictions = predictions.flatten()
    if targets.ndim > 1:
        targets = targets.flatten()

    pred_directions = np.sign(predictions)
    true_directions = np.sign(targets)

    mask = (true_directions != 0)
    if np.sum(mask) > 0:
        return np.mean(pred_directions[mask] == true_directions[mask])
    return 0
