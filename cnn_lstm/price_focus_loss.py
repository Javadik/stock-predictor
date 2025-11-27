import torch
import torch.nn as nn
import torch.nn.functional as F

class PriceFocusDirectionalLoss(nn.Module):
    """Функция потерь, сфокусированная на точности цены с уменьшенным акцентом на DA"""
    def __init__(self, mse_weight=0.6, da_weight=0.3, attention_weight=0.05, pattern_weight=0.05):
        super().__init__()
        self.mse_weight = mse_weight
        self.da_weight = da_weight
        self.attention_weight = attention_weight
        self.pattern_weight = pattern_weight

    def forward(self, pred, target, attention_weights=None, pattern_features=None):
        pred = pred.squeeze()
        target = target.squeeze()

        # 1. MSE потеря - основная компонента теперь
        mse_loss = F.mse_loss(pred, target)

        # 2. Направленная потеря с уменьшенным весом
        pred_direction = torch.sign(pred)
        target_direction = torch.sign(target)

        # Уменьшенный штраф за неправильное направление
        directional_penalty = torch.mean(
            torch.relu(-pred_direction * target_direction) *
            (1 + torch.abs(target))  # Вес зависит от величины изменения
        )

        total_loss = self.mse_weight * mse_loss + self.da_weight * directional_penalty

        # 3. Регуляризация внимания
        if attention_weights is not None:
            # Поощряем распределенное внимание
            attention_entropy = -torch.mean(
                torch.sum(attention_weights * torch.log(attention_weights + 1e-8), dim=1)
            )
            total_loss += self.attention_weight * attention_entropy

        # 4. Регуляризация паттернов
        if pattern_features is not None:
            # L2 регуляризация для паттерн признаков
            pattern_regularization = torch.mean(torch.sum(pattern_features ** 2, dim=1))
            total_loss += self.pattern_weight * pattern_regularization

        return total_loss

class BalancedMSEDALoss(nn.Module):
    """Балансировка между MSE и DA с динамическим весом"""
    def __init__(self, initial_mse_weight=0.5, initial_da_weight=0.4,
                 attention_weight=0.05, pattern_weight=0.05,
                 mse_to_da_ratio_target=1.0):
        super().__init__()
        self.initial_mse_weight = initial_mse_weight
        self.initial_da_weight = initial_da_weight
        self.attention_weight = attention_weight
        self.pattern_weight = pattern_weight
        self.mse_to_da_ratio_target = mse_to_da_ratio_target

        # Регистрируем буферы для динамических весов
        self.register_buffer('mse_weight', torch.tensor(initial_mse_weight))
        self.register_buffer('da_weight', torch.tensor(initial_da_weight))

    def forward(self, pred, target, attention_weights=None, pattern_features=None):
        pred = pred.squeeze()
        target = target.squeeze()

        # MSE потеря
        mse_loss = F.mse_loss(pred, target)

        # Направленная потеря
        pred_direction = torch.sign(pred)
        target_direction = torch.sign(target)

        directional_penalty = torch.mean(
            torch.relu(-pred_direction * target_direction) *
            (1 + torch.abs(target))
        )

        # Динамическая балансировка весов
        with torch.no_grad():
            # Если MSE потеря значительно больше DA потери, уменьшаем вес MSE
            if mse_loss.item() > 0 and directional_penalty.item() > 0:
                current_ratio = mse_loss.item() / directional_penalty.item()
                if current_ratio > self.mse_to_da_ratio_target * 1.5:
                    self.mse_weight *= 0.95  # Уменьшаем вес MSE
                    self.da_weight *= 1.05  # Увеличиваем вес DA
                elif current_ratio < self.mse_to_da_ratio_target / 1.5:
                    self.mse_weight *= 1.05  # Увеличиваем вес MSE
                    self.da_weight *= 0.95  # Уменьшаем вес DA

                # Нормализуем веса, чтобы их сумма оставалась постоянной
                total_weight = self.mse_weight + self.da_weight + self.attention_weight + self.pattern_weight
                scaling_factor = (self.initial_mse_weight + self.initial_da_weight +
                                self.attention_weight + self.pattern_weight) / total_weight
                self.mse_weight *= scaling_factor
                self.da_weight *= scaling_factor

        total_loss = self.mse_weight * mse_loss + self.da_weight * directional_penalty

        # Регуляризации
        if attention_weights is not None:
            attention_entropy = -torch.mean(
                torch.sum(attention_weights * torch.log(attention_weights + 1e-8), dim=1)
            )
            total_loss += self.attention_weight * attention_entropy

        if pattern_features is not None:
            pattern_regularization = torch.mean(torch.sum(pattern_features ** 2, dim=1))
            total_loss += self.pattern_weight * pattern_regularization

        return total_loss

class ClassBalancedDirectionalLoss(nn.Module):
    """Функция потерь с балансировкой по классам на основе эффективных количеств"""
    def __init__(self, beta=0.99, mse_weight=0.2, da_weight=0.6):
        super().__init__()
        self.beta = beta  # Параметр для вычисления эффективных количеств
        self.mse_loss = nn.MSELoss()
        self.mse_weight = mse_weight
        self.da_weight = da_weight

        # Инициализируем веса для каждого класса
        self.register_buffer('pos_weight', torch.tensor(1.0))
        self.register_buffer('neg_weight', torch.tensor(1.0))

    def forward(self, pred, target, attention_weights=None, pattern_features=None):
        pred = pred.squeeze()
        target = target.squeeze()

        # MSE потеря
        mse_loss = self.mse_loss(pred, target)

        # Направленная потеря с балансировкой
        pred_direction = torch.sign(pred)
        target_direction = torch.sign(target)

        # Создаем маски для роста и падения
        pos_mask = (target_direction == 1)
        neg_mask = (target_direction == -1)

        # Вычисляем эффективные количества для каждого класса
        n_samples = len(target)
        n_pos = pos_mask.sum().float()
        n_neg = neg_mask.sum().float()

        # Веса, основанные на эффективных количествах
        # Используем формулу: (1 - beta^n) / (1 - beta) для эффективного количества
        effective_n_pos = (1 - self.beta**n_pos) / (1 - self.beta)
        effective_n_neg = (1 - self.beta**n_neg) / (1 - self.beta)

        # Веса потерь
        pos_weight = (n_samples / 2) / effective_n_pos
        neg_weight = (n_samples / 2) / effective_n_neg

        # Нормализуем веса
        total_effective = effective_n_pos + effective_n_neg
        pos_weight = effective_n_neg / total_effective  # Чем меньше положительных примеров, тем выше вес
        neg_weight = effective_n_pos / total_effective  # Чем меньше отрицательных примеров, тем выше вес

        # Вычисляем потери для каждого направления с весами
        pos_loss = 0
        neg_loss = 0

        if pos_mask.any():
            pos_loss = torch.mean(
                torch.relu(-pred_direction[pos_mask] * target_direction[pos_mask]) *
                (1 + torch.abs(target[pos_mask]))
            ) * pos_weight

        if neg_mask.any():
            neg_loss = torch.mean(
                torch.relu(-pred_direction[neg_mask] * target_direction[neg_mask]) *
                (1 + torch.abs(target[neg_mask]))
            ) * neg_weight

        # Объединяем потери
        directional_loss = (pos_loss + neg_loss) / 2

        total_loss = self.mse_weight * mse_loss + self.da_weight * directional_loss
        return total_loss
