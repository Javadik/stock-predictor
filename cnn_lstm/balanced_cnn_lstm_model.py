import torch
import torch.nn as nn
import torch.nn.functional as F

class BalancedCNNLSTMDirectionalLoss(nn.Module):
    """Улучшенная функция потерь с балансировкой для Directional Accuracy"""
    def __init__(self, mse_weight=0.2, da_weight=0.5,
                 balance_weight=0.2, attention_weight=0.05, pattern_weight=0.05):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.mse_weight = mse_weight
        self.da_weight = da_weight
        self.balance_weight = balance_weight
        self.attention_weight = attention_weight
        self.pattern_weight = pattern_weight

    def forward(self, pred, target, attention_weights=None, pattern_features=None):
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

        # 3. Балансировка потерь для роста и падения
        up_mask = (target_direction == 1)
        down_mask = (target_direction == -1)

        up_penalty = 0
        down_penalty = 0

        if up_mask.any():
            up_penalty = torch.mean(
                torch.relu(-pred_direction[up_mask] * target_direction[up_mask]) *
                (1 + torch.abs(target[up_mask]))
            )

        if down_mask.any():
            down_penalty = torch.mean(
                torch.relu(-pred_direction[down_mask] * target_direction[down_mask]) *
                (1 + torch.abs(target[down_mask]))
            )

        # Балансировка: штраф за игнорирование одного из направлений
        balance_penalty = torch.abs(up_penalty - down_penalty)

        total_loss = (self.mse_weight * mse_loss +
                      self.da_weight * directional_penalty +
                      self.balance_weight * balance_penalty)

        # 4. Регуляризация внимания (для предотвращения перефокусировки)
        if attention_weights is not None:
            # Поощряем распределенное внимание
            attention_entropy = -torch.mean(
                torch.sum(attention_weights * torch.log(attention_weights + 1e-8), dim=1)
            )
            total_loss += self.attention_weight * attention_entropy

        # 5. Регуляризация паттернов (для предотвращения переобучения на паттерны)
        if pattern_features is not None:
            # L2 регуляризация для паттерн признаков
            pattern_regularization = torch.mean(torch.sum(pattern_features ** 2, dim=1))
            total_loss += self.pattern_weight * pattern_regularization

        return total_loss

class FocalDirectionalLoss(nn.Module):
    """Focal Loss для Directional Accuracy - помогает при дисбалансе классов"""
    def __init__(self, alpha=0.25, gamma=2.0, mse_weight=0.2, da_weight=0.6):
        super().__init__()
        self.alpha = alpha  # Вес для редких классов
        self.gamma = gamma  # Параметр фокусировки
        self.mse_loss = nn.MSELoss()
        self.mse_weight = mse_weight
        self.da_weight = da_weight

    def forward(self, pred, target, attention_weights=None, pattern_features=None):
        pred = pred.squeeze()
        target = target.squeeze()

        # MSE потеря
        mse_loss = self.mse_loss(pred, target)

        # Направленная потеря с focal весами
        pred_direction = torch.sign(pred)
        target_direction = torch.sign(target)

        # Бинарные метки для focal loss (0 для падения, 1 для роста)
        binary_target = (target_direction + 1) / 2  # -1->0, 1->1
        binary_pred = (pred_direction + 1) / 2  # -1->0, 1->1

        # Вычисление focal loss для каждого направления
        pt = torch.where(binary_target == 1, binary_pred, 1 - binary_pred)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        focal_loss = -focal_weight * torch.log(pt + 1e-8)

        # Усреднение с учетом весов
        directional_loss = torch.mean(focal_loss)

        total_loss = self.mse_weight * mse_loss + self.da_weight * directional_loss
        return total_loss

class WeightedDirectionalLoss(nn.Module):
    """Взвешенная функция потерь для балансировки роста и падения"""
    def __init__(self, pos_weight=1.0, neg_weight=1.0, mse_weight=0.2, da_weight=0.6):
        super().__init__()
        self.pos_weight = pos_weight  # Вес для положительных (рост) примеров
        self.neg_weight = neg_weight  # Вес для отрицательных (падение) примеров
        self.mse_loss = nn.MSELoss()
        self.mse_weight = mse_weight
        self.da_weight = da_weight

    def forward(self, pred, target, attention_weights=None, pattern_features=None):
        pred = pred.squeeze()
        target = target.squeeze()

        # MSE потеря
        mse_loss = self.mse_loss(pred, target)

        # Направленная потеря с весами
        pred_direction = torch.sign(pred)
        target_direction = torch.sign(target)

        # Создаем маски для роста и падения
        pos_mask = (target_direction == 1)
        neg_mask = (target_direction == -1)

        # Вычисляем потери для каждого направления с весами
        pos_loss = 0
        neg_loss = 0

        if pos_mask.any():
            pos_loss = torch.mean(
                torch.relu(-pred_direction[pos_mask] * target_direction[pos_mask]) *
                (1 + torch.abs(target[pos_mask]))
            ) * self.pos_weight

        if neg_mask.any():
            neg_loss = torch.mean(
                torch.relu(-pred_direction[neg_mask] * target_direction[neg_mask]) *
                (1 + torch.abs(target[neg_mask]))
            ) * self.neg_weight

        # Объединяем потери
        directional_loss = (pos_loss + neg_loss) / 2

        total_loss = self.mse_weight * mse_loss + self.da_weight * directional_loss
        return total_loss
