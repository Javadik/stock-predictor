import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicWeightedDirectionalLoss(nn.Module):
    """Динамически взвешенная функция потерь для балансировки роста и падения"""
    def __init__(self, initial_pos_weight=1.0, initial_neg_weight=1.0,
                 mse_weight=0.2, da_weight=0.6, momentum=0.9):
        super().__init__()
        self.initial_pos_weight = initial_pos_weight
        self.initial_neg_weight = initial_neg_weight
        self.mse_loss = nn.MSELoss()
        self.mse_weight = mse_weight
        self.da_weight = da_weight
        self.momentum = momentum  # Для скользящего среднего

        # Инициализируем динамические веса
        self.register_buffer('pos_weight', torch.tensor(initial_pos_weight))
        self.register_buffer('neg_weight', torch.tensor(initial_neg_weight))

        # Для отслеживания статистики
        self.register_buffer('running_pos_count', torch.tensor(0.0))
        self.register_buffer('running_neg_count', torch.tensor(0.0))
        self.register_buffer('update_step', torch.tensor(0))

    def forward(self, pred, target, attention_weights=None, pattern_features=None):
        pred = pred.squeeze()
        target = target.squeeze()

        # MSE потеря
        mse_loss = self.mse_loss(pred, target)

        # Направленная потеря с динамическими весами
        pred_direction = torch.sign(pred)
        target_direction = torch.sign(target)

        # Обновляем статистику и веса каждые N шагов
        with torch.no_grad():
            pos_mask = (target_direction == 1)
            neg_mask = (target_direction == -1)

            current_pos_count = pos_mask.sum().float()
            current_neg_count = neg_mask.sum().float()

            # Обновляем скользящие средние
            self.running_pos_count = self.momentum * self.running_pos_count + (1 - self.momentum) * current_pos_count
            self.running_neg_count = self.momentum * self.running_neg_count + (1 - self.momentum) * current_neg_count

            # Динамически обновляем веса на основе дисбаланса
            total_count = self.running_pos_count + self.running_neg_count + 1e-8  # избегаем деления на 0
            pos_proportion = self.running_pos_count / total_count
            neg_proportion = self.running_neg_count / total_count

            # Веса обратно пропорциональны доле класса (чем меньше класс, тем выше вес)
            new_pos_weight = 1.0 / (pos_proportion + 1e-8)
            new_neg_weight = 1.0 / (neg_proportion + 1e-8)

            # Нормализуем веса
            avg_weight = (new_pos_weight + new_neg_weight) / 2
            new_pos_weight = new_pos_weight / avg_weight
            new_neg_weight = new_neg_weight / avg_weight

            # Обновляем веса с учетом momentum
            self.pos_weight = 0.9 * self.pos_weight + 0.1 * new_pos_weight
            self.neg_weight = 0.9 * self.neg_weight + 0.1 * new_neg_weight

        # Вычисляем потери для каждого направления с динамическими весами
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
        effective_n_pos = (1 - self.beta) / (1 - self.beta ** n_pos)
        effective_n_neg = (1 - self.beta) / (1 - self.beta ** n_neg)

        # Веса потерь
        pos_weight = (n_samples / 2) / effective_n_pos
        neg_weight = (n_samples / 2) / effective_n_neg

        # Нормализуем веса
        total_effective = effective_n_pos + effective_n_neg
        pos_weight = n_samples * effective_n_neg / (2 * total_effective)
        neg_weight = n_samples * effective_n_pos / (2 * total_effective)

        # Сохраняем веса для отладки
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight

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
