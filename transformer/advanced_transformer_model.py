import torch
import torch.nn as nn
import math

class AdvancedPositionalEncoding(nn.Module):
    """Продвинутое позиционное кодирование для финансовых временных рядов"""
    def __init__(self, d_model, max_len=500, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Создание матрицы позиционного кодирования
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Добавление финансово-ориентированных паттернов
        # Недельные циклы (5 рабочих дней)
        weekly_pattern = torch.sin(torch.arange(max_len, dtype=torch.float) * 2 * math.pi / 5).unsqueeze(1)
        # Месячные циклы (около 21 рабочего дня)
        monthly_pattern = torch.sin(torch.arange(max_len, dtype=torch.float) * 2 * math.pi / 21).unsqueeze(1)

        # Добавление этих паттернов в разные части вектора
        pe[:, :d_model//4] += 0.1 * weekly_pattern[:max_len].expand(-1, d_model//4)
        pe[:, d_model//4:d_model//2] += 0.1 * monthly_pattern[:max_len].expand(-1, d_model//4)

        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class FinancialAttentionLayer(nn.Module):
    """Финансово-специфичный слой внимания"""
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Слои для анализа волатильности и объема
        self.volatility_predictor = nn.Linear(d_model, d_model)
        self.volume_predictor = nn.Linear(d_model, d_model)
        self.price_predictor = nn.Linear(d_model, d_model)

        # Специальный гейт для управления вниманием
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Основное внимание
        attn_output, attn_weights = self.multihead_attn(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.norm(x)

        # Финансовая модуляция
        vol_signal = torch.tanh(self.volatility_predictor(x))
        vol_gate = self.gate(x)

        # Комбинируем с основным выходом
        modulated = x * (1 + 0.1 * vol_signal * vol_gate)

        return modulated, attn_weights


class MultiScaleTemporalExtractor(nn.Module):
    """Многошкальный извлечение временных паттернов"""
    def __init__(self, d_model):
        super().__init__()

        # Сверточные слои для извлечения паттернов разной длины
        self.short_term_conv = nn.Conv1d(d_model, d_model//4, kernel_size=3, padding=1)
        self.medium_term_conv = nn.Conv1d(d_model, d_model//4, kernel_size=7, padding=3)
        self.long_term_conv = nn.Conv1d(d_model, d_model//4, kernel_size=15, padding=7)

        # Слой для объединения паттернов
        self.fusion = nn.Linear(d_model*3//4, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x_permuted = x.permute(0, 2, 1) # [batch_size, d_model, seq_len]

        short_patterns = self.short_term_conv(x_permuted).permute(0, 2, 1)  # [batch_size, seq_len, d_model//4]
        medium_patterns = self.medium_term_conv(x_permuted).permute(0, 2, 1)  # [batch_size, seq_len, d_model//4]
        long_patterns = self.long_term_conv(x_permuted).permute(0, 2, 1) # [batch_size, seq_len, d_model//4]

        # Объединение паттернов
        combined = torch.cat([short_patterns, medium_patterns, long_patterns], dim=-1)  # [batch_size, seq_len, 3*d_model//4]

        # Проекция обратно к d_model
        fused = self.fusion(combined)

        return self.norm(fused + x)  # Residual connection


class AdvancedTransformerDirectionalLoss(nn.Module):
    """Продвинутая функция потерь для Directional Accuracy"""
    def __init__(self, mse_weight=0.05, da_weight=0.7, trend_weight=0.2, confidence_weight=0.05):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.mse_weight = mse_weight
        self.da_weight = da_weight
        self.trend_weight = trend_weight
        self.confidence_weight = confidence_weight

    def forward(self, pred, target):
        pred = pred.squeeze()
        target = target.squeeze()

        # Стандартная MSE потеря
        mse_loss = self.mse_loss(pred, target)

        # Направленная потеря с адаптивными весами
        pred_direction = torch.sign(pred)
        target_direction = torch.sign(target)

        # Усиленный штраф за неправильное направление
        directional_penalty = torch.mean(
            torch.relu(-pred_direction * target_direction) *
            (1 + torch.abs(target))  # Вес зависит от величины изменения
        )

        # Потеря на основе трендов (штраф за игнорирование сильных движений)
        trend_penalty = torch.mean(
            torch.abs(pred_direction - target_direction) *
            torch.abs(target) * (torch.abs(target) > 0.02).float()
        )

        # Потеря уверенности (штраф за предсказания близкие к нулю когда целевое значение далеко от нуля)
        confidence_penalty = torch.mean(
            torch.abs(pred) * (torch.abs(target) > 0.01).float() *
            torch.relu(-pred_direction * target_direction)
        )

        total_loss = (
            self.mse_weight * mse_loss +
            self.da_weight * directional_penalty +
            self.trend_weight * trend_penalty +
            self.confidence_weight * confidence_penalty
        )

        return total_loss


class AdvancedStockTransformer(nn.Module):
    """Продвинутая архитектура Transformer для предсказания акций"""
    def __init__(self, input_size=20, d_model=256, nhead=4,
                 num_encoder_layers=3, dim_feedforward=1024,
                 dropout=0.2, max_seq_length=5000):
        super().__init__()

        # Входной слой - проекция признаков
        self.input_projection = nn.Linear(input_size, d_model)

        # Продвинутое позиционное кодирование
        self.pos_encoder = AdvancedPositionalEncoding(d_model, max_seq_length, dropout)

        # Слой нормализации
        self.norm_layer = nn.LayerNorm(d_model)

        # Продвинутые слои внимания
        self.attention_layers = nn.ModuleList([
            FinancialAttentionLayer(d_model, nhead, dropout)
            for _ in range(num_encoder_layers)
        ])

        # Слой извлечения временных паттернов
        self.temporal_extractor = MultiScaleTemporalExtractor(d_model)

        # Улучшенные выходные слои
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim_feedforward//2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Слой внимания для финального предсказания
        self.prediction_attention = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        # Финальный выходной слой
        self.output_layer = nn.Linear(d_model, 1)

        self.d_model = d_model
        self.dropout = dropout

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        batch_size, seq_len, input_size = x.size()

        # Проекция входных признаков
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)

        # Применение позиционного кодирования
        x = x.transpose(0, 1)  # (seq_len, batch_size, d_model)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # (batch_size, seq_len, d_model)

        # Прохождение через слои внимания
        for layer in self.attention_layers:
            x, _ = layer(x)

        # Извлечение временных паттернов
        x = self.temporal_extractor(x)

        # Примение feedforward
        ff_output = self.feedforward(x)

        # Использование последнего временного шага для предсказания
        # Но с учетом всех шагов через attention
        last_output = x[:, -1, :]  # (batch_size, d_model)

        # Вместо усреднения, используем последний временной шаг
        output = self.output_layer(last_output)

        return output


def create_advanced_transformer_model(input_size=20, config=None):
    """Создание продвинутой модели Transformer с заданной конфигурацией"""
    if config is None:
        # Продвинутая конфигурация по умолчанию
        config = {
            'd_model': 256,  # Уменьшенный размер для лучшей сходимости
            'nhead': 4,      # Должно делиться на d_model
            'num_encoder_layers': 3,  # Уменьшенное количество слоев для стабильности
            'dim_feedforward': 1024,
            'dropout': 0.2   # Увеличенный dropout для регуляризации
        }

    return AdvancedStockTransformer(
        input_size=input_size,
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_encoder_layers=config['num_encoder_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout']
    )
