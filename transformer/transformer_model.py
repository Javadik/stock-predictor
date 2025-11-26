import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """Позиционное кодирование для финансовых временных рядов"""
    def __init__(self, d_model, max_len=500, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Адаптированное позиционное кодирование для финансовых данных
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Дополнительная кодировка для недельных/месячных паттернов
        weekly_pattern = torch.sin(2 * math.pi * position / 5)  # 5 торговых дней
        monthly_pattern = torch.sin(2 * math.pi * position / 21)  # ~21 торговый день

        pe[:, :d_model//4] += weekly_pattern[:max_len, :d_model//4]
        pe[:, d_model//4:d_model//2] += monthly_pattern[:max_len, :d_model//4-d_model%4]

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TemporalFusionLayer(nn.Module):
    """Временная фьюжн-слоя для многошкального анализа"""
    def __init__(self, d_model):
        super().__init__()

        # Внимание для разных временных горизонтов
        self.short_term_attention = nn.MultiheadAttention(d_model, 4, batch_first=True)
        self.medium_term_attention = nn.MultiheadAttention(d_model, 4, batch_first=True)
        self.long_term_attention = nn.MultiheadAttention(d_model, 4, batch_first=True)

        # Фьюжн слой
        self.fusion = nn.Linear(d_model * 3, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape

        # Краткосрочное внимание (последние 10 дней)
        short_x = x[:, -10:, :] if seq_len >= 10 else x
        short_out, _ = self.short_term_attention(short_x, short_x, short_x)
        short_pooled = torch.mean(short_out, dim=1)  # [batch_size, d_model]

        # Среднесрочное внимание (последние 30 дней)
        medium_x = x[:, -30:, :] if seq_len >= 30 else x
        medium_out, _ = self.medium_term_attention(medium_x, medium_x, medium_x)
        medium_pooled = torch.mean(medium_out, dim=1)

        # Долгосрочное внимание (вся последовательность)
        long_out, _ = self.long_term_attention(x, x, x)
        long_pooled = torch.mean(long_out, dim=1)

        # Объединение временных масштабов
        combined = torch.cat([short_pooled, medium_pooled, long_pooled], dim=-1)
        fused = self.fusion(combined)

        return self.norm(fused)


class TransformerDirectionalLoss(nn.Module):
    """Специализированная функция потерь для Directional Accuracy"""
    def __init__(self, mse_weight=0.2, da_weight=0.6,
                 volatility_weight=0.1, confidence_weight=0.1):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.mse_weight = mse_weight
        self.da_weight = da_weight
        self.volatility_weight = volatility_weight
        self.confidence_weight = confidence_weight

    def forward(self, pred, target, volatility=None):
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

        # 3. Волатильно-адаптивная потеря
        if volatility is not None:
            # Больший штраф в периоды высокой волатильности
            volatility_adjusted_penalty = directional_penalty * (1 + volatility)
        else:
            volatility_adjusted_penalty = directional_penalty

        # 4. Штраф за неуверенные предсказания
        confidence_penalty = torch.mean(torch.abs(pred - target))

        total_loss = (
            self.mse_weight * mse_loss +
            self.da_weight * volatility_adjusted_penalty +
            self.confidence_weight * confidence_penalty
        )

        return total_loss


class StockTransformer(nn.Module):
    """Основная архитектура Transformer для предсказания акций"""
    def __init__(self, input_size=20, d_model=512, nhead=8,
                 num_encoder_layers=6, dim_feedforward=2048,
                 dropout=0.1, max_seq_length=5000):
        super().__init__()

        # 1. Входной слой - проекция признаков
        self.input_projection = nn.Linear(input_size, d_model)

        # 2. Позиционное кодирование для временных рядов
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length, dropout)

        # 3. Энкодер Transformer с каузальным маскированием
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )

        # 4. Многошкальный анализ временных паттернов
        self.temporal_fusion = TemporalFusionLayer(d_model)

        # 5. Выходные слои для предсказания
        self.output_layers = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1)
        )

        self.d_model = d_model
        self.dropout = dropout

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        batch_size, seq_len, input_size = x.size()

        # 1. Проекция входных признаков
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)

        # 2. Позиционное кодирование
        x = self.pos_encoder(x)

        # 3. Transformer энкодер
        x = self.transformer_encoder(x)

        # 4. Многошкальный анализ временных паттернов
        x = self.temporal_fusion(x)

        # 5. Выходные слои
        output = self.output_layers(x)

        return output


# Функция для создания модели с определенными гиперпараметрами
def create_transformer_model(input_size=20, config=None):
    """Создание модели Transformer с заданной конфигурацией"""
    if config is None:
        # Конфигурация по умолчанию (сбалансированная модель)
        config = {
            'd_model': 512,
            'nhead': 8,
            'num_encoder_layers': 6,
            'dim_feedforward': 2048,
            'dropout': 0.1
        }

    return StockTransformer(
        input_size=input_size,
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_encoder_layers=config['num_encoder_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout']
    )
