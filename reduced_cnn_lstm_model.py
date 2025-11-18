import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleCNN(nn.Module):
    """Многошкальная CNN для извлечения паттернов разных масштабов"""
    def __init__(self, in_channels, out_channels_list):
        super().__init__()
        self.convs = nn.ModuleList()

        # Разные размеры ядер для извлечения паттернов разных масштабов
        kernel_sizes = [3, 5, 7, 9]

        for i, kernel_size in enumerate(kernel_sizes):
            self.convs.append(nn.Sequential(
                # Первая свертка
                nn.Conv1d(in_channels, out_channels_list[i],
                         kernel_size=kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(out_channels_list[i]),
                nn.ReLU(),
                nn.Dropout(0.2),

                # Вторая свертка для углубления
                nn.Conv1d(out_channels_list[i], out_channels_list[i],
                         kernel_size=kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(out_channels_list[i]),
                nn.ReLU(),
                nn.Dropout(0.2)
            ))

    def forward(self, x):
        # x shape: (batch_size, features, seq_length)
        outputs = [conv(x) for conv in self.convs]
        return torch.cat(outputs, dim=1)  # Конкатенация по каналам


class TechnicalPatternExtractor(nn.Module):
    """Извлечение технических паттернов (свечные, трендовые, волатильность)"""
    def __init__(self, num_features):
        super().__init__()

        # Свёртки для извлечения свечных паттернов (краткосрочные)
        self.candlestick_patterns = nn.Sequential(
            nn.Conv1d(num_features, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Свёртки для извлечения трендовых паттернов (среднесрочные)
        self.trend_patterns = nn.Sequential(
            nn.Conv1d(num_features, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Свёртки для извлечения волатильности (долгосрочные)
        self.volatility_patterns = nn.Sequential(
            nn.Conv1d(num_features, 32, kernel_size=11, padding=5),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=13, padding=6),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Адаптивные веса для разных типов паттернов
        self.pattern_weights = nn.Parameter(torch.ones(3))

    def forward(self, x):
        # x shape: (batch_size, features, seq_length)
        candlestick = self.candlestick_patterns(x)
        trend = self.trend_patterns(x)
        volatility = self.volatility_patterns(x)

        # Взвешенное объединение паттернов
        weights = F.softmax(self.pattern_weights, dim=0)

        weighted_candlestick = candlestick * weights[0]
        weighted_trend = trend * weights[1]
        weighted_volatility = volatility * weights[2]

        # Объединение всех паттернов
        combined = torch.cat([weighted_candlestick, weighted_trend, weighted_volatility], dim=1)
        return combined


class AttentionLSTM(nn.Module):
    """LSTM с механизмом внимания для временного моделирования"""
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.3):
        super().__init__()

        # Двунаправленный LSTM для захвата прошлого и будущего контекста
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout, bidirectional=True)

        # Механизм внимания
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),  # *2 из-за bidirectional
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )

        # Дополнительный слой для улучшения репрезентации
        self.enhancement = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch_size, seq_len, features)
        lstm_out, (hidden, cell) = self.lstm(x)

        # Улучшение признаков
        enhanced_out = self.enhancement(lstm_out)

        # Вычисление весов внимания
        attention_scores = self.attention(enhanced_out)  # (batch_size, seq_len, 1)
        attention_weights = F.softmax(attention_scores, dim=1)

        # Применение весов внимания
        attended_output = torch.sum(lstm_out * attention_weights, dim=1)

        return attended_output, attention_weights


class ReducedStockCNNLSTM(nn.Module):
    """Гибридная CNN+LSTM архитектура для предсказания акций с уменьшенным количеством признаков"""
    def __init__(self, input_size=10, seq_length=60,
                 cnn_channels=[16, 32, 64, 128],  # Уменьшенные каналы пропорционально количеству признаков
                 lstm_hidden_size=128,  # Уменьшенный размер LSTM
                 lstm_layers=2,
                 dropout=0.3):
        super().__init__()

        # Многошкальное извлечение признаков
        self.multiscale_cnn = MultiScaleCNN(input_size, cnn_channels)

        # Извлечение технических паттернов
        self.pattern_extractor = TechnicalPatternExtractor(input_size)

        # Расчет размерности после CNN
        # multiscale_cnn: [16, 32, 64, 128] -> сумма = 256
        # pattern_extractor: 64 + 64 = 192 (candlestick + trend + volatility)
        multiscale_channels = sum(cnn_channels)  # 16+32+64+128 = 256
        pattern_channels = 64 + 64 + 64  # candlestick + trend + volatility = 192
        cnn_output_channels = multiscale_channels + pattern_channels  # 448

        # Слой уменьшения размерности для соответствия ожидаемому размеру LSTM
        self.dimension_reducer = nn.Conv1d(cnn_output_channels, lstm_hidden_size * 2, kernel_size=1)

        # LSTM с вниманием - принимает признаки нужной размерности
        self.attention_lstm = AttentionLSTM(lstm_hidden_size * 2, lstm_hidden_size, lstm_layers, dropout)

        # Выходной слой
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, 64),  # *2 из-за bidirectional, уменьшенный размер
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(32, 1)
        )

        self.dropout = nn.Dropout(dropout)

        # Для отладки
        self.seq_length = seq_length

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        batch_size, seq_len, input_size = x.size()

        # Транспонирование для Conv1d: (batch_size, input_size, seq_length)
        x_transposed = x.transpose(1, 2)

        # Извлечение признаков разными методами
        multiscale_features = self.multiscale_cnn(x_transposed)  # (batch, total_multiscale_channels, seq_len)
        pattern_features = self.pattern_extractor(x_transposed)  # (batch, 192, seq_len)

        # Объединение признаков
        combined_features = torch.cat([multiscale_features, pattern_features], dim=1)  # (batch, total_channels, seq_len)

        # Уменьшение размерности до нужной для LSTM
        # Применяем 1D свертку для уменьшения количества каналов до lstm_hidden_size*2
        combined_features = self.dimension_reducer(combined_features)  # (batch, lstm_hidden_size*2, seq_len)

        # Транспонирование для LSTM: (batch, seq_len, features)
        combined_features = combined_features.transpose(1, 2) # (batch, seq_len, lstm_hidden_size*2)

        # Обработка LSTM с вниманием
        lstm_output, attention_weights = self.attention_lstm(combined_features)

        # Классификация
        output = self.classifier(lstm_output)

        return output, attention_weights


def create_reduced_cnn_lstm_model(input_size=10, config=None):
    """Создание уменьшенной модели CNN+LSTM с заданной конфигурацией"""
    if config is None:
        # Конфигурация по умолчанию (сбалансированная модель)
        config = {
            'cnn_channels': [16, 32, 64, 128],
            'lstm_hidden_size': 128,
            'lstm_layers': 2,
            'dropout': 0.3
        }

    return ReducedStockCNNLSTM(
        input_size=input_size,
        cnn_channels=config['cnn_channels'],
        lstm_hidden_size=config['lstm_hidden_size'],
        lstm_layers=config['lstm_layers'],
        dropout=config['dropout']
    )
