import torch
import torch.nn as nn


class DemandLSTM(nn.Module):
    """
    Pure LSTM baseline for hourly steel energy demand forecasting.
    Input:  (batch, seq_len, n_features)
    Output: (batch, 1)
    """

    def __init__(self, input_size: int, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


class CNNLSTMDemand(nn.Module):
    """
    CNN-LSTM hybrid for steel energy demand forecasting.

    Flow:
        (batch, seq_len, n_features)
        → permute → (batch, n_features, seq_len)          # Conv1d expects (N, C, L)
        → CNN: two conv blocks + MaxPool → halves seq_len
        → permute → (batch, seq_len/2, cnn_channels)      # LSTM expects (N, L, C)
        → LSTM: learns temporal dependencies on CNN features
        → head: FC layers → scalar prediction

    Why CNN first:
      - Conv kernels (size=3) detect local patterns: idle→production ramp,
        shift starts, sudden load spikes — faster than LSTM gates
      - MaxPool(2): compresses 48-step sequence to 24, removing redundancy
        and giving LSTM a cleaner, higher-level representation
      - BatchNorm: normalises the huge scale range (idle ~10 vs production ~400 kWh)
        so LSTM doesn't fight with gradient magnitude differences
    """

    def __init__(self, input_size: int, cnn_channels: int = 64,
                 hidden_size: int = 128, num_layers: int = 2,
                 dropout: float = 0.3):
        super().__init__()

        # ── CNN block ──────────────────────────────────────
        self.cnn = nn.Sequential(
            # first conv: raw features → local pattern detectors
            nn.Conv1d(input_size, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),           # lighter dropout in CNN

            # second conv: combine detected patterns
            nn.Conv1d(cnn_channels, cnn_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels * 2),
            nn.ReLU(),

            # halve sequence length: 48 → 24
            nn.MaxPool1d(kernel_size=2),
        )
        cnn_out_channels = cnn_channels * 2     # 128

        # ── LSTM block ─────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=cnn_out_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # ── Prediction head ────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features)
        x = x.permute(0, 2, 1)          # → (batch, features, seq_len)
        x = self.cnn(x)                  # → (batch, cnn_channels*2, seq_len/2)
        x = x.permute(0, 2, 1)          # → (batch, seq_len/2, cnn_channels*2)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])
