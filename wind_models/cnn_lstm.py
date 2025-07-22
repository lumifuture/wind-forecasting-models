import torch
import torch.nn as nn


class CNNLSTMModel(nn.Module):
    """
    CNN + LSTM hybrid for time series forecasting.

    Args:
        input_dim (int): Number of input features.
        cnn_channels (int): Number of CNN output channels.
        lstm_hidden_dim (int): Hidden size of the LSTM layer.
        lstm_layers (int): Number of LSTM layers.
        pred_len (int): Number of future steps to predict.
        dropout (float): Dropout probability.
    """

    def __init__(
            self,
            input_dim: int,
            cnn_channels: int,
            lstm_hidden_dim: int,
            lstm_layers: int,
            pred_len: int,
            dropout: float = 0.0
    ):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=cnn_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )

        self.fc = nn.Linear(lstm_hidden_dim, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, input_dim)
        x = x.permute(0, 2, 1)  # (batch, input_dim, seq_len)
        x = self.conv1(x)  # (batch, cnn_channels, seq_len)
        x = self.relu(x)
        x = x.permute(0, 2, 1)  # (batch, seq_len, cnn_channels)

        out, _ = self.lstm(x)  # (batch, seq_len, lstm_hidden_dim)
        last_hidden = out[:, -1, :]  # (batch, lstm_hidden_dim)
        return self.fc(self.dropout(last_hidden))  # (batch, pred_len)
