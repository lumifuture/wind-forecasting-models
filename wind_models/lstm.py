import torch.nn as nn


# Define the LSTM-based forecasting model
class LSTMModel(nn.Module):
    """
    LSTM-based time series forecasting model.

    Args:
        input_dim (int): Number of input features.
        hidden_dim (int): Number of LSTM hidden units.
        num_layers (int): Number of stacked LSTM layers.
        pred_len (int): Number of future time steps to predict.
    """

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            num_layers: int,
            pred_len: int,
            dropout: float = 0.0
    ):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pred_len = pred_len

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0  # PyTorch ignores dropout if num_layers == 1
        )

        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        self.fc = nn.Linear(hidden_dim, pred_len)  # predict multiple future steps at once

    def forward(self, x):
        """
        Args:
            x (Tensor): shape (batch_size, input_len, input_dim)

        Returns:
            Tensor: shape (batch_size, pred_len)
        """
        out, _ = self.lstm(x)  # out: (batch_size, input_len, hidden_dim)
        last_hidden = out[:, -1, :]  # take last time step's hidden state
        prediction = self.fc(self.dropout(last_hidden))  # (batch_size, pred_len)
        return prediction
