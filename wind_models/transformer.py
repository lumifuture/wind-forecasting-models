import torch
import torch.nn as nn


class TransformerModel(nn.Module):
    """
    Transformer encoder-only model for forecasting.

    Args:
        input_dim (int): Number of input features.
        d_model (int): Transformer embedding dimension.
        nhead (int): Number of attention heads.
        num_layers (int): Number of encoder layers.
        dim_feedforward (int): FFN hidden layer size.
        pred_len (int): Forecast steps.
        dropout (float): Dropout rate.
    """

    def __init__(
            self,
            input_dim: int,
            d_model: int,
            nhead: int,
            num_layers: int,
            dim_feedforward: int,
            pred_len: int,
            dropout: float = 0.1
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        self.fc = nn.Linear(d_model, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, input_dim)
        x = self.input_proj(x)  # (batch_size, seq_len, d_model)
        out = self.transformer(x)  # (batch_size, seq_len, d_model)
        last_hidden = out[:, -1, :]  # (batch_size, d_model)
        return self.fc(self.dropout(last_hidden))  # (batch_size, pred_len)
