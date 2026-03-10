"""
GenAI Usage Statement:
This code was developed with the assistance of Claude (Anthropic) as a coding assistant.
The AI tool was used for code structuring, debugging, and implementation guidance.
All outputs were manually reviewed and verified for technical correctness by the authors.
"""

import torch.nn as nn


class DeterministicLSTM(nn.Module):
    """Standard LSTM model that outputs a single RUL value (baseline).

    Args:
        input_dim: Number of input features.
        hidden_dim: LSTM hidden state dimension.
        num_layers: Number of stacked LSTM layers.
        dropout: Dropout probability applied between LSTM layers.
    """

    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.25):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout
        )
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim).

        Returns:
            Predicted RUL value, shape (batch_size, 1).
        """
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        last_hidden = self.dropout(last_hidden)
        return self.fc(last_hidden)
