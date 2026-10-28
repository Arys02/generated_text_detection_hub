import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class SimpleGRUWithGlobal(nn.Module):
    """Features de niveau phrase dans la récurrence, features de niveau review
    injectées après pooling. Évite de diffuser un scalaire sur L positions."""

    def __init__(self, n_features=6, n_global=0, hidden=32, n_layers=1):
        super().__init__()
        self.n_global = n_global
        self.gru = nn.GRU(n_features, hidden, num_layers=n_layers, batch_first=True)
        self.head = nn.Linear(hidden + n_global, 1)

    def forward(self, x, lengths, g=None):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed)
        z = h_n[-1]
        if self.n_global:
            z = torch.cat([z, g], dim=1)
        return self.head(z).squeeze(-1)