# file: src/detector/subjectivity_aware_generation_estimator/model/SimpleGRU.py
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class SimpleGRU(nn.Module):
    """GRU naïf : 1 couche, unidirectionnel, dernier état caché -> tête linéaire.
    n_outputs=1 : 1 logit (BCEWithLogits, binaire) ; n_outputs=K>1 : K logits (CrossEntropy)."""

    def __init__(self, n_features=5, hidden=32, n_layers=1, n_outputs=1):
        super().__init__()
        self.n_outputs = n_outputs
        self.gru = nn.GRU(input_size=n_features, hidden_size=hidden,
                          num_layers=n_layers, batch_first=True)
        self.head = nn.Linear(hidden, n_outputs)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)
        _, h_n = self.gru(packed)
        last = h_n[-1]  # (batch, hidden)
        out = self.head(last)
        # getattr : les modèles picklés avant l'ajout de n_outputs n'ont pas l'attribut
        if getattr(self, "n_outputs", 1) == 1:
            return out.squeeze(-1)  # (batch,) logits
        return out  # (batch, n_outputs) logits
