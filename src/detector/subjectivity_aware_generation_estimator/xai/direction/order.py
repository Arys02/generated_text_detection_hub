from __future__ import annotations

import torch
from torch.utils.data import Dataset


def _perm_within(n: int, mode: str, window: int | None,
                 gen: torch.Generator | None = None) -> torch.Tensor:
    """Permutation des n positions RÉELLES d'une séquence (padding exclu)."""
    if mode == "identity" or n <= 1:
        return torch.arange(n)
    if mode == "shuffle":
        return torch.randperm(n, generator=gen)
    if mode == "reverse":
        return torch.arange(n - 1, -1, -1)
    if mode == "block":
        w = max(1, int(window))
        # l'ordre INTER-blocs est préservé ; seul l'ordre intra-bloc est détruit
        return torch.cat([torch.randperm(min(w, n - s), generator=gen) + s
                          for s in range(0, n, w)])
    raise ValueError(f"mode inconnu : {mode!r}")


class OrderedDataset(Dataset):
    """Permutation tirée à CHAQUE accès, donc différente à chaque epoch.
    Une permutation fixée une fois laisserait subsister un ordre arbitraire
    mais stable, que le modèle pourrait apprendre — ce qui invaliderait
    l'interprétation du run comme 'sans information d'ordre'."""

    def __init__(self, X: torch.Tensor, lengths: torch.Tensor, y: torch.Tensor,
                 mode: str = "identity", window: int | None = None):
        self.X, self.lengths, self.y = X, lengths, y
        self.mode, self.window = mode, window

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i):
        x, n = self.X[i], int(self.lengths[i])
        perm = _perm_within(n, self.mode, self.window)
        # padding laissé en queue : pack_padded_sequence l'ignore via lengths
        return torch.cat([x[perm], x[n:]]), self.lengths[i], self.y[i]


def permute_fixed(X: torch.Tensor, lengths: torch.Tensor, seed: int,
                  mode: str = "shuffle", window: int | None = None) -> torch.Tensor:
    """Permutation FIXE, pour l'évaluation. À moyenner sur plusieurs seeds :
    une permutation isolée apporte sa propre variance, qui se confondrait
    avec l'effet mesuré."""
    gen = torch.Generator().manual_seed(seed)
    out = X.clone()
    for i in range(len(X)):
        n = int(lengths[i])
        if n > 1:
            out[i, :n] = X[i, _perm_within(n, mode, window, gen)]
    return out