from __future__ import annotations

import numpy as np
import torch


def gradient_profile(model, X, L, device: str, n_bins: int = 10,
                     batch_size: int = 64) -> np.ndarray:
    """Norme du gradient du logit par rapport à chaque position, agrégée en
    bins de position RELATIVE (les longueurs varient de 1 à 208)."""
    model.eval().to(device)
    acc = np.zeros(n_bins)
    cnt = np.zeros(n_bins)

    for s in range(0, len(X), batch_size):
        xb = X[s:s + batch_size].clone().to(device).requires_grad_(True)
        lb = L[s:s + batch_size]
        logits = model(xb, lb)
        logits.sum().backward()
        g = xb.grad.norm(dim=-1).detach().cpu().numpy()      # (B, Lmax)

        for i, n in enumerate(lb.tolist()):
            if n < n_bins:
                continue
            u = np.arange(n) / max(1, n - 1)                 # position relative 0..1
            b = np.clip((u * n_bins).astype(int), 0, n_bins - 1)
            np.add.at(acc, b, g[i, :n])
            np.add.at(cnt, b, 1)

    return acc / np.maximum(cnt, 1)


@torch.no_grad()
def occlusion_profile(model, X, L, device: str, n_bins: int = 10,
                      batch_size: int = 64) -> np.ndarray:
    """Delta de logit quand on neutralise une position. Reste sur la variété,
    contrairement au gradient qui est une linéarisation locale."""
    model.eval().to(device)
    acc, cnt = np.zeros(n_bins), np.zeros(n_bins)

    base = []
    for s in range(0, len(X), batch_size):
        base.append(model(X[s:s + batch_size].to(device), L[s:s + batch_size]).cpu())
    base = torch.cat(base).numpy()

    for b in range(n_bins):
        Xo = X.clone()
        touched = []
        for i, n in enumerate(L.tolist()):
            if n < n_bins:
                continue
            t = min(n - 1, int((b + 0.5) / n_bins * n))
            Xo[i, t] = 0.0                                   # même constante que l'ablation
            touched.append(i)

        out = []
        for s in range(0, len(Xo), batch_size):
            out.append(model(Xo[s:s + batch_size].to(device), L[s:s + batch_size]).cpu())
        out = torch.cat(out).numpy()

        idx = np.array(touched)
        acc[b] = np.abs(base[idx] - out[idx]).mean()
        cnt[b] = len(idx)

    return acc