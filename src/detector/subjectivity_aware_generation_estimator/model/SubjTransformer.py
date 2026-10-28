# file: model/SubjTransformer.py
from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor


def lengths_to_padding_mask(lengths: Tensor, max_len: int, device: torch.device) -> Tensor:
    """True = position ignorée. Convention nn.MultiheadAttention (src_key_padding_mask)."""
    ar = torch.arange(max_len, device=device)
    # clamp(min=1) : une séquence de longueur 0 masquerait toute la ligne -> NaN au softmax
    return ar.unsqueeze(0) >= lengths.to(device).clamp(min=1).unsqueeze(1)


class SinusoidalPE(nn.Module):
    """PE non paramétrique : extrapole au-delà du L_max vu à l'entraînement."""
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        assert d_model % 2 == 0, "d_model doit être pair"
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                        * (-math.log(10_000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pe[:, : x.size(1)]


class LearnedPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        self.emb = nn.Parameter(torch.empty(max_len, d_model))
        nn.init.normal_(self.emb, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        L = x.size(1)
        if L > self.emb.size(0):
            raise ValueError(f"L={L} dépasse max_len={self.emb.size(0)} (PE apprise)")
        return x + self.emb[:L].unsqueeze(0)


class AttentionPooling(nn.Module):
    """Pooling additif (Bahdanau). Les poids sont exploitables comme trace d'explication :
    quelles phrases portent la décision MGT/HWT."""
    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(d_model, d_model), nn.Tanh(),
            nn.Dropout(dropout), nn.Linear(d_model, 1),
        )

    def forward(self, h: Tensor, pad_mask: Tensor) -> Tuple[Tensor, Tensor]:
        s = self.score(h).squeeze(-1)                              # (B, L)
        s = s.masked_fill(pad_mask, torch.finfo(s.dtype).min)
        w = torch.softmax(s, dim=1)                                # (B, L)
        return torch.einsum("bl,bld->bd", w, h), w


class SubjectivityTransformer(nn.Module):
    """Encodeur Transformer sur la séquence de phrases d'une review.

    Entrée : (B, L, input_dim) où input_dim = |speech_act_cats| + 2
             (one-hot acte de langage, degré de subjectivité z-scoré, flag de validité).
    Sortie : logits (B,) — compatible BCEWithLogitsLoss.
    """

    def __init__(
        self,
        input_dim: int = 7,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        ff_mult: int = 4,
        dropout: float = 0.1,
        input_dropout: float = 0.0,
        pooling: str = "attention",          # {"attention", "mean", "cls"}
        pos_encoding: str = "sinusoidal",    # {"sinusoidal", "learned", "none"}
        max_len: int = 512,
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model doit être divisible par nhead")
        self.input_dim, self.d_model, self.pooling = input_dim, d_model, pooling

        self.in_proj = nn.Sequential(nn.Linear(input_dim, d_model), nn.LayerNorm(d_model))
        self.in_drop = nn.Dropout(input_dropout)

        if pos_encoding == "sinusoidal":
            self.pos = SinusoidalPE(d_model, max_len + 1)
        elif pos_encoding == "learned":
            self.pos = LearnedPE(d_model, max_len + 1)
        elif pos_encoding == "none":
            self.pos = nn.Identity()         # -> encodeur d'ensemble, invariant par permutation
        else:
            raise ValueError(f"pos_encoding inconnu: {pos_encoding}")

        self.cls: Optional[nn.Parameter] = None
        if pooling == "cls":
            self.cls = nn.Parameter(torch.empty(1, 1, d_model))
            nn.init.normal_(self.cls, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ff_mult * d_model,
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True,   # Pre-LN : pas de warmup obligatoire, gradients stables
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=num_layers, norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,          # incompatible norm_first, évite le fallback silencieux
        )

        self.attn_pool = AttentionPooling(d_model, dropout) if pooling == "attention" else None
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, 1))

    def forward(
        self, x: Tensor, lengths: Tensor, return_attention: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Optional[Tensor]]]:
        B, L, _ = x.shape
        pad = lengths_to_padding_mask(lengths, L, x.device)         # (B, L)

        h = self.in_drop(self.in_proj(x))
        if self.cls is not None:
            h = torch.cat([self.cls.expand(B, -1, -1), h], dim=1)
            pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=x.device), pad], dim=1)

        h = self.pos(h)
        h = self.encoder(h, src_key_padding_mask=pad)

        if self.pooling == "cls":
            pooled, w = h[:, 0], None
        elif self.pooling == "mean":
            keep = (~pad).unsqueeze(-1).to(h.dtype)
            pooled, w = (h * keep).sum(1) / keep.sum(1).clamp(min=1.0), None
        else:
            pooled, w = self.attn_pool(h, pad)

        logits = self.head(pooled).squeeze(-1)                      # (B,)
        return (logits, w) if return_attention else logits

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)