# file: src/detector/subjectivity_aware_generation_estimator/features/tensor_builder.py
from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence

from .feature_spec import _BUILDERS, FeatureSpec

REVIEW_ID = "review-id"
LABEL_COL = "label"
SUBJ_COL = "raw_prediction_1_5"

GROUP_KEYS = [REVIEW_ID, "generator"]

# --------------------------------------------------------------------------- #
# Contexte : tout ce qui est AJUSTÉ sur le train et transporté vers le test
# --------------------------------------------------------------------------- #

def fit_context(df_train: pd.DataFrame, spec: FeatureSpec) -> dict:
    """Estime les statistiques de normalisation. À n'appeler QUE sur le train."""
    # astype(float) explicite : la colonne sort de score_subjectivity_optimized
    # en dtype object, et .mean() sur object dépend de la version de pandas
    subj = df_train[SUBJ_COL].astype(float)
    mean, std = float(subj.mean()), float(subj.std())  # ddof=1, convention pandas
    if not np.isfinite(mean) or not np.isfinite(std) or std == 0.0:
        raise ValueError(f"statistiques dégénérées : mean={mean}, std={std}")
    return {"label_cats": spec.label_cats, "subj_mean": mean, "subj_std": std}


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #
def block_to_matrix(block: pd.DataFrame, spec: FeatureSpec, ctx: dict) -> np.ndarray:
    """(L, input_dim) float64. L'ordre des groupes dans la spec fixe celui des colonnes.

    Contrat des builders : (block, ctx, group) -> (L, group.width).
    Le groupe porte sa propre configuration d'encodage — merge, merge_into — ce qui
    permet plusieurs groupes du même builder avec des paramètres distincts.
    """
    parts = []
    for g in spec.groups:
        part = _BUILDERS[g.builder](block, ctx, g)
        if part.shape != (len(block), g.width):
            raise ValueError(f"builder {g.builder!r} du groupe {g.name!r} a rendu "
                             f"{part.shape}, attendu {(len(block), g.width)}")
        parts.append(part)
    mat = np.concatenate(parts, axis=1)
    if mat.shape[1] != spec.input_dim:
        raise ValueError(f"largeur {mat.shape[1]} != input_dim {spec.input_dim}")
    return mat




def _iter_blocks(df, order_col):
    keys = [k for k in GROUP_KEYS if k in df.columns]
    for key, block in df.groupby(keys, sort=False):
        yield (key if len(keys) > 1 else (key,))[0], (block.sort_values(order_col) if order_col else block)
        
def _assert_unambiguous(df: pd.DataFrame) -> None:
    """Corpus apparié : un review-id porte une version HWT et une version MGT.
    Les grouper ensemble produirait une séquence chimérique, sans erreur."""
    if LABEL_COL in df.columns and df.groupby(REVIEW_ID)[LABEL_COL].nunique().gt(1).any():
        raise ValueError(
            f"{REVIEW_ID} ambigu : plusieurs labels par id. Filtrer par label en amont."
        )


def build_sequences(
    df: pd.DataFrame, spec: FeatureSpec, ctx: dict, order_col: str | None = None
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Un bloc par review-id. Exige un label constant par id (inférence)."""
    _assert_unambiguous(df)
    seqs, lengths, ids = [], [], []
    for rid, block in _iter_blocks(df, order_col):
        m = block_to_matrix(block, spec, ctx)
        seqs.append(torch.tensor(m, dtype=torch.float32))
        lengths.append(len(m))
        ids.append(rid)
    if not seqs:
        raise ValueError("aucune séquence construite : DataFrame vide ?")
    return (pad_sequence(seqs, batch_first=True, padding_value=0.0),
            torch.tensor(lengths, dtype=torch.int64),
            np.array(ids))


def build_dataset(
    df: pd.DataFrame, spec: FeatureSpec, ctx: dict, order_col: str | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    """Blocs regroupés par label croissant (0/1, ou 0/1/2 en 3 classes), L_max GLOBAL."""
    labels_present = sorted(int(v) for v in df[LABEL_COL].dropna().unique())
    seqs, lengths, labels, ids = [], [], [], []
    for lab in labels_present:
        part = df[df[LABEL_COL] == lab]
        for rid, block in _iter_blocks(part, order_col):
            m = block_to_matrix(block, spec, ctx)
            seqs.append(torch.tensor(m, dtype=torch.float32))
            lengths.append(len(m))
            labels.append(lab)
            ids.append(rid)
    if not seqs:
        raise ValueError("aucune séquence construite")
    return (pad_sequence(seqs, batch_first=True, padding_value=0.0),
            torch.tensor(lengths, dtype=torch.int64),
            torch.tensor(labels, dtype=torch.int64),
            np.array(ids))


# --------------------------------------------------------------------------- #
# Ablation — le layout vient de la spec, jamais d'un slice écrit à la main
# --------------------------------------------------------------------------- #

def ablate(X: torch.Tensor, spec: FeatureSpec, keep) -> torch.Tensor:
    """Remplace par sa constante chaque groupe hors coalition.
    input_dim préservé : le compte de paramètres du modèle reste identique."""
    keep = frozenset(keep)
    unknown = keep - set(spec.players)
    if unknown:
        raise KeyError(f"groupes inconnus dans la coalition : {sorted(unknown)}")
    layout = spec.layout
    Xa = X.clone()
    for g in spec.groups:
        if g.name not in keep:
            Xa[:, :, layout[g.name]] = g.ablation_value
    return Xa