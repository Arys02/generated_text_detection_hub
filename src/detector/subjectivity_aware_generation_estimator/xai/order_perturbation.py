# file: src/detector/subjectivity_aware_generation_estimator/xai/order_perturbation.py
from __future__ import annotations

import numpy as np
import pandas as pd

SA_COL = "rule_based_SA-label"
RAW_COL = "raw_prediction_1_5"
ORDER_COL = "__order__"


def add_order(df: pd.DataFrame) -> pd.DataFrame:
    """Rang de chaque phrase dans sa relecture. L'ordre d'origine est celui des lignes,
    produit par explode_into_sentences ; on le matérialise pour pouvoir le manipuler."""
    out = df.copy()
    out[ORDER_COL] = out.groupby(["label", "review-id"], sort=False).cumcount()
    return out


def shuffle_within_review(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Mélange l'ordre des phrases à l'intérieur de chaque relecture.
    La composition (nombre de phrases par catégorie, scores) est strictement conservée :
    seule l'information d'ordre est détruite."""
    out = df.copy()
    idx = out.index.to_numpy()
    perm = np.concatenate([
        rng.permutation(idx[pos]) for pos in
        out.groupby(["label", "review-id"], sort=False).indices.values()
    ])
    return out.loc[perm].reset_index(drop=True)


def reverse_within_review(df: pd.DataFrame, rng: np.random.Generator = None) -> pd.DataFrame:
    """Inverse l'ordre. Toute l'information d'ordre est conservée, mais la dernière
    phrase devient la première : isole le biais de récence de l'information d'ordre."""
    out = df.copy()
    idx = out.index.to_numpy()
    rev = np.concatenate([idx[pos][::-1] for pos in
                          out.groupby(["label", "review-id"], sort=False).indices.values()])
    return out.loc[rev].reset_index(drop=True)


def keep_tail(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Ne garde que les k dernières phrases de chaque relecture."""
    d = add_order(df)
    n = d.groupby(["label", "review-id"], sort=False)[ORDER_COL].transform("max") + 1
    return d[d[ORDER_COL] >= (n - k)].drop(columns=ORDER_COL).reset_index(drop=True)


def keep_head(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Ne garde que les k premières phrases de chaque relecture."""
    d = add_order(df)
    return d[d[ORDER_COL] < k].drop(columns=ORDER_COL).reset_index(drop=True)