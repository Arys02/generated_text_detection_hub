# file: src/detector/subjectivity_aware_generation_estimator/xai/direction.py
from __future__ import annotations

import numpy as np
import pandas as pd

SA_COL = "rule_based_SA-label"
RAW_COL = "raw_prediction_1_5"


def subjectivity_pools(df: pd.DataFrame, scored: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Scores observés par catégorie, pour rééchantillonner de façon plausible."""
    raw = pd.to_numeric(df[RAW_COL], errors="coerce")
    pools = {}
    for c in scored:
        vals = raw[(df[SA_COL] == c) & raw.notna()].to_numpy(dtype=float)
        if len(vals):
            pools[c] = vals
    return pools


def convert_toward(df: pd.DataFrame, target: str, fraction: float,
                   scored: tuple[str, ...], pools: dict[str, np.ndarray],
                   rng: np.random.Generator) -> pd.DataFrame:
    """Convertit une fraction de phrases tirées au hasard vers `target`.

    Intervention dirigée, par opposition à la permutation : elle donne le SENS de
    l'effet, que la PFI ne fournit pas (elle mesure l'ampleur, pas la direction).
    Le score de subjectivité est réconcilié avec la nouvelle catégorie pour éviter
    des combinaisons impossibles (une phrase non scorable portant un score).
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction hors (0, 1] : {fraction}")

    old = df[SA_COL].to_numpy(dtype=object)
    raw = pd.to_numeric(df[RAW_COL], errors="coerce").to_numpy(dtype=float)
    n = len(old)

    picked = rng.random(n) < fraction
    new = old.copy()
    new[picked] = target

    changed = picked & (old != target)
    if target not in scored:
        raw = raw.copy()
        raw[changed] = np.nan
    else:
        raw = raw.copy()
        need = changed & np.isnan(raw)
        if need.any():
            if target not in pools:
                raise ValueError(f"aucun score observé pour {target!r}")
            raw[need] = rng.choice(pools[target], size=int(need.sum()), replace=True)

    out = df.copy()
    out[SA_COL] = new
    out[RAW_COL] = raw
    return out