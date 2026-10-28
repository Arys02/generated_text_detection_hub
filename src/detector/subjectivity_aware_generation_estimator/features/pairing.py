# file: src/detector/subjectivity_aware_generation_estimator/features/pairing.py
from __future__ import annotations

import hashlib

import pandas as pd

REVIEW_ID = "review-id"
LABEL_COL = "label"
GEN_COL = "generator"


def _stable_choice(key: str, options: list[str], seed: int) -> str:
    """Choix déterministe et reproductible entre processus.
    hash() de Python est randomisé par PYTHONHASHSEED : inutilisable ici."""
    h = hashlib.md5(f"{seed}:{key}".encode()).hexdigest()
    return options[int(h, 16) % len(options)]


def pair_and_balance(df: pd.DataFrame, seed: int = 0, verbose: bool = True) -> pd.DataFrame:
    """Un bloc HWT et un bloc MGT par review-id, le générateur MGT choisi de
    façon déterministe. Garantit l'équilibre exact et l'unicité de la clé de
    groupement, donc l'absence de séquences chimériques."""
    hwt = df[df[LABEL_COL] == 0]
    mgt = df[df[LABEL_COL] == 1]

    # côté HWT, la colonne generator est le fichier d'origine : sans signification
    dup = hwt.drop_duplicates([REVIEW_ID, GEN_COL]).groupby(REVIEW_ID).size()
    if (dup > 1).any():
        raise ValueError(f"{int((dup > 1).sum())} review-id HWT présents sous plusieurs générateurs")

    avail = (mgt[[REVIEW_ID, GEN_COL]].drop_duplicates()
                .groupby(REVIEW_ID)[GEN_COL].apply(lambda s: sorted(s)))
    common = sorted(set(hwt[REVIEW_ID]) & set(avail.index))
    if not common:
        raise ValueError("aucun review-id commun entre HWT et MGT : appariement impossible")

    chosen = {rid: _stable_choice(str(rid), avail.loc[rid], seed) for rid in common}

    keep_hwt = hwt[hwt[REVIEW_ID].isin(chosen)]
    keep_mgt = mgt[mgt[REVIEW_ID].map(chosen).eq(mgt[GEN_COL])]

    out = pd.concat([keep_hwt, keep_mgt], ignore_index=True)

    n_blocks = out.groupby([REVIEW_ID, LABEL_COL]).ngroups
    assert n_blocks == 2 * len(common), f"{n_blocks} blocs pour {len(common)} paires"

    if verbose:
        drop_h = hwt[REVIEW_ID].nunique() - len(common)
        drop_m = mgt.groupby([REVIEW_ID, GEN_COL]).ngroups - len(common)
        print(f"appariement : {len(common)} paires | HWT écartés {drop_h} | versions MGT écartées {drop_m}")
        print("générateurs retenus :", pd.Series(list(chosen.values())).value_counts().to_dict())

    return out