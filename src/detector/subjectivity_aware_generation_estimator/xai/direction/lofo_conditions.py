# file: src/detector/subjectivity_aware_generation_estimator/xai/lofo_conditions.py
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

SA_COL = "rule_based_SA-label"
RAW_COL = "raw_prediction_1_5"
NEUTRAL = "Assertion"   # cible de fusion : catégorie dominante, jamais supprimée


def _drop_subjectivity(df: pd.DataFrame) -> pd.DataFrame:
    """Retire le degré de subjectivité. valid passe à 0 partout, la colonne devient
    constante : le modèle ne voit plus que le one-hot des actes de langage."""
    out = df.copy()
    out[RAW_COL] = np.nan
    return out


def _collapse_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Fusionne toutes les catégories en une seule. Le one-hot devient constant ;
    seuls subsistent le degré de subjectivité et le flag valid."""
    out = df.copy()
    out[SA_COL] = NEUTRAL
    return out


def merge_category(df: pd.DataFrame, target: str, into: str = NEUTRAL) -> pd.DataFrame:
    """Fusionne `target` dans `into` : le modèle ne peut plus les distinguer.
    Le score de subjectivité est conservé — c'est une propriété du texte, pas de l'étiquette."""
    if target == into:
        raise ValueError(f"fusion de {target!r} dans elle-même")
    out = df.copy()
    out.loc[out[SA_COL] == target, SA_COL] = into
    return out


def build_conditions(label_cats: list[str], with_merges: bool = True
                     ) -> dict[str, Callable[[pd.DataFrame], pd.DataFrame]]:
    """Conditions LOFO. Chaque entrée transforme le DataFrame de phrases, à appliquer
    identiquement au train et au test."""
    conditions: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
        "reference": lambda d: d,
        "categories_seules": _drop_subjectivity,
        "subjectivite_seule": _collapse_categories,
    }
    if with_merges:
        for c in label_cats:
            if c != NEUTRAL:
                conditions[f"fusion_{c}"] = (lambda d, c=c: merge_category(d, c, NEUTRAL))
    return conditions

# file: src/detector/subjectivity_aware_generation_estimator/xai/lofo_conditions.py
# — à ajouter à la fin du fichier

SCORED_PLACEHOLDER = 0.0   # valeur arbitraire : seul compte le fait que le score existe


def _degree_only(df: pd.DataFrame) -> pd.DataFrame:
    """Catégories fusionnées, et le flag valid rendu constant à 1 : toutes les phrases
    portent un score. Les phrases non scorées reçoivent la moyenne du train, qui après
    z-scoring vaut 0 — soit la même valeur que l'imputation par défaut.
    Isole le degré continu de l'information de présence."""
    out = _collapse_categories(df)
    raw = pd.to_numeric(out[RAW_COL], errors="coerce")
    out[RAW_COL] = raw.fillna(raw.mean() if raw.notna().any() else SCORED_PLACEHOLDER)
    return out


def _validity_only(df: pd.DataFrame) -> pd.DataFrame:
    """Catégories fusionnées, et tous les scores existants remplacés par une constante :
    seul subsiste le fait qu'une phrase soit scorée ou non, c'est-à-dire valid.
    Isole l'information de présence du degré continu."""
    out = _collapse_categories(df)
    raw = pd.to_numeric(out[RAW_COL], errors="coerce")
    out[RAW_COL] = np.where(raw.notna(), float(raw.mean()) if raw.notna().any() else 0.0, np.nan)
    return out


def build_subjectivity_conditions() -> dict[str, Callable[[pd.DataFrame], pd.DataFrame]]:
    """Décomposition du bloc subjectivité, catégories neutralisées dans les trois cas."""
    return {
        "reference": lambda d: d,
        "subjectivite_seule": _collapse_categories,
        "degre_seul": _degree_only,
        "validite_seule": _validity_only,
    }