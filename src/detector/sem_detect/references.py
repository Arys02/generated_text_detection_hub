# src/detector/sem_detect/references.py
"""Protocoles d'appariement cible -> références IA, matrice de features et probabilités."""
from __future__ import annotations

import hashlib
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.detector.sem_detect.feature_store import SemDetectFeatureStore
from src.detector.sem_detect.semdetect.feature_builder import MIN_EMBEDDINGS, build_feature_vector

PAIR_KEYS = ["conference", "year", "submission_id", "rating"]
MIN_SOURCES = 3  # scripts/evaluate.py


def _stable_unit(key: str, seed: int) -> float:
    h = hashlib.md5(f"{seed}:{key}".encode()).hexdigest()
    return int(h[:15], 16) / float(16 ** 15)


def paper_protocol_references(
    df: pd.DataFrame,
    k: int = 3,
    seed: int = 0,
    is_valid: Optional[Callable[[str], bool]] = None,
    key_cols: Optional[List[str]] = None,
) -> Dict[str, List[str]]:
    """Papier §3.2 : k reviews IA de la même clé (par défaut même papier et même note), une par modèle ;
    si la cible est IA, son propre modèle est exclu. Tirage déterministe parmi les candidats.
    is_valid : si fourni, les candidates invalides (ex. sans claim sémantique) sont écartées AVANT
    le tirage ; quand toutes les candidates sont valides, le résultat est identique à is_valid=None.
    key_cols : colonnes de regroupement (APRDB : ["group"], le review-id partagé par l'humaine et ses jumelles)."""
    key_cols = list(key_cols) if key_cols is not None else PAIR_KEYS
    ai = df[df["class"] == "ai"]
    pools = {key: list(zip(g["review-id"], g["generator"]))
             for key, g in ai.groupby(key_cols, sort=False)}
    refs: Dict[str, List[str]] = {}
    for rid, cls, gen, key in zip(df["review-id"], df["class"], df["generator"],
                                  df[key_cols].itertuples(index=False, name=None)):
        cands = [(r, g) for r, g in pools.get(key, []) if r != rid and not (cls == "ai" and g == gen)]
        if is_valid is not None:
            cands = [(r, g) for r, g in cands if is_valid(r)]
        by_model: Dict[str, str] = {}
        for r, g in sorted(cands, key=lambda c: _stable_unit(f"{rid}|{c[0]}", seed)):
            by_model.setdefault(g, r)  # au plus une référence par modèle
        refs[rid] = sorted(by_model.values(), key=lambda r: _stable_unit(f"{rid}|{r}", seed))[:k]
    return refs



def build_feature_matrix(
    store: SemDetectFeatureStore,
    target_ids: List[str],
    refs: Dict[str, List[str]],
    similarity_threshold: float,
    min_sources: int = MIN_SOURCES,
) -> Tuple[np.ndarray, List[str], Dict[str, str]]:
    """Mêmes règles d'exclusion que scripts/evaluate.py ; les exclusions sont renvoyées, pas ignorées."""
    X, kept, skipped = [], [], {}
    for rid in target_ids:
        target = store.semantic(rid, MIN_EMBEDDINGS)
        if target is None:
            skipped[rid] = f"cible < {MIN_EMBEDDINGS} claims sémantiques"
            continue
        sources = [s for s in (store.semantic(r, 1) for r in refs.get(rid, [])) if s is not None]
        if len(sources) < min_sources:
            skipped[rid] = f"références insuffisantes ({len(sources)}/{min_sources})"
            continue
        feats = build_feature_vector(target, sources, store.textual(rid), similarity_threshold)
        if feats is None:
            skipped[rid] = "échec du calcul des features"
            continue
        X.append(feats)
        kept.append(rid)
    return np.asarray(X, dtype=float).reshape(-1, 9), kept, skipped


def predict_proba(clf, X: np.ndarray) -> np.ndarray:
    """Colonnes dans l'ordre de CLASS_NAMES = [ai, rewrite, human] (indice 0 = IA).
    Sans SHAP : inutile pour l'évaluation et coûteux en lot."""
    Xs = clf.scaler.transform(X)
    if clf._feature_names:
        Xs = pd.DataFrame(Xs, columns=clf._feature_names)
    return clf.model.predict_proba(Xs)

def fixed_models_references(
    targets: pd.DataFrame,
    pool: pd.DataFrame,
    models: List[str],
    key_cols: Tuple[str, ...] = ("submission_id", "rating"),
    is_valid: Optional[Callable[[str], bool]] = None,
) -> Dict[str, List[str]]:
    """Protocole sans recours aux labels : chaque cible, quelle que soit sa classe, reçoit la référence
    de chacun des `models` pour sa clé (papier, note). pool : colonnes review-id, generator + key_cols.
    Les références invalides (is_valid) sont écartées ; aucune substitution (une seule par modèle et par clé)."""
    key_cols = list(key_cols)
    index: Dict[tuple, Dict[str, str]] = {}
    for rid, gen, key in zip(pool["review-id"], pool["generator"], pool[key_cols].itertuples(index=False, name=None)):
        if gen not in models:
            continue
        slot = index.setdefault(key, {})
        if gen in slot:
            raise ValueError(f"plusieurs références {gen!r} pour la clé {key} : {slot[gen]} et {rid}")
        if is_valid is None or is_valid(rid):
            slot[gen] = rid
    return {
        rid: [index.get(key, {})[m] for m in models if m in index.get(key, {})]
        for rid, key in zip(targets["review-id"], targets[key_cols].itertuples(index=False, name=None))
    }