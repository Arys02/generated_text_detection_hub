from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from .ablation import coalition_label, interaction_index, powerset, shapley_exact

LOG2 = float(np.log(2.0))


def _nats(y: np.ndarray, p: np.ndarray) -> float:
    """ln2 - CE. Suppose des classes équilibrées DANS le sous-groupe :
    sinon la référence d'ignorance n'est plus ln2 (voir _entropy)."""
    return LOG2 - log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1])


def _entropy(y: np.ndarray) -> float:
    """Entropie du label dans le sous-groupe. Référence correcte quand la
    stratification déséquilibre les classes."""
    q = float(y.mean())
    if q in (0.0, 1.0):
        return 0.0
    return -(q * np.log(q) + (1 - q) * np.log(1 - q))


def load_predictions(npz_path: Path, players: tuple[str, ...],
                     seeds=(0, 1, 2, 3, 4)) -> tuple[np.ndarray, dict]:
    z = np.load(npz_path)
    y = z["y_true"]
    preds = {}
    for keep in powerset(players):
        c = coalition_label(keep, players)
        for s in seeds:
            key = f"{c}__{s}"
            if key not in z.files:
                raise KeyError(f"prédiction manquante : {key}")
            preds[(keep, s)] = z[key]
    return y, preds


def stratified_shapley(y: np.ndarray, preds: dict, meta: pd.DataFrame,
                       players: tuple[str, ...], by: list[str],
                       seeds=(0, 1, 2, 3, 4), min_n: int = 200) -> pd.DataFrame:
    """Valeurs de Shapley recalculées sur chaque sous-groupe. Aucun réentraînement :
    les modèles restent ceux entraînés sur le corpus complet, seule l'ÉVALUATION
    est stratifiée. On mesure donc où le signal de chaque groupe s'exprime, pas
    ce qu'un modèle spécialisé apprendrait."""
    rows = []
    for key, idx in meta.groupby(by, sort=True).indices.items():
        if len(idx) < min_n:
            continue
        key = key if isinstance(key, tuple) else (key,)
        ys = y[idx]
        if len(np.unique(ys)) < 2:
            continue
        h = _entropy(ys)

        for s in seeds:
            v = {c: _nats(ys, preds[(c, s)][idx]) for c in powerset(players)}
            # recentrage sur l'entropie locale : ln2 n'est la bonne référence
            # que si le sous-groupe est équilibré
            v = {c: x - (LOG2 - h) for c, x in v.items()}
            phi = shapley_exact(v, players)
            inter = interaction_index(v, players)
            rows.append({**dict(zip(by, key)), "seed": s, "n": len(idx),
                         "H": h, **phi,
                         "v_full": v[frozenset(players)], "v_empty": v[frozenset()],
                         **{f"{a}x{b}": x for (a, b), x in inter.items()}})

    df = pd.DataFrame(rows)
    num = [c for c in df.columns if c not in by + ["seed"]]
    return df.groupby(by)[num].agg(["mean", "std"]).round(4)