from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from .ablation import interaction_index, powerset, shapley_exact
from .stratified_shapley import load_predictions


def best_threshold(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Seuil maximisant l'accuracy, par balayage exact sur les valeurs triées.
    Vectorisé : O(n log n), pas de boucle sur une grille arbitraire."""
    order = np.argsort(p, kind="stable")
    ys, ps = y[order], p[order]
    # accuracy si l'on coupe juste après l'indice k
    neg = np.cumsum(1 - ys)                    # négatifs à gauche, bien classés
    pos = ys.sum() - np.cumsum(ys)             # positifs à droite, bien classés
    acc = (neg + pos) / len(ys)
    k = int(np.argmax(acc))
    thr = ps[k] if k + 1 >= len(ps) else 0.5 * (ps[k] + ps[k + 1])
    return float(thr), float(acc[k])


def accuracy_shapley(y: np.ndarray, preds: dict, players: tuple[str, ...],
                     seeds=(0, 1, 2, 3, 4), mode: str = "calibrated",
                     split_seed: int = 0) -> dict:
    """mode='fixed'   : seuil 0.5, aucun ajustement.
       mode='calibrated' : seuil ajusté sur une moitié du test, reporté sur l'autre.
    """
    if mode == "calibrated":
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=split_seed)
        i_dev, i_eval = next(sss.split(np.zeros(len(y)), y))
    else:
        i_dev = i_eval = np.arange(len(y))

    rows_phi, rows_int, rows_v = [], [], []
    for s in seeds:
        v, thr_by_c = {}, {}
        for c in powerset(players):
            p = preds[(c, s)]
            if mode == "calibrated":
                thr, _ = best_threshold(y[i_dev], p[i_dev])
            else:
                thr = 0.5
            thr_by_c[c] = thr
            v[c] = float(((p[i_eval] >= thr).astype(int) == y[i_eval]).mean())

        # référence : le classificateur majoritaire, soit 0.5 sur classes équilibrées
        base = max(y[i_eval].mean(), 1 - y[i_eval].mean())
        vc = {c: x - base for c, x in v.items()}

        phi = shapley_exact(vc, players)
        gap = sum(phi.values()) - (vc[frozenset(players)] - vc[frozenset()])
        assert abs(gap) < 1e-9, f"efficacité violée : {gap}"

        rows_phi.append({"seed": s, **phi,
                         "v_full": vc[frozenset(players)], "v_empty": vc[frozenset()]})
        rows_int.append({"seed": s,
                         **{f"{a}x{b}": x for (a, b), x in interaction_index(vc, players).items()}})
        rows_v.append({"seed": s, **{("+".join(sorted(c)) or "-"): v[c] for c in v},
                       **{f"thr_{'+'.join(sorted(c)) or '-'}": thr_by_c[c] for c in v}})

    phi_df = pd.DataFrame(rows_phi).set_index("seed")
    return {"phi": phi_df,
            "summary": pd.DataFrame({"mean": phi_df.mean(), "std": phi_df.std()}).round(4),
            "interactions": pd.DataFrame(rows_int).set_index("seed"),
            "raw": pd.DataFrame(rows_v).set_index("seed"),
            "n_eval": len(i_eval)}