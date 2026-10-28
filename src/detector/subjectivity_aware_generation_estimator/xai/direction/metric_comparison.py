from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from .ablation import interaction_index, powerset, shapley_exact
from .accuracy_shapley import best_threshold

LOG2 = math.log(2.0)


def _value(y: np.ndarray, p: np.ndarray, metric: str, thr: float = 0.5) -> float:
    """Chaque métrique est recentrée sur SA propre référence d'ignorance,
    pour que les trois tables soient lisibles sur la même échelle relative."""
    if metric == "auc":
        return float(roc_auc_score(y, p)) - 0.5
    if metric == "nats":
        return LOG2 - float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1]))
    if metric == "accuracy":
        base = max(y.mean(), 1 - y.mean())
        return float(((p >= thr).astype(int) == y).mean()) - base
    raise ValueError(metric)


def shapley_by_metric(y: np.ndarray, preds: dict, players: tuple[str, ...],
                      metric: str, seeds=(0, 1, 2, 3, 4), thr: float = 0.5) -> dict:
    rows_phi, rows_int, rows_v = [], [], []
    for s in seeds:
        v = {c: _value(y, preds[(c, s)], metric, thr) for c in powerset(players)}
        phi = shapley_exact(v, players)
        assert abs(sum(phi.values()) - (v[frozenset(players)] - v[frozenset()])) < 1e-9

        rows_phi.append({"seed": s, **phi,
                         "v_full": v[frozenset(players)], "v_empty": v[frozenset()]})
        rows_int.append({"seed": s,
                         **{f"{a}x{b}": x for (a, b), x in interaction_index(v, players).items()}})
        rows_v.append({"seed": s, **{("+".join(sorted(c)) or "-"): x for c, x in v.items()}})

    phi = pd.DataFrame(rows_phi).set_index("seed")
    return {"phi": phi,
            "summary": pd.DataFrame({"mean": phi.mean(), "std": phi.std()}).round(4),
            "interactions": pd.DataFrame(rows_int).set_index("seed"),
            "coalitions": pd.DataFrame(rows_v).set_index("seed")}


def compare_metrics(y: np.ndarray, preds: dict, players: tuple[str, ...],
                    seeds=(0, 1, 2, 3, 4)) -> pd.DataFrame:
    """Table de synthèse : valeur, écart-type, part du budget, et rapport
    signal/bruit — c'est ce dernier qui justifie le choix de métrique."""
    out = []
    for metric in ("accuracy", "auc", "nats"):
        r = shapley_by_metric(y, preds, players, metric, seeds)
        s = r["summary"]
        budget = s.loc[list(players), "mean"].sum()
        for pl in players:
            m, sd = s.loc[pl, "mean"], s.loc[pl, "std"]
            out.append({"metrique": metric, "groupe": pl,
                        "phi": round(m, 4), "std": round(sd, 4),
                        "part_%": round(100 * m / budget, 1),
                        "snr": round(m / sd, 1) if sd > 0 else np.inf})
        out.append({"metrique": metric, "groupe": "— v(vide)",
                    "phi": round(s.loc["v_empty", "mean"], 4),
                    "std": round(s.loc["v_empty", "std"], 4),
                    "part_%": None, "snr": None})
    return pd.DataFrame(out)