# file: src/detector/subjectivity_aware_generation_estimator/xai/metrics.py
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

FPR_TARGETS = (0.01, 0.001)


def threshold_at_fpr(y: np.ndarray, p: np.ndarray, target_fpr: float) -> tuple[float, float, float]:
    """Seuil le plus permissif dont le FPR ne dépasse pas la cible.

    Retourne (seuil, fpr_atteint, tpr). roc_curve renvoie les points par seuil
    décroissant ; on prend le dernier point sous la contrainte, donc le TPR maximal.
    """
    fpr, tpr, thr = roc_curve(y, p)
    ok = np.where(fpr <= target_fpr)[0]
    if len(ok) == 0:
        return float("inf"), 0.0, 0.0
    i = ok[-1]
    t = float(thr[i])
    # roc_curve place en tête un seuil artificiel (inf ou max+1) : on le ramène au-dessus du max
    if not np.isfinite(t):
        t = float(p.max()) + 1.0
    return t, float(fpr[i]), float(tpr[i])


def detection_metrics(y: np.ndarray, p: np.ndarray, threshold: float = 0.5,
                      fpr_targets: tuple[float, ...] = FPR_TARGETS) -> dict:
    """AUC et accuracy au seuil courant, plus TPR et accuracy à FPR contraint.

    Les seuils à FPR fixé sont calibrés sur le jeu évalué lui-même : c'est une borne
    optimiste, à ne comparer qu'entre conditions traitées de la même façon.
    """
    out = {"auc": float(roc_auc_score(y, p)),
           "acc": float(accuracy_score(y, (p >= threshold).astype(int)))}
    for target in fpr_targets:
        t, fpr_eff, tpr = threshold_at_fpr(y, p, target)
        tag = f"{target:g}".replace("0.", "").rstrip() or "0"
        key = f"fpr{target * 100:g}".replace(".", "_")
        out[f"tpr@{key}"] = tpr
        out[f"acc@{key}"] = float(accuracy_score(y, (p >= t).astype(int)))
        out[f"fpr_eff@{key}"] = fpr_eff
    return out


def metrics_frame(y: np.ndarray, p: np.ndarray, threshold: float = 0.5,
                  fpr_targets: tuple[float, ...] = FPR_TARGETS) -> pd.DataFrame:
    return pd.DataFrame([detection_metrics(y, p, threshold, fpr_targets)])