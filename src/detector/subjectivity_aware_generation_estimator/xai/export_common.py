# file: src/detector/subjectivity_aware_generation_estimator/xai/export_common.py
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src.detector.subjectivity_aware_generation_estimator.xai.results_store import load

METRIC_LABELS = {
    "auc": "AUROC",
    "acc": "Accuracy",
    "tpr@fpr1": r"TPR@1\%FPR",
    "tpr@fpr0_1": r"TPR@0.1\%FPR",
    "dose": "Perturbed share",
    "p_mgt_moyen": r"Mean $\hat p_{\mathrm{MGT}}$",
    "variation": r"$\Delta \hat p_{\mathrm{MGT}}$",
}
METRIC_LABELS_PLAIN = {k: re.sub(r"[\\$]|\\mathrm|[{}]", "", v).replace("%", "%")
                       for k, v in METRIC_LABELS.items()}

CONDITION_LABELS = {
    "reference": "Full model",
    "ordre_intact": "Full model",
    "v3_multi_novalid": "Full model",
    # nommés par l'information DÉTRUITE, pour que PFI et LOFO partagent les mêmes lignes
    "categories_seules": "Subjectivity removed",     # LOFO : bloc subjectivité retiré
    "subjectivite_seule": "Speech acts removed",     # LOFO : bloc catégories retiré
    "bloc_subjectivite": "Subjectivity removed",     # PFI  : bloc subjectivité perturbé
    "bloc_categories": "Speech acts removed",        # PFI  : bloc catégories perturbé
    "melange_train_et_test": "Shuffled (train + test)",
    "melange_train_seul": "Shuffled (train only)",
    "melange_test_seul": "Shuffled (test only)",
    "inversion_test_seul": "Reversed (test only)",
    "__ALL__": "All speech acts",
    "bruit_TOUTES": "All speech acts",
}

_PREFIX = {"fusion_": "", "bruit_": "", "v3_merge_": ""}


def label_condition(name: str) -> str:
    if name in CONDITION_LABELS:
        return CONDITION_LABELS[name]
    for pre in _PREFIX:
        if name.startswith(pre):
            return name[len(pre):].capitalize()
    return name.replace("_", " ").capitalize()


def latest_store(protocol: str = "paired", drop_duplicates: bool = True) -> pd.DataFrame:
    """Magasin filtré. Si une expérience a été enregistrée plusieurs fois (relance du
    notebook, renommage de conditions), on ne garde que le run le plus récent."""
    d = load()
    if d.empty:
        return d
    d = d[d["protocol"] == protocol].copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    if drop_duplicates:
        keep = (d.sort_values("timestamp").groupby("experiment")["run_id"].last())
        d = d[d["run_id"].isin(set(keep))]
    d["label"] = d["condition"].map(label_condition)
    return d


def deltas(d: pd.DataFrame, experiment: str, metric: str, reference: str,
           lo: float = 0.05, hi: float = 0.95) -> pd.DataFrame:
    """Importance par condition : delta contre la référence, quantiles et erreur-type.

    Convention Molnar : pour une métrique où plus grand est meilleur (AUROC, accuracy),
    on inverse le sens de la différence, d'où importance = ref - condition.
    """
    sub = d[(d["experiment"] == experiment) & (d["metric"] == metric)]
    if sub.empty:
        raise ValueError(f"aucune mesure pour ({experiment}, {metric})")
    ref_vals = sub.loc[sub["condition"] == reference, "value"].to_numpy()
    if ref_vals.size == 0:
        raise ValueError(f"condition de référence {reference!r} absente de {experiment!r}")
    ref_mean, ref_var, ref_n = ref_vals.mean(), ref_vals.var(ddof=1), len(ref_vals)

    rows = []
    for cond, g in sub.groupby("condition"):
        v = g["value"].to_numpy()
        imp = ref_mean - v
        rows.append({
            "condition": cond, "label": label_condition(cond),
            "value_mean": v.mean(), "value_std": v.std(ddof=1) if len(v) > 1 else np.nan,
            "importance": imp.mean(),
            "q_lo": np.quantile(imp, lo), "q_hi": np.quantile(imp, hi),
            "se": np.sqrt((v.var(ddof=1) / len(v) if len(v) > 1 else 0.0) + ref_var / ref_n),
            "n": len(v),
        })
    out = pd.DataFrame(rows)
    out["significant"] = np.where(out["importance"].abs() > 2 * out["se"], "yes", "no")
    out.loc[out["condition"] == reference, "significant"] = "—"
    return out.sort_values("importance", ascending=False).reset_index(drop=True)