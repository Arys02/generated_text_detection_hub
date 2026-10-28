# file: src/detector/subjectivity_aware_generation_estimator/xai/category_permutation.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
# file: src/detector/subjectivity_aware_generation_estimator/xai/category_permutation.py
from typing import Callable, Optional

from src.detector.subjectivity_aware_generation_estimator.training.gru_features import build_features

SA_COL = "rule_based_SA-label"
RAW_COL = "raw_prediction_1_5"
ALL = "__ALL__"   # perturbation du bloc catégorie entier


@dataclass
class CategoryMarginals:
    label_cats: list[str]
    p: dict[str, float]            # prévalence de chaque catégorie sur l'ensemble des phrases
    pools: dict[str, np.ndarray]   # scores de subjectivité observés, par catégorie scorée
    scored: list[str]              # catégories recevant un score (subj_category)


def fit_marginals(df: pd.DataFrame, label_cats: list[str], scored: list[str]) -> CategoryMarginals:
    """Distribution marginale des catégories sur toutes les phrases du corpus."""
    counts = df[SA_COL].value_counts().reindex(label_cats, fill_value=0)
    if counts.sum() == 0:
        raise ValueError("aucune phrase dans le corpus")
    p = (counts / counts.sum()).to_dict()
    raw = pd.to_numeric(df[RAW_COL], errors="coerce")
    pools = {}
    for c in scored:
        vals = raw[(df[SA_COL] == c) & raw.notna()].to_numpy(dtype=float)
        if len(vals) > 0:
            pools[c] = vals
    return CategoryMarginals(list(label_cats), p, pools, list(scored))


def _reconcile_subjectivity(old: np.ndarray, new: np.ndarray, raw: np.ndarray,
                            m: CategoryMarginals, rng: np.random.Generator) -> np.ndarray:
    """Maintient la cohérence catégorie / subjectivité après changement de catégorie.
    - devient non scorée            -> score NaN, donc valid = 0
    - passe de non scorée à scorée  -> score tiré dans la distribution de la nouvelle catégorie
    - reste scorée                  -> score conservé (propriété du texte de la phrase)
    """
    raw = raw.copy()
    changed = old != new
    new_scored = np.isin(new, m.scored)
    old_scored = np.isin(old, m.scored)

    raw[changed & ~new_scored] = np.nan

    need = changed & new_scored & ~old_scored
    for c in np.unique(new[need]):
        mask = need & (new == c)
        if c not in m.pools:
            raise ValueError(f"aucun score de subjectivité observé pour {c!r}")
        raw[mask] = rng.choice(m.pools[c], size=int(mask.sum()), replace=True)
    return raw


def resample_category(df: pd.DataFrame, target: str, m: CategoryMarginals,
                      rng: np.random.Generator) -> pd.DataFrame:
    """Casse le lien entre l'indicateur 1[C = target] et l'étiquette, sur tout le corpus.

    Chaque phrase devient `target` avec probabilité p_target, indépendamment de sa
    catégorie d'origine : équivalent i.i.d. d'une permutation globale de l'indicateur.
    Les phrases qui perdent `target` reçoivent une catégorie tirée dans P(C | C != target).
    Avec target == ALL, chaque phrase reçoit une catégorie tirée dans P(C).
    Les marginales sont conservées en espérance.
    """
    old = df[SA_COL].to_numpy(dtype=object)
    raw = pd.to_numeric(df[RAW_COL], errors="coerce").to_numpy(dtype=float)
    n = len(old)

    if target == ALL:
        probs = np.array([m.p[c] for c in m.label_cats], dtype=float)
        new = rng.choice(np.array(m.label_cats, dtype=object), size=n, p=probs / probs.sum())
    else:
        new = old.copy()
        draw = rng.random(n) < m.p[target]
        new[draw] = target
        lose = (~draw) & (old == target)
        others = [c for c in m.label_cats if c != target]
        q = np.array([m.p[c] for c in others], dtype=float)
        if q.sum() == 0:
            raise ValueError("aucune catégorie de remplacement disponible")
        new[lose] = rng.choice(np.array(others, dtype=object), size=int(lose.sum()), p=q / q.sum())

    out = df.copy()
    out[SA_COL] = new
    out[RAW_COL] = _reconcile_subjectivity(old, new, raw, m, rng)
    return out


def build_eval_tensors(df: pd.DataFrame, subj_mean: float, subj_std: float,
                       feature_version: int) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Même construction que gru_train : HWT puis MGT, via build_features."""
    hwt = df[df["label"] == 0]
    mgt = df[df["label"] == 1]
    X, L, Y, _ = build_features(hwt, mgt, subj_mean, subj_std, feature_version=feature_version)
    return X, L, Y.numpy()


@torch.no_grad()
def predict_proba(model: torch.nn.Module, X: torch.Tensor, L: torch.Tensor,
                  device: str, batch_size: int = 512) -> np.ndarray:
    model.eval()
    ps = []
    for xb, lb in DataLoader(TensorDataset(X, L), batch_size=batch_size, shuffle=False):
        ps.append(torch.sigmoid(model(xb.to(device), lb)).cpu().numpy())
    return np.concatenate(ps).ravel()



def score(y: np.ndarray, p: np.ndarray, threshold: float) -> tuple[float, float]:
    """Conservée pour compatibilité : AUC et accuracy au seuil donné."""
    return float(roc_auc_score(y, p)), float(accuracy_score(y, (p >= threshold).astype(int)))


def category_pfi(model: torch.nn.Module, df: pd.DataFrame, m: CategoryMarginals,
                 subj_mean: float, subj_std: float, feature_version: int, device: str,
                 targets: Optional[list[str]] = None, n_repeats: int = 10, seed: int = 0,
                 threshold: float = 0.5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Importance par permutation de chaque catégorie, et du bloc catégorie entier.
    delta > 0 : la perturbation dégrade la métrique, donc le modèle s'appuie sur la catégorie."""
    targets = list(m.label_cats) + [ALL] if targets is None else list(targets)
    rng = np.random.default_rng(seed)

    X, L, y = build_eval_tensors(df, subj_mean, subj_std, feature_version)
    base_auc, base_acc = score(y, predict_proba(model, X, L, device), threshold)
    print(f"référence : AUC {base_auc:.4f}  acc {base_acc:.4f}")

    rows = []
    for t in targets:
        for r in range(n_repeats):
            dfp = resample_category(df, t, m, rng)
            Xp, Lp, yp = build_eval_tensors(dfp, subj_mean, subj_std, feature_version)
            if not np.array_equal(y, yp):
                raise RuntimeError("désalignement des étiquettes après perturbation")
            auc, acc = score(yp, predict_proba(model, Xp, Lp, device), threshold)
            rows.append({"cible": t, "repetition": r, "auc": auc, "acc": acc,
                         "delta_auc": base_auc - auc, "delta_acc": base_acc - acc})
        last = pd.DataFrame(rows[-n_repeats:])
        print(f"{t:>12s}  dAUC {last['delta_auc'].mean():+.4f}  dacc {last['delta_acc'].mean():+.4f}")

    runs = pd.DataFrame(rows)
    summary = runs.groupby("cible").agg(
        delta_auc_mean=("delta_auc", "mean"),
        delta_auc_std=("delta_auc", "std"),
        delta_acc_mean=("delta_acc", "mean"),
        delta_acc_std=("delta_acc", "std"),
    )
    summary["prevalence"] = [m.p.get(t, np.nan) for t in summary.index]
    summary = summary.sort_values("delta_auc_mean", ascending=False)
    return runs, summary


# file: src/detector/subjectivity_aware_generation_estimator/xai/category_permutation.py
# — à ajouter à la fin du fichier


def resample_subjectivity(df: pd.DataFrame, m: CategoryMarginals, rng: np.random.Generator,
                          within_category: bool = True) -> pd.DataFrame:
    """Casse le lien entre degré de subjectivité et étiquette, catégories inchangées.

    within_category=True  : chaque score est retiré dans la distribution de sa propre catégorie
                            (conditionnel : ce que la subjectivité apporte au-delà de la catégorie).
    within_category=False : chaque score est retiré parmi toutes les phrases scorées
                            (marginal : dépendance totale du modèle à la subjectivité).
    """
    cats = df[SA_COL].to_numpy(dtype=object)
    raw = pd.to_numeric(df[RAW_COL], errors="coerce").to_numpy(dtype=float)
    new_raw = raw.copy()
    scored_mask = ~np.isnan(raw)

    if within_category:
        for c, pool in m.pools.items():
            mask = scored_mask & (cats == c)
            new_raw[mask] = rng.choice(pool, size=int(mask.sum()), replace=True)
    else:
        pool = np.concatenate(list(m.pools.values()))
        new_raw[scored_mask] = rng.choice(pool, size=int(scored_mask.sum()), replace=True)

    out = df.copy()
    out[RAW_COL] = new_raw
    return out


def changed_fraction(a: pd.DataFrame, b: pd.DataFrame) -> float:
    """Fraction de phrases dont la catégorie ou le score de subjectivité a été modifié."""
    ca = a[SA_COL].to_numpy(dtype=object)
    cb = b[SA_COL].to_numpy(dtype=object)
    ra = pd.to_numeric(a[RAW_COL], errors="coerce").to_numpy(dtype=float)
    rb = pd.to_numeric(b[RAW_COL], errors="coerce").to_numpy(dtype=float)
    same_raw = (ra == rb) | (np.isnan(ra) & np.isnan(rb))
    return float(np.mean((ca != cb) | ~same_raw))


def run_pfi(model: torch.nn.Module, df: pd.DataFrame,
            perturbations: dict[str, Callable[[pd.DataFrame, np.random.Generator], pd.DataFrame]],
            subj_mean: float, subj_std: float, feature_version: int, device: str,
            n_repeats: int = 10, seed: int = 0,
            threshold: float = 0.5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """PFI générique : chaque perturbation est une fonction (df, rng) -> df perturbé."""
    rng = np.random.default_rng(seed)
    X, L, y = build_eval_tensors(df, subj_mean, subj_std, feature_version)
    base_auc, base_acc = score(y, predict_proba(model, X, L, device), threshold)
    print(f"référence : AUC {base_auc:.4f}  acc {base_acc:.4f}")

    rows = []
    for name, fn in perturbations.items():
        for r in range(n_repeats):
            dfp = fn(df, rng)
            Xp, Lp, yp = build_eval_tensors(dfp, subj_mean, subj_std, feature_version)
            if not np.array_equal(y, yp):
                raise RuntimeError(f"désalignement des étiquettes après perturbation {name!r}")
            auc, acc = score(yp, predict_proba(model, Xp, Lp, device), threshold)
            rows.append({"cible": name, "repetition": r, "auc": auc, "acc": acc,
                         "delta_auc": base_auc - auc, "delta_acc": base_acc - acc,
                         "frac_modifiee": changed_fraction(df, dfp)})
        last = pd.DataFrame(rows[-n_repeats:])
        print(f"{name:>22s}  dAUC {last['delta_auc'].mean():+.4f}  "
              f"dacc {last['delta_acc'].mean():+.4f}  modifiées {last['frac_modifiee'].mean():.3f}")

    runs = pd.DataFrame(rows)
    summary = runs.groupby("cible").agg(
        delta_auc_mean=("delta_auc", "mean"),
        delta_auc_std=("delta_auc", "std"),
        delta_acc_mean=("delta_acc", "mean"),
        delta_acc_std=("delta_acc", "std"),
        frac_modifiee=("frac_modifiee", "mean"),
    )
    summary["dauc_par_phrase_modifiee"] = summary["delta_auc_mean"] / summary["frac_modifiee"]
    return runs, summary.sort_values("delta_auc_mean", ascending=False)

# file: src/detector/subjectivity_aware_generation_estimator/xai/category_permutation.py
# — à ajouter à la fin du fichier

from src.detector.subjectivity_aware_generation_estimator.xai.metrics import (
    FPR_TARGETS, detection_metrics,
)


def run_pfi_detection(model: torch.nn.Module, df: pd.DataFrame,
                      perturbations: dict[str, Callable[[pd.DataFrame, np.random.Generator], pd.DataFrame]],
                      subj_mean: float, subj_std: float, feature_version: int, device: str,
                      n_repeats: int = 10, seed: int = 0, threshold: float = 0.5,
                      fpr_targets: tuple[float, ...] = FPR_TARGETS,
                      measure_dose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Comme run_pfi, mais avec TPR et accuracy à FPR contraint.

    measure_dose=False pour les perturbations d'ordre, où la comparaison ligne à ligne
    de changed_fraction n'a pas de sens (les lignes sont réordonnées, pas modifiées).
    """
    rng = np.random.default_rng(seed)
    X, L, y = build_eval_tensors(df, subj_mean, subj_std, feature_version)
    base = detection_metrics(y, predict_proba(model, X, L, device), threshold, fpr_targets)
    print("référence : " + "  ".join(f"{k} {v:.4f}" for k, v in base.items()))

    metric_keys = list(base)
    rows = []
    for name, fn in perturbations.items():
        for r in range(n_repeats):
            dfp = fn(df, rng)
            Xp, Lp, yp = build_eval_tensors(dfp, subj_mean, subj_std, feature_version)
            if not np.array_equal(y, yp):
                raise RuntimeError(f"désalignement des étiquettes après perturbation {name!r}")
            m = detection_metrics(yp, predict_proba(model, Xp, Lp, device), threshold, fpr_targets)
            row = {"cible": name, "repetition": r}
            row.update(m)
            row.update({f"delta_{k}": base[k] - m[k] for k in metric_keys})
            row["frac_modifiee"] = changed_fraction(df, dfp) if measure_dose else np.nan
            rows.append(row)
        last = pd.DataFrame(rows[-n_repeats:])
        print(f"{name:>22s}  dAUC {last['delta_auc'].mean():+.4f}  "
              f"dTPR@1% {last['delta_tpr@fpr1'].mean():+.4f}")

    runs = pd.DataFrame(rows)
    agg = {f"delta_{k}": (f"delta_{k}", "mean") for k in metric_keys}
    agg["delta_auc_std"] = ("delta_auc", "std")
    agg["frac_modifiee"] = ("frac_modifiee", "mean")
    summary = runs.groupby("cible").agg(**agg)
    summary["dauc_par_phrase_modifiee"] = summary["delta_auc"] / summary["frac_modifiee"]
    return runs, summary.sort_values("delta_auc", ascending=False)