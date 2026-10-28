# file: xai/category_profile.py
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

SA_COL = "rule_based_SA-label"


def category_profile(df: pd.DataFrame, label_cats: list[str]) -> pd.DataFrame:
    """Profil descriptif des catégories d'actes de langage, au niveau review. Aucun modèle.

    df : sortie de segment_database_ (une ligne par phrase).
    Convention : label 1 = MGT, donc auc_univariee > 0.5 <=> catégorie associée à MGT.
    """
    key = ["label", "review-id"]   # ne suppose pas review-id unique entre les deux classes
    counts = (df.groupby(key)[SA_COL].value_counts().unstack(fill_value=0)
                .reindex(columns=label_cats, fill_value=0))
    rates = counts.div(counts.sum(axis=1), axis=0)
    y = counts.index.get_level_values("label").to_numpy()
    hwt, mgt = counts[y == 0], counts[y == 1]

    rows = []
    for c in label_cats:
        present = counts[c].to_numpy() > 0
        auc = roc_auc_score(y, rates[c].to_numpy())
        rows.append({
            "categorie": c,
            "part_phrases_HWT": hwt[c].sum() / hwt.to_numpy().sum(),
            "part_phrases_MGT": mgt[c].sum() / mgt.to_numpy().sum(),
            "presence_HWT": (hwt[c] > 0).mean(),
            "presence_MGT": (mgt[c] > 0).mean(),
            "auc_univariee": auc,
            "sens": "MGT" if auc > 0.5 else "HWT",
            "p_mgt_si_presente": y[present].mean() if present.any() else np.nan,
            "n_presente": int(present.sum()),
        })

    out = pd.DataFrame(rows)
    out["pouvoir"] = (out["auc_univariee"] - 0.5).abs()
    print(f"taux de base MGT = {y.mean():.3f}  |  reviews = {len(y)}")
    return out.sort_values("pouvoir", ascending=False).reset_index(drop=True)