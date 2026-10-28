# file: src/detector/subjectivity_aware_generation_estimator/xai/reporting.py
from __future__ import annotations

import pandas as pd


def print_table(df: pd.DataFrame, title: str = "", decimals: int = 4, index: bool = True) -> None:
    """Affichage texte brut, sans troncature ni retour à la ligne, prêt à copier-coller."""
    if title:
        print(f"== {title} ==")
    with pd.option_context("display.max_rows", None, "display.max_columns", None,
                           "display.width", None, "display.max_colwidth", None):
        print(df.to_string(index=index, float_format=lambda x: f"{x:.{decimals}f}"))
    print()


def category_report(summary_pfi: pd.DataFrame, profile: pd.DataFrame) -> pd.DataFrame:
    """Vue unique, une ligne par catégorie : signal dans les données vs dépendance du modèle."""
    pfi = summary_pfi.reset_index().rename(columns={"cible": "categorie"})
    prof_cols = ["categorie", "part_phrases_HWT", "part_phrases_MGT",
                 "auc_univariee", "sens", "p_mgt_si_presente"]
    pfi_cols = ["categorie", "delta_auc_mean", "delta_auc_std",
                "frac_modifiee", "dauc_par_phrase_modifiee"]
    out = profile[prof_cols].merge(pfi[pfi_cols], on="categorie", how="inner")
    out = out.rename(columns={
        "part_phrases_HWT": "part_HWT",
        "part_phrases_MGT": "part_MGT",
        "auc_univariee": "auc_uni",
        "p_mgt_si_presente": "p_mgt_pres",
        "delta_auc_mean": "dAUC",
        "delta_auc_std": "dAUC_std",
        "frac_modifiee": "dose",
        "dauc_par_phrase_modifiee": "dAUC_dose",
    })
    return out.sort_values("dAUC_dose", ascending=False).reset_index(drop=True)