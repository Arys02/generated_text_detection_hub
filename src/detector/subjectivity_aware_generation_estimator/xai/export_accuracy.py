# file: src/detector/subjectivity_aware_generation_estimator/xai/export_accuracy.py
from __future__ import annotations

import pandas as pd

from src.detector.subjectivity_aware_generation_estimator.xai.export_common import (
    deltas, label_condition, latest_store,
)

FULL = "Full model"

# (ligne LaTeX, expérience, libellé de condition) dans l'ordre du tableau tab:speech_acts_noise0
ROWS: list[tuple[str, str, str]] = [
    (r"Permutation of {\bf speech acts}", "lofo_bruit_categories", "All speech acts"),
    (r"Permutation of {\bf \texttt{assertion}}", "lofo_bruit_categories", "Assertion"),
    (r"Permutation of {\bf \texttt{neither}}", "lofo_bruit_categories", "Neither"),
    (r"Permutation of {\bf \texttt{question}}", "lofo_bruit_categories", "Question"),
    (r"Permutation of {\bf \texttt{suggestion}}", "lofo_bruit_categories", "Suggestion"),
    (r"Permutation of {\bf \texttt{expressive}}", "lofo_bruit_categories", "Expressive"),
    (r"{\bf Shuffled} (train + test)", "ordre_lofo", "Shuffled (train + test)"),
    (r"{\bf Full} model", "lofo_bruit_categories", FULL),
]


def _reference(d: pd.DataFrame, experiment: str) -> str:
    conds = d.loc[d["experiment"] == experiment, "condition"].unique()
    refs = [c for c in conds if label_condition(c) == FULL]
    if len(refs) != 1:
        raise ValueError(f"{experiment}: référence absente ou ambiguë {refs}")
    return refs[0]


def _row(d: pd.DataFrame, experiment: str, label: str, metric: str) -> pd.Series:
    tab = deltas(d, experiment, metric, _reference(d, experiment))
    hit = tab[tab["label"] == label]
    if len(hit) != 1:
        raise ValueError(f"{experiment}/{metric}: {label!r} introuvable parmi {sorted(tab['label'])}")
    return hit.iloc[0]


def _cells(r: pd.Series, is_ref: bool) -> str:
    delta = "--" if is_ref else f"{r['importance']:+.4f}"
    return f"& {r['value_mean']:.4f} & {delta}"


def main() -> None:
    d = latest_store("paired")

    print("Métriques disponibles par expérience :")
    for exp, g in d.groupby("experiment"):
        print(f"  {exp:<30} {sorted(g['metric'].unique())}")

    print("\nColonne Accuracy (Value & Delta) :")
    for tex, exp, label in ROWS:
        try:
            r = _row(d, exp, label, "acc")
            print(f"  {tex:<45} {_cells(r, label == FULL)}"
                  f"   % SE={r['se']:.4f}, n={int(r['n'])}")
        except ValueError as e:
            print(f"  {tex:<45} ERREUR : {e}")

    print("\nContrôle : les deux expériences partagent-elles la même référence ?")
    for exp in ("lofo_bruit_categories", "ordre_lofo"):
        for m in ("auc", "acc"):
            try:
                r = _row(d, exp, FULL, m)
                print(f"  {exp:<24} {m:<4} {r['value_mean']:.4f} (n={int(r['n'])})")
            except ValueError as e:
                print(f"  {exp:<24} {m:<4} ERREUR : {e}")

    print("\nCandidats pour la ligne subjectivité :")
    mask = d["condition"].str.contains("subjectiv", case=False)
    for exp in sorted(d.loc[mask, "experiment"].unique()):
        for cond in sorted(d.loc[mask & (d["experiment"] == exp), "condition"].unique()):
            lab = label_condition(cond)
            try:
                auc = _row(d, exp, lab, "auc")
                acc = _row(d, exp, lab, "acc")
                ref = _row(d, exp, FULL, "acc")
                print(f"  {exp:<24} {cond:<28} AUROC={auc['value_mean']:.4f} "
                      f"(Δ {auc['importance']:+.4f}) | Acc {_cells(acc, False)} "
                      f"| ref Acc={ref['value_mean']:.4f}")
            except ValueError as e:
                print(f"  {exp:<24} {cond:<28} ERREUR : {e}")


if __name__ == "__main__":
    main()