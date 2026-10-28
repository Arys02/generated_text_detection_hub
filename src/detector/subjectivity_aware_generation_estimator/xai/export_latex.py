# file: src/detector/subjectivity_aware_generation_estimator/xai/export_latex.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.detector.subjectivity_aware_generation_estimator.xai.export_common import (
    METRIC_LABELS, deltas, label_condition, latest_store,
)

PREAMBLE_NOTE = r"% requires: \usepackage{booktabs,amsmath}"
ALL_METRICS = ("auc", "tpr@fpr1", "tpr@fpr0_1")

# Note partagée : l'effectif du test ne permet pas de conclure au seuil le plus strict.
LOW_FPR_CAVEAT = (r"At 0.1\% FPR the test set allows roughly 4.5 false positives, so "
                  r"differences at that operating point are within sampling noise and "
                  r"are reported for completeness only.")
SIGNIFICANCE_NOTE = (r"$^{*}$ marks a difference larger than twice its standard error. "
                     r"Parentheses hold the standard deviation across replicates for "
                     r"values, and the standard error of the difference for $\Delta$.")


def _esc(s: str) -> str:
    return str(s).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def _fmt(value: float, digits: int, signed: bool = False) -> str:
    if value is None or not np.isfinite(value):
        return "--"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def _table(header: str, body: str, colspec: str, caption: str, label: str,
           starred_env: bool = False) -> str:
    env = "table*" if starred_env else "table"
    return "\n".join([
        PREAMBLE_NOTE, rf"\begin{{{env}}}[t]", r"  \centering",
        rf"  \caption{{{caption}}}", rf"  \label{{tab:{label}}}",
        rf"  \begin{{tabular}}{{{colspec}}}", r"    \toprule", header,
        r"    \midrule", body, r"    \bottomrule", r"  \end{tabular}",
        rf"\end{{{env}}}",
    ])


def table_importance(d: pd.DataFrame, experiment: str, reference: str, caption: str,
                     label: str, metrics: tuple[str, ...] = ALL_METRICS,
                     note: str | None = None, digits: int = 4,
                     order_by: str | None = None, starred_env: bool = False) -> str:
    """Une ligne par condition ; par métrique, la valeur et son delta contre la référence.

    L'ordre des lignes est fixé par une seule métrique et conservé pour toutes les
    autres : c'est ce qui rend visibles les réordonnancements entre points de
    fonctionnement, notamment entre AUROC et TPR à faible FPR.
    """
    order_by = order_by or metrics[0]
    frames = {m: deltas(d, experiment, m, reference).set_index("condition") for m in metrics}
    order = frames[order_by].sort_values("importance", ascending=False).index

    body = []
    for cond in order:
        cells = [_esc(label_condition(cond))]
        for m in metrics:
            r = frames[m].loc[cond]
            cells.append(_fmt(r["value_mean"], digits))
            if cond == reference:
                cells.append("--")
            else:
                star = "$^{*}$" if abs(r["importance"]) > 2 * r["se"] else ""
                cells.append(_fmt(r["importance"], digits, signed=True) + star)
        body.append("    " + " & ".join(cells) + r" \\")

    groups = " & ".join(rf"\multicolumn{{2}}{{c}}{{{METRIC_LABELS[m]}}}" for m in metrics)
    rules = " ".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(metrics)))
    subhead = "Condition & " + " & ".join(r"Value & $\Delta$" for _ in metrics)
    header = f"     & {groups} \\\\\n    {rules}\n    {subhead} \\\\"

    # les réserves méthodologiques migrent dans la légende : sans elles, l'astérisque
    # et le point de fonctionnement le plus strict seraient illisibles
    full_caption = " ".join(x for x in [
        caption, note,
        LOW_FPR_CAVEAT if "tpr@fpr0_1" in metrics else None,
        SIGNIFICANCE_NOTE,
    ] if x)
    return _table(header, "\n".join(body), "l" + "rr" * len(metrics),
                  full_caption, label, starred_env)


def table_direction(d: pd.DataFrame, caption: str, label: str,
                    note: str | None = None, digits: int = 4,
                    starred_env: bool = False) -> str:
    """Réponse dose-effet : une ligne par catégorie, une colonne par dose."""
    sub = d[(d["experiment"] == "direction_categories") & (d["metric"] == "variation")].copy()
    parts = sub["condition"].str.split("@", expand=True)
    sub["target"], sub["fraction"] = parts[0], parts[1].astype(float)
    piv = (sub.groupby(["target", "fraction"])["value"].mean()
           .unstack("fraction"))
    fractions = sorted(sub["fraction"].unique())

    body = []
    for target in piv.index:
        cells = [_esc(target)]
        cells += [_fmt(piv.loc[target, f], digits, signed=True) for f in fractions]
        cells.append("MGT" if piv.loc[target, fractions[-1]] > 0 else "HWT")
        body.append("    " + " & ".join(cells) + r" \\")

    header = ("    Speech act & " +
              " & ".join(rf"$\Delta$ @ {int(f * 100)}\%" for f in fractions) +
              r" & Direction \\")
    full_caption = " ".join(x for x in [caption, note] if x)
    return _table(header, "\n".join(body), "l" + "r" * len(fractions) + "c",
                  full_caption, label, starred_env)

def export_all(out_dir: Path, protocol: str = "paired",
               metrics: tuple[str, ...] = ALL_METRICS) -> dict[str, Path]:
    d = latest_store(protocol)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tables = {
        "blocks_pfi": table_importance(
            d, "pfi_blocs", "reference", metrics=metrics,
            caption=("Permutation importance of feature blocks. Each block is resampled at "
                     "its own marginal distribution on the test set and the trained model is "
                     "re-scored without retraining (10 draws). Positive $\\Delta$ means "
                     "destroying the information degrades the metric."),
            label="blocks_pfi",
            note=("Marginal interpretation: a block duplicating information available "
                  "elsewhere scores low even when predictive.")),
        "blocks_lofo": table_importance(
            d, "lofo_blocs", "reference", metrics=metrics,
            caption=("Leave-one-feature-out importance of feature blocks. The block is "
                     "removed from the input tensor and the model retrained (5 seeds, fixed "
                     "hyperparameters)."),
            label="blocks_lofo",
            note=("Conditional interpretation: the figure is what the block adds beyond the "
                  "remaining ones. \\emph{Speech acts removed} leaves a single input column, "
                  "so its value is a lower bound.")),
        "speech_acts_noise": table_importance(
            d, "lofo_bruit_categories", "reference", metrics=metrics,
            caption=("Speech-act importance by permutation with retraining. The category "
                     "indicator is resampled at its own marginal rate in both train and "
                     "test, so prevalence is preserved and only the association with the "
                     "label is destroyed; the model is then retrained (5 seeds)."),
            label="speech_acts_noise",
            note=("Feature count and model capacity are identical across conditions, unlike "
                  "an ablation. The subjectivity score stays attached to the sentence, so "
                  "part of the information may leak through that channel and the reported "
                  "drop is a lower bound.")),
        "sentence_order": table_importance(
            d, "ordre_lofo", "ordre_intact", metrics=metrics,
            caption=("Sensitivity to sentence order. Shuffling is applied within each "
                     "review, which preserves composition exactly and destroys only "
                     "ordering. Only the train+test condition measures order information; "
                     "the remaining conditions isolate the recency bias induced by the "
                     "last-hidden-state readout."),
            label="sentence_order"),
    }

    paths = {}
    for name, tex in tables.items():
        p = out_dir / f"{name}.tex"
        p.write_text(tex, encoding="utf-8")
        paths[name] = p
    return paths