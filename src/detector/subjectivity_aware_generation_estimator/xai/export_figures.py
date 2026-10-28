# file: src/detector/subjectivity_aware_generation_estimator/xai/export_figures.py
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.detector.subjectivity_aware_generation_estimator.xai.export_common import (  # noqa: E402
    METRIC_LABELS_PLAIN, deltas, latest_store,
)

DOT, LINE, GRID, REFLINE = "#1f3b57", "#5b7c99", "#d8dee4", "#a33b3b"


def _style(ax) -> None:
    ax.grid(axis="x", color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(left=False)


def importance_plot(d: pd.DataFrame, experiment: str, metric: str, reference: str,
                    title: str, out: Path, spread: str = "quantile",
                    drop_reference: bool = True, figsize=(6.2, 3.2)) -> Path:
    """Figure d'importance au format de Molnar : point pour la moyenne, ligne pour
    l'intervalle, conditions triées par importance décroissante.

    spread='quantile' donne l'intervalle 5-95 % sur les réplicats, comme dans le livre ;
    avec 5 seeds il équivaut presque au min-max. spread='se' donne +/- 2 erreurs-types
    de la différence, préférable quand les réplicats sont peu nombreux.
    """
    df = deltas(d, experiment, metric, reference)
    if drop_reference:
        df = df[df["condition"] != reference]
    df = df.sort_values("importance")          # tracé de bas en haut

    if spread == "se":
        lo, hi = df["importance"] - 2 * df["se"], df["importance"] + 2 * df["se"]
        legend = "mean $\\pm$ 2 SE"
    else:
        lo, hi = df["q_lo"], df["q_hi"]
        legend = "5–95% range over replicates"

    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=figsize)
    ax.axvline(0.0, color=REFLINE, linewidth=1.0, linestyle="--", zorder=1)
    ax.hlines(y, lo, hi, color=LINE, linewidth=2.0, zorder=2)
    ax.scatter(df["importance"], y, s=38, color=DOT, zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(df["label"])
    ax.set_xlabel(f"Drop in {METRIC_LABELS_PLAIN.get(metric, metric)}")
    ax.set_title(title, loc="left", fontsize=11)
    ax.text(0.99, -0.22, legend, transform=ax.transAxes, ha="right",
            fontsize=8, color="#555")
    _style(ax)
    fig.tight_layout()

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out.with_suffix(".pdf")


def comparison_plot(d: pd.DataFrame, experiments: dict[str, str], metric: str,
                    reference: str, title: str, out: Path,
                    figsize=(6.6, 3.4)) -> Path:
    """Deux opérationnalisations côte à côte (fusion vs permutation)."""
    frames = {}
    for name, exp in experiments.items():
        f = deltas(d, exp, metric, reference)
        frames[name] = f[f["condition"] != reference].set_index("label")

    labels = list(frames[list(experiments)[0]].sort_values("importance").index)
    y = np.arange(len(labels))
    offsets = np.linspace(-0.16, 0.16, len(frames))
    colors = [DOT, "#b07a3c"]

    fig, ax = plt.subplots(figsize=figsize)
    ax.axvline(0.0, color=REFLINE, linewidth=1.0, linestyle="--", zorder=1)
    for (name, f), off, col in zip(frames.items(), offsets, colors):
        f = f.reindex(labels)
        ax.hlines(y + off, f["importance"] - 2 * f["se"], f["importance"] + 2 * f["se"],
                  color=col, alpha=0.5, linewidth=2.0, zorder=2)
        ax.scatter(f["importance"], y + off, s=34, color=col, label=name, zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(f"Drop in {METRIC_LABELS_PLAIN.get(metric, metric)}")
    ax.set_title(title, loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    _style(ax)
    fig.tight_layout()

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out.with_suffix(".pdf")


def direction_plot(d: pd.DataFrame, title: str, out: Path, figsize=(6.0, 3.4)) -> Path:
    """Réponse dose-effet : variation de p_MGT en fonction de la part convertie."""
    sub = d[(d["experiment"] == "direction_categories") & (d["metric"] == "variation")].copy()
    parts = sub["condition"].str.split("@", expand=True)
    sub["target"], sub["fraction"] = parts[0], parts[1].astype(float)
    agg = sub.groupby(["target", "fraction"])["value"].agg(["mean", "std"]).reset_index()

    fig, ax = plt.subplots(figsize=figsize)
    ax.axhline(0.0, color=REFLINE, linewidth=1.0, linestyle="--", zorder=1)
    cmap = plt.get_cmap("tab10")
    for i, (target, g) in enumerate(agg.groupby("target")):
        g = g.sort_values("fraction")
        x = np.concatenate([[0.0], g["fraction"].to_numpy()])
        ymean = np.concatenate([[0.0], g["mean"].to_numpy()])
        ax.errorbar(x, ymean, yerr=np.concatenate([[0.0], g["std"].to_numpy()]),
                    marker="o", markersize=4, linewidth=1.6, capsize=3,
                    color=cmap(i), label=target, zorder=2)

    ax.set_xlabel("Share of sentences converted to the target speech act")
    ax.set_ylabel(r"$\Delta$ mean predicted $p_{\mathrm{MGT}}$")
    ax.set_title(title, loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9, ncol=2)
    ax.grid(color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out.with_suffix(".pdf")

# file: src/detector/subjectivity_aware_generation_estimator/xai/export_figures.py
# — à ajouter à la fin du fichier

def multi_metric_plot(d: pd.DataFrame, experiment: str, reference: str, title: str,
                      out: Path, metrics: tuple[str, ...] = ("auc", "tpr@fpr1", "tpr@fpr0_1"),
                      spread: str = "se", drop_reference: bool = True,
                      order_by: str | None = None, panel_width: float = 2.9,
                      height: float = 3.2) -> Path:
    """Une colonne par métrique, conditions alignées sur les mêmes lignes.

    L'ordre des lignes est fixé par `order_by` (par défaut la première métrique) et
    conservé sur tous les panneaux : c'est ce qui rend les réordonnancements visibles
    d'un point de fonctionnement à l'autre.
    """
    order_by = order_by or metrics[0]
    frames = {m: deltas(d, experiment, m, reference).set_index("condition") for m in metrics}

    ref_frame = frames[order_by]
    conds = [c for c in ref_frame.sort_values("importance").index
             if not (drop_reference and c == reference)]
    labels = [ref_frame.loc[c, "label"] for c in conds]
    y = np.arange(len(conds))

    fig, axes = plt.subplots(1, len(metrics), sharey=True,
                             figsize=(panel_width * len(metrics), height))
    axes = np.atleast_1d(axes)

    for ax, m in zip(axes, metrics):
        f = frames[m].reindex(conds)
        if spread == "se":
            lo, hi = f["importance"] - 2 * f["se"], f["importance"] + 2 * f["se"]
        else:
            lo, hi = f["q_lo"], f["q_hi"]

        ax.axvline(0.0, color=REFLINE, linewidth=1.0, linestyle="--", zorder=1)
        ax.hlines(y, lo, hi, color=LINE, linewidth=2.0, zorder=2)
        # un gain (importance < 0) se distingue d'une perte : retirer l'information
        # a amélioré la métrique, ce qui doit sauter aux yeux
        colors = np.where(f["importance"].to_numpy() < 0, "#b07a3c", DOT)
        ax.scatter(f["importance"], y, s=36, c=colors, zorder=3)
        ax.set_xlabel(f"Drop in {METRIC_LABELS_PLAIN.get(m, m)}", fontsize=9)
        ax.tick_params(labelsize=9)
        _style(ax)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    legend = "mean $\\pm$ 2 SE" if spread == "se" else "5–95% range over replicates"
    fig.text(0.99, 0.005,
             f"",
             ha="right", fontsize=8, color="#555")
    fig.suptitle(title, x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out.with_suffix(".pdf")