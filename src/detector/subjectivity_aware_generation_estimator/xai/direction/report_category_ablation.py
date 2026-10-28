"""
Reconstruit la table de résultats de l'ablation par catégorie à partir des
fichiers sauvegardés. Aucun entraînement, aucun GPU.

Usage :
    python -m src.detector.subjectivity_aware_generation_estimator.xai.report_category_ablation
    python -m src.detector.subjectivity_aware_generation_estimator.xai.report_category_ablation \
        --spec-hash <hash> --seg-dir 13af569ff61302c3c2d7fbb1eea34954_eaf4e799
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.config import RESULT_DIR
from src.detector.subjectivity_aware_generation_estimator.features.pairing import pair_and_balance

REF = "-"
SA_COL = "rule_based_SA-label"
SEG_ROOT = RESULT_DIR / "implicite_segmentation"
DEFAULT_TEST_SEG = "13af569ff61302c3c2d7fbb1eea34954_eaf4e799"
EXPECTED_REF_NATS = 0.400

COMPACT = ["auc_mean", "nats_mean", "nats_std",
           "d_nats", "sd_d_nats", "ci_lo_nats", "ci_hi_nats", "t_nats", "p_holm_nats",
           "d_auc", "freq_hwt", "freq_mgt", "ratio_mgt_hwt", "freq", "d_per_freq"]


# --------------------------------------------------------------------------- #
# Chargement
# --------------------------------------------------------------------------- #

def find_run_dir(spec_hash: str | None) -> Path:
    root = RESULT_DIR / "category_ablation"
    if spec_hash:
        d = root / spec_hash
        if not (d / "runs.jsonl").exists():
            raise FileNotFoundError(d / "runs.jsonl")
        return d
    cands = sorted(p.parent for p in root.glob("*/runs.jsonl"))
    if len(cands) != 1:
        raise RuntimeError(f"{len(cands)} études trouvées, préciser --spec-hash : "
                           f"{[c.name for c in cands]}")
    return cands[0]


def load_runs(run_dir: Path) -> pd.DataFrame:
    lines = (run_dir / "runs.jsonl").read_text().splitlines()
    df = pd.DataFrame([json.loads(l) for l in lines if l.strip()])
    n_dup = int(df.duplicated(["merged", "seed"]).sum())
    if n_dup:
        print(f"[avertissement] {n_dup} doublons (merged, seed) : dernière occurrence conservée")
        df = df.drop_duplicates(["merged", "seed"], keep="last")
    return df


# --------------------------------------------------------------------------- #
# Contrôles
# --------------------------------------------------------------------------- #

def sanity_checks(runs: pd.DataFrame) -> None:
    piv = runs.pivot(index="seed", columns="merged", values="nats")
    if REF not in piv.columns:
        raise KeyError("condition de référence '-' absente du fichier de runs")

    print("seeds par condition :", piv.notna().sum().to_dict())

    ref = piv[REF].mean()
    flag = "" if abs(ref - EXPECTED_REF_NATS) < 0.03 else "  <-- écart à la référence attendue"
    print(f"référence : {ref:.4f} nats (attendu ~{EXPECTED_REF_NATS}){flag}")

    # le défaut de la première implémentation produisait des conditions identiques à la référence
    for c in piv.columns.drop(REF):
        pair = piv[[REF, c]].dropna()
        if len(pair) and np.allclose(pair[REF], pair[c], atol=1e-9):
            print(f"[ALERTE] '{c}' est identique à la référence seed par seed : "
                  f"la fusion n'a probablement pas modifié le tenseur")


# --------------------------------------------------------------------------- #
# Statistiques appariées
# --------------------------------------------------------------------------- #

def paired_deltas(runs: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Delta = référence - condition, calculé seed par seed.
    L'appariement retire la variance d'initialisation commune aux deux runs
    d'un même seed : l'écart-type d'une différence appariée est plus faible
    que la somme des écarts-types marginaux."""
    piv = runs.pivot(index="seed", columns="merged", values=metric)
    out = []
    for cond in piv.columns.drop(REF):
        pair = piv[[REF, cond]].dropna()
        d = (pair[REF] - pair[cond]).to_numpy()          # > 0 : la fusion coûte
        n = len(d)
        m = float(d.mean())
        sd = float(d.std(ddof=1)) if n > 1 else np.nan
        se = sd / np.sqrt(n) if n > 1 else np.nan
        t = m / se if n > 1 and se > 0 else np.nan
        p = float(2 * stats.t.sf(abs(t), df=n - 1)) if np.isfinite(t) else np.nan
        half = float(stats.t.ppf(0.975, df=n - 1) * se) if n > 1 else np.nan
        out.append({"merged": cond, f"d_{metric}": m, f"sd_d_{metric}": sd,
                    f"ci_lo_{metric}": m - half, f"ci_hi_{metric}": m + half,
                    f"t_{metric}": t, f"p_{metric}": p, "n_seeds": n})
    return pd.DataFrame(out).set_index("merged")


def holm(p: pd.Series) -> pd.Series:
    """Correction de Holm-Bonferroni pour comparaisons multiples."""
    s = p.dropna().sort_values()
    m = len(s)
    adj = np.maximum.accumulate([(m - i) * v for i, v in enumerate(s.to_numpy())])
    return pd.Series(np.minimum(adj, 1.0), index=s.index).reindex(p.index)


# --------------------------------------------------------------------------- #
# Fréquences — sur les données APPARIÉES, c'est-à-dire ce que le modèle a vu
# --------------------------------------------------------------------------- #

def category_frequencies(seg_dir: Path, pairing_seed: int = 0) -> pd.DataFrame:
    df = pd.read_csv(seg_dir / "sentences_degrees_final.csv",
                     usecols=["review-id", "label", "generator", SA_COL])
    df = pair_and_balance(df, seed=pairing_seed, verbose=False)

    by = pd.crosstab(df[SA_COL], df["label"], normalize="columns")
    by.columns = ["freq_hwt", "freq_mgt"]
    by["ratio_mgt_hwt"] = by["freq_mgt"] / by["freq_hwt"]
    by["freq"] = df[SA_COL].value_counts(normalize=True)
    by.index.name = "merged"
    return by


# --------------------------------------------------------------------------- #
# Assemblage
# --------------------------------------------------------------------------- #

def build_table(run_dir: Path, seg_dir: Path, pairing_seed: int = 0) -> pd.DataFrame:
    runs = load_runs(run_dir)
    sanity_checks(runs)

    perf = runs.groupby("merged")[["auc", "nats"]].agg(["mean", "std"])
    perf.columns = [f"{m}_{s}" for m, s in perf.columns]

    dn = paired_deltas(runs, "nats")
    dn["p_holm_nats"] = holm(dn["p_nats"])
    da = paired_deltas(runs, "auc")[["d_auc", "sd_d_auc", "p_auc"]]

    tab = (perf.join(dn, how="left")
               .join(da, how="left")
               .join(category_frequencies(seg_dir, pairing_seed), how="left"))
    tab["d_per_freq"] = tab["d_nats"] / tab["freq"]

    order = tab["d_nats"].fillna(-np.inf).sort_values(ascending=False).index
    return tab.loc[order]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec-hash", default=None)
    ap.add_argument("--seg-dir", default=DEFAULT_TEST_SEG,
                    help="répertoire de segmentation du split d'évaluation")
    ap.add_argument("--pairing-seed", type=int, default=0,
                    help="doit égaler dataset_val.seed utilisé à l'entraînement")
    args = ap.parse_args()

    run_dir = find_run_dir(args.spec_hash)
    print(f"étude : {run_dir}\n")
    tab = build_table(run_dir, SEG_ROOT / args.seg_dir, args.pairing_seed)
    tab = tab[['auc_mean', 'd_auc', 'freq_hwt', 'freq_mgt', 'ratio_mgt_hwt', 'freq', 'd_per_freq',]]

    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:.4f}".format)
    print("\n" + tab[[c for c in COMPACT if c in tab.columns]].to_string())

    tab.to_csv(run_dir / "report_category_ablation.csv")
    try:
        (run_dir / "report_category_ablation.md").write_text(
            tab[[c for c in COMPACT if c in tab.columns]].round(4).to_markdown())
    except ImportError:
        pass                                    # to_markdown requiert 'tabulate'
    print(f"\nécrit : {run_dir / 'report_category_ablation.csv'}")


if __name__ == "__main__":
    main()