# file: src/detector/subjectivity_aware_generation_estimator/xai/ablation.py
from __future__ import annotations

import json
import math
from itertools import chain, combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import log_loss, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from src.config import RESULT_DIR
from src.detector.subjectivity_aware_generation_estimator.features.feature_spec import FeatureSpec
from src.detector.subjectivity_aware_generation_estimator.features.tensor_builder import ablate
from src.detector.subjectivity_aware_generation_estimator.model.SimpleGRU import SimpleGRU
from src.detector.subjectivity_aware_generation_estimator.training.gru_train import (
    predict_proba, prepare_data, set_seed, train_model,
)

LOG2 = math.log(2.0)


def powerset(items):
    items = list(items)
    return [frozenset(s) for s in
            chain.from_iterable(combinations(items, r) for r in range(len(items) + 1))]


def coalition_label(keep: frozenset, players: tuple[str, ...]) -> str:
    """Étiquette stable : ordre de la spec, pas ordre d'itération du frozenset."""
    return "+".join(p for p in players if p in keep) or "-"


def run_one(keep: frozenset, seed: int, d: dict, cfg: DictConfig) -> dict:
    spec: FeatureSpec = d["spec"]
    set_seed(seed)

    Xtr = ablate(d["Xtr"], spec, keep)
    Xva = ablate(d["Xva"], spec, keep)

    bs = cfg.training.model["batch_size"]
    train_dl = DataLoader(TensorDataset(Xtr, d["Ltr"], d["Ytr"]), batch_size=bs, shuffle=True)
    val_dl = DataLoader(TensorDataset(Xva, d["Lva"], d["Yva"]), batch_size=bs, shuffle=False)

    params = OmegaConf.to_container(cfg.training.model.params_model, resolve=True)
    params.pop("n_features", None)
    # input_dim constant entre coalitions : le compte de paramètres ne varie pas
    model = SimpleGRU(n_features=spec.input_dim, **params)
    train_model(model, train_dl,
                **OmegaConf.to_container(cfg.training.model.params_training, resolve=True),
                device=cfg.hardware.device)

    y, p = predict_proba(model, val_dl, cfg.hardware.device)
    ce = log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1])
    return {"coalition": coalition_label(keep, spec.players),
            "keep": sorted(keep), "seed": seed,
            "auc": float(roc_auc_score(y, p)), "nats": float(LOG2 - ce),
            "y": y, "p": p}


def run_study(cfg: DictConfig, seeds=(0, 1, 2, 3, 4), out_dir: Path | None = None) -> pd.DataFrame:
    """Seeds appariés : chaque seed parcourt TOUTES les coalitions, ce qui élimine
    la chance d'initialisation comme source de variance entre coalitions."""
    d = prepare_data(cfg)
    spec: FeatureSpec = d["spec"]
    coalitions = powerset(spec.players)

    out_dir = out_dir or RESULT_DIR / "ablation" / spec.hash()
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / "runs.jsonl"

    done = set()
    if jsonl.exists():   # reprise après interruption
        for line in jsonl.read_text().splitlines():
            r = json.loads(line)
            done.add((r["coalition"], r["seed"]))
        print(f"reprise : {len(done)} runs déjà présents")

    preds = {}
    for seed in seeds:
        for keep in coalitions:
            key = (coalition_label(keep, spec.players), seed)
            if key in done:
                continue
            r = run_one(keep, seed, d, cfg)
            preds[key] = (r.pop("y"), r.pop("p"))
            with jsonl.open("a") as f:
                f.write(json.dumps(r) + "\n")
            print(f"seed {seed}  {r['coalition']:>18s}  auc {r['auc']:.4f}  nats {r['nats']:+.4f}")

    if preds:
        np.savez_compressed(out_dir / "predictions.npz",
                            **{f"{c}__{s}": p for (c, s), (_, p) in preds.items()},
                            y_true=next(iter(preds.values()))[0])

    df = pd.DataFrame([json.loads(l) for l in jsonl.read_text().splitlines()])
    df.to_csv(out_dir / "runs.csv", index=False)
    return df


# --------------------------------------------------------------------------- #
# Valeurs de Shapley exactes — énumération complète, aucune approximation
# --------------------------------------------------------------------------- #

def shapley_exact(v: dict[frozenset, float], players: tuple[str, ...]) -> dict[str, float]:
    n = len(players)
    phi = {}
    for i in players:
        others = [p for p in players if p != i]
        phi[i] = sum(
            math.factorial(len(S)) * math.factorial(n - len(S) - 1) / math.factorial(n)
            * (v[frozenset(S) | {i}] - v[frozenset(S)])
            for r in range(n) for S in combinations(others, r)
        )
    return phi


def interaction_index(v: dict[frozenset, float], players: tuple[str, ...]) -> dict[tuple, float]:
    """Grabisch & Roubens (1999) : différence seconde moyennée. > 0 = synergie."""
    n = len(players)
    out = {}
    for i, j in combinations(players, 2):
        others = [p for p in players if p not in (i, j)]
        out[(i, j)] = sum(
            math.factorial(len(S)) * math.factorial(n - len(S) - 2) / math.factorial(n - 1)
            * (v[frozenset(S) | {i, j}] - v[frozenset(S) | {i}]
               - v[frozenset(S) | {j}] + v[frozenset(S)])
            for r in range(len(others) + 1) for S in combinations(others, r)
        )
    return out


def analyse(df: pd.DataFrame, players: tuple[str, ...], metric: str = "auc") -> dict:
    rows_phi, rows_int = [], []
    for seed, g in df.groupby("seed"):
        v = {frozenset(k): float(x) for k, x in zip(g["keep"], g[metric])}
        missing = [c for c in powerset(players) if c not in v]
        if missing:
            raise ValueError(f"seed {seed} : coalitions manquantes {missing}")

        phi = shapley_exact(v, players)
        gap = sum(phi.values()) - (v[frozenset(players)] - v[frozenset()])
        assert abs(gap) < 1e-9, f"axiome d'efficacité violé : {gap}"   # contrôle numérique

        rows_phi.append({"seed": seed, **phi,
                         "v_full": v[frozenset(players)], "v_empty": v[frozenset()]})
        rows_int.append({"seed": seed,
                         **{f"{a}x{b}": x for (a, b), x in interaction_index(v, players).items()}})

    phi_df = pd.DataFrame(rows_phi).set_index("seed")
    int_df = pd.DataFrame(rows_int).set_index("seed")
    summary = pd.DataFrame({"mean": phi_df.mean(), "std": phi_df.std()}).round(4)
    return {"phi": phi_df, "interactions": int_df, "summary": summary}