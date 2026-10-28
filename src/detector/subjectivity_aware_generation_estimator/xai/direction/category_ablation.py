from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from sklearn.metrics import log_loss, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from src.config import RESULT_DIR
from src.detector.subjectivity_aware_generation_estimator.features.tensor_builder import build_dataset
from src.detector.subjectivity_aware_generation_estimator.model.SimpleGRU import SimpleGRU
from src.detector.subjectivity_aware_generation_estimator.training.gru_train import (
    predict_proba, prepare_data, set_seed, train_model,
)

LOG2 = math.log(2.0)

def run_category_loo(cfg, seeds=(0, 1, 2, 3, 4), merge_into: str = "Assertion",
                     out_dir: Path | None = None) -> pd.DataFrame:
    d = prepare_data(cfg)
    spec, ctx_base = d["spec"], d["ctx"]
    # la classe d'accueil ne peut pas être ablatée : elle absorbe les autres.
    # À 80 % de fréquence, fusionner Assertion supprimerait de toute façon
    # l'essentiel de la taxonomie, ce qui ne serait pas une contribution de catégorie.
    cats = [c for c in spec.label_cats if c != merge_into]
    bs = cfg.training.model["batch_size"]
    params = OmegaConf.to_container(cfg.training.model.params_model, resolve=True)
    params.pop("n_features", None)
    tparams = OmegaConf.to_container(cfg.training.model.params_training, resolve=True)

    out_dir = out_dir or RESULT_DIR / "category_ablation" / spec.hash()
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / "runs.jsonl"
    done = {(r["merged"], r["seed"]) for r in
            (json.loads(l) for l in jsonl.read_text().splitlines())} if jsonl.exists() else set()

    # les tenseurs dépendent de `merge` : reconstruits par condition, pas ablatés
    targets = [None] + [c for c in cats if c != "Assertion"]
    for merged in targets:
        ctx = {**ctx_base, "merge": () if merged is None else (merged,)}
        Xtr, Ltr, Ytr, _ = build_dataset(d["df_train"], spec, ctx)
        Xva, Lva, Yva, _ = build_dataset(d["df_val"], spec, ctx)
        tag = merged or "-"

        for seed in seeds:
            if (tag, seed) in done:
                continue
            set_seed(seed)
            train_dl = DataLoader(TensorDataset(Xtr, Ltr, Ytr), batch_size=bs, shuffle=True)
            val_dl = DataLoader(TensorDataset(Xva, Lva, Yva), batch_size=bs, shuffle=False)
            model = SimpleGRU(n_features=spec.input_dim, **params)
            train_model(model, train_dl, **tparams, device=cfg.hardware.device)

            y, p = predict_proba(model, val_dl, cfg.hardware.device)
            ce = log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1])
            rec = {"merged": tag, "seed": seed, "auc": float(roc_auc_score(y, p)),
                   "nats": float(LOG2 - ce)}
            with jsonl.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"seed {seed}  fusionné={tag:12s} auc {rec['auc']:.4f} nats {rec['nats']:+.4f}")

    df = pd.DataFrame([json.loads(l) for l in jsonl.read_text().splitlines()])
    agg = df.groupby("merged")[["auc", "nats"]].agg(["mean", "std"]).round(4)
    ref = agg.loc["-", ("nats", "mean")]
    agg[("nats", "delta")] = (ref - agg[("nats", "mean")]).round(4)
    return agg.sort_values(("nats", "delta"), ascending=False)


from itertools import combinations
from typing import Callable

import numpy as np
import pandas as pd

SA_COL = "rule_based_SA-label"
RAW_COL = "raw_prediction_1_5"
OTHER = "Assertion"   # catégorie absorbante par défaut


def merge_into(df: pd.DataFrame, targets: list[str], into: str) -> pd.DataFrame:
    """Rend indistinguables les catégories de `targets` et `into`.
    Le score de subjectivité est conservé : c'est une propriété du texte."""
    out = df.copy()
    out.loc[out[SA_COL].isin(targets), SA_COL] = into
    return out


def isolate_category(df: pd.DataFrame, target: str, other: str = OTHER) -> pd.DataFrame:
    """Binarise : `target` contre tout le reste. Mesure ce que cette seule distinction
    permet, sans aucune autre information catégorielle. Complément de la fusion."""
    out = df.copy()
    out.loc[out[SA_COL] != target, SA_COL] = other
    if target == other:
        raise ValueError(f"isolation impossible : {target!r} est la catégorie absorbante")
    return out


def build_category_conditions(label_cats: list[str], other: str = OTHER,
                              pairs: bool = True) -> dict[str, Callable[[pd.DataFrame], pd.DataFrame]]:
    """Trois familles complémentaires.

    fusion_X      : X devient indistinguable de `other` -> apport PROPRE de X (conditionnel).
    isolement_X   : seule la distinction X / reste subsiste -> apport TOTAL de X (marginal).
    fusion_X_et_Y : les deux retirées ensemble -> détecte la redondance entre X et Y.

    `Assertion` n'a pas de fusion (elle est la cible) mais a un isolement : c'est ainsi
    qu'on mesure enfin son apport, jamais accessible par fusion.
    """
    conditions: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
        "reference": lambda d: d,
        "aucune_categorie": lambda d: merge_into(d, list(label_cats), other),
    }

    minority = [c for c in label_cats if c != other]
    for c in minority:
        conditions[f"fusion_{c}"] = (lambda d, c=c: merge_into(d, [c], other))
    for c in minority:
        conditions[f"isolement_{c}"] = (lambda d, c=c: isolate_category(d, c, other))

    if pairs:
        for a, b in combinations(minority, 2):
            conditions[f"fusion_{a}_et_{b}"] = (lambda d, a=a, b=b: merge_into(d, [a, b], other))

    return conditions