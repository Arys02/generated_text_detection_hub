from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import log_loss, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from src.config import RESULT_DIR
from src.detector.subjectivity_aware_generation_estimator.model.SimpleGRU import SimpleGRU
from src.detector.subjectivity_aware_generation_estimator.model.PooledGRU import PooledGRU
from src.detector.subjectivity_aware_generation_estimator.training.gru_train import (
    predict_proba, prepare_data, set_seed, train_model,
)

ARCHS = {
    "last": lambda n, p: SimpleGRU(n_features=n, **p),
    "mean": lambda n, p: PooledGRU(n_features=n, pooling="mean", **p),
    "max":  lambda n, p: PooledGRU(n_features=n, pooling="max", **p),
}
from .order import OrderedDataset, permute_fixed

LOG2 = float(np.log(2.0))

TRAIN_MODES = [("identity", None), ("shuffle", None),
               ("block", 2), ("block", 5), ("block", 10), ("reverse", None)]


def _score(model, X, L, Y, cfg) -> tuple[float, float]:
    dl = DataLoader(TensorDataset(X, L, Y),
                    batch_size=cfg.training.model["batch_size"], shuffle=False)
    y, p = predict_proba(model, dl, cfg.hardware.device)
    ce = log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1])
    return float(roc_auc_score(y, p)), float(LOG2 - ce)


def run_order_study(cfg: DictConfig, seeds=(0, 1, 2, 3, 4),
                    eval_perms=(101, 102, 103), archs=("last", "mean"), out_dir: Path | None = None) -> pd.DataFrame:
    d = prepare_data(cfg)
    spec = d["spec"]
    bs = cfg.training.model["batch_size"]
    params = OmegaConf.to_container(cfg.training.model.params_model, resolve=True)
    params.pop("n_features", None)
    tparams = OmegaConf.to_container(cfg.training.model.params_training, resolve=True)

    out_dir = out_dir or RESULT_DIR / "order_study" / spec.hash()
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / "runs.jsonl"

    done = set()
    if jsonl.exists():
        for line in jsonl.read_text().splitlines():
            r = json.loads(line)
            # arch absent des lignes déjà écrites -> 'last' par défaut
            done.add((r.get("arch", "last"), r["train_mode"], r["window"], r["seed"], r["test_mode"]))

    for seed in seeds:
        for arch in archs:
            for mode, window in TRAIN_MODES:
                if (arch, mode, window, seed, "identity") in done:
                    continue
                set_seed(seed)
                train_dl = DataLoader(
                    OrderedDataset(d["Xtr"], d["Ltr"], d["Ytr"], mode=mode, window=window),
                    batch_size=bs, shuffle=True)
                model = ARCHS[arch](spec.input_dim, params)
                train_model(model, train_dl, **tparams, device=cfg.hardware.device)

                batch = []
                auc, nats = _score(model, d["Xva"], d["Lva"], d["Yva"], cfg)
                batch.append({"arch": arch, "train_mode": mode, "window": window,
                              "seed": seed, "test_mode": "identity", "auc": auc, "nats": nats})

                test_mode = "shuffle" if mode == "identity" else mode
                for ps in eval_perms:
                    Xs = permute_fixed(d["Xva"], d["Lva"], seed=ps, mode=test_mode, window=window)
                    a, n = _score(model, Xs, d["Lva"], d["Yva"], cfg)
                    batch.append({"arch": arch, "train_mode": mode, "window": window,
                                  "seed": seed, "test_mode": f"{test_mode}_{ps}",
                                  "auc": a, "nats": n})

                with jsonl.open("a") as f:
                    for r in batch:
                        f.write(json.dumps(r) + "\n")
                print(f"seed {seed}  {arch:5s} train={mode}{window or '':>3}  "
                      f"nats_ord={batch[0]['nats']:+.4f}  "
                      f"nats_perm={np.mean([r['nats'] for r in batch[1:]]):+.4f}")
   
    df = pd.DataFrame([json.loads(l) for l in jsonl.read_text().splitlines()])
    df.to_csv(out_dir / "runs.csv", index=False)
    return df

def summarise(df, metric="nats"):
    d = df.copy()
    d["arch"] = d.get("arch", "last")
    d["eval"] = np.where(d["test_mode"] == "identity", "ordered", "permuted")
    d["mode"] = d["train_mode"] + d["window"].fillna("").astype(str).str.replace(r"\.0$", "", regex=True)
    per_seed = d.groupby(["arch", "mode", "eval", "seed"])[metric].mean().unstack("eval")
    return per_seed.groupby(["arch", "mode"]).agg(["mean", "std"]).round(4)