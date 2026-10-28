# file: src/detector/subjectivity_aware_generation_estimator/xai/lofo_run.py
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from src.detector.subjectivity_aware_generation_estimator.xai.metrics import detection_metrics
from src.config import RESULT_DIR
from src.data.dataset_aprdb import AprdbDataset
from src.detector.subjectivity_aware_generation_estimator.model.SimpleGRU import SimpleGRU
from src.detector.subjectivity_aware_generation_estimator.training.gru_features import (
    LABEL_CATS_1, LABEL_CATS_2, build_features,
)
from src.detector.subjectivity_aware_generation_estimator.training.gru_train import train_model
from src.detector.subjectivity_aware_generation_estimator.training.linguistic_segmentation import (
    segment_database_,
)
from src.detector.subjectivity_aware_generation_estimator.xai.lofo_conditions import build_conditions


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def _proba(model, dl, device: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, ps = [], []
    for xb, lb, yb in dl:
        ps.append(torch.sigmoid(model(xb.to(device), lb)).cpu().numpy())
        ys.append(yb.numpy())
    return np.concatenate(ys).ravel(), np.concatenate(ps).ravel()


def run_lofo(cfg: DictConfig, seeds=(0, 1, 2), with_merges: bool = True,
             save_dir: Path | None = None) -> pd.DataFrame:
    """LOFO à hyperparamètres figés : chaque condition transforme le DataFrame de
    phrases identiquement au train et au test, puis un GRU neuf est entraîné.

    Les statistiques de normalisation subj_mean/subj_std sont recalculées par condition
    sur son propre train : sinon la condition subjectivite_seule serait normalisée
    avec les statistiques d'un jeu qu'elle ne voit pas."""
    feature_version = OmegaConf.to_container(cfg.training.model.feature_version, resolve=True)
    seg = feature_version["segmentation"]
    label_cats = LABEL_CATS_1 if seg == 1 else LABEL_CATS_2
    device = cfg.hardware.device
    batch_size = cfg.training.model["batch_size"]
    params_model = OmegaConf.to_container(cfg.training.model.params_model, resolve=True)
    params_training = OmegaConf.to_container(cfg.training.model.params_training, resolve=True)

    df_train = segment_database_(AprdbDataset(config=cfg.dataset_train), **feature_version)
    df_val = segment_database_(AprdbDataset(config=cfg.dataset_val), **feature_version)
    conditions = build_conditions(label_cats, with_merges=with_merges)

    rows = []
    for name, transform in conditions.items():
        dtr, dva = transform(df_train), transform(df_val)

        raw = pd.to_numeric(dtr["raw_prediction_1_5"], errors="coerce")
        subj_mean = float(raw.mean()) if raw.notna().any() else 0.0
        subj_std = float(raw.std()) if raw.notna().any() and raw.std() > 0 else 1.0

        Xtr, Ltr, Ytr, _ = build_features(dtr[dtr["label"] == 0], dtr[dtr["label"] == 1],
                                          subj_mean, subj_std, feature_version=seg)
        Xva, Lva, Yva, _ = build_features(dva[dva["label"] == 0], dva[dva["label"] == 1],
                                          subj_mean, subj_std, feature_version=seg)

        for seed in seeds:
            set_seed(seed)
            train_dl = DataLoader(TensorDataset(Xtr, Ltr, Ytr), batch_size=batch_size, shuffle=True)
            val_dl = DataLoader(TensorDataset(Xva, Lva, Yva), batch_size=batch_size, shuffle=False)

            model = SimpleGRU(**params_model)
            train_model(model, train_dl, **params_training, device=device)
            y, p = _proba(model, val_dl, device)

            row = {"condition": name, "seed": seed}
            row.update(detection_metrics(y, p, threshold=0.5))
            rows.append(row)
            print(f"[{name} | seed {seed}]  AUC {row['auc']:.4f}  acc {row['acc']:.4f}  "
                  f"TPR@1%FPR {row['tpr@fpr1']:.4f}")
            
    runs = pd.DataFrame(rows)
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        runs.to_csv(save_dir / "lofo_runs.csv", index=False)
    return runs


def summarize_lofo(runs: pd.DataFrame, reference: str = "reference") -> pd.DataFrame:
    """delta > 0 : retirer l'information dégrade, donc elle apporte quelque chose
    au-delà de ce qui reste. delta <= 0 : information redondante."""
    metric_cols = [c for c in runs.columns if c not in ("condition", "seed")]
    agg = runs.groupby("condition")[metric_cols].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg["n"] = runs.groupby("condition").size()

    ref = agg.loc[reference]
    for c in metric_cols:
        agg[f"delta_{c}"] = ref[f"{c}_mean"] - agg[f"{c}_mean"]
    agg["delta_auc_se"] = np.sqrt(agg["auc_std"] ** 2 / agg["n"] + ref["auc_std"] ** 2 / ref["n"])

    keep = (["auc_mean", "auc_std", "delta_auc", "delta_auc_se", "acc_mean", "delta_acc"]
            + [f"delta_{c}" for c in metric_cols if c.startswith("tpr@")]
            + [f"{c}_mean" for c in metric_cols if c.startswith("tpr@")])
    return agg[keep].sort_values("delta_auc", ascending=False)

