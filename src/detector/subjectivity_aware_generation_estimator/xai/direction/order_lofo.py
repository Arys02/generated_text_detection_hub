# file: src/detector/subjectivity_aware_generation_estimator/xai/order_lofo.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from src.data.dataset_aprdb import AprdbDataset
from src.detector.subjectivity_aware_generation_estimator.model.SimpleGRU import SimpleGRU
from src.detector.subjectivity_aware_generation_estimator.training.gru_features import build_features
from src.detector.subjectivity_aware_generation_estimator.training.gru_train import train_model
from src.detector.subjectivity_aware_generation_estimator.training.linguistic_segmentation import (
    segment_database_,
)
from src.detector.subjectivity_aware_generation_estimator.xai.lofo_run import _proba, set_seed
from src.detector.subjectivity_aware_generation_estimator.xai.metrics import detection_metrics
from src.detector.subjectivity_aware_generation_estimator.xai.order_perturbation import (
    reverse_within_review, shuffle_within_review,
)

# (mélange à l'entraînement, mélange au test)
ORDER_CONDITIONS = {
    "ordre_intact":          (False, False),   # référence
    "melange_train_et_test": (True, True),     # test B : l'ordre porte-t-il de l'information ?
    "melange_train_seul":    (True, False),    # isole le biais de récence
    "melange_test_seul":     (False, True),    # test A, réentraîné pour comparabilité
}


def run_order_lofo(cfg: DictConfig, seeds=(0, 1, 2), save_dir: Path | None = None) -> pd.DataFrame:
    """Grille ordre : mélange intra-review appliqué au train, au test, aux deux, ou à aucun.

    Le mélange conserve la composition de chaque relecture — mêmes phrases, mêmes
    catégories, mêmes scores — et ne détruit que l'information d'ordre.
    Un tirage distinct par seed : la variabilité du mélange est incluse dans l'écart-type.
    """
    feature_version = OmegaConf.to_container(cfg.training.model.feature_version, resolve=True)
    seg = feature_version["segmentation"]
    device = cfg.hardware.device
    batch_size = cfg.training.model["batch_size"]
    params_model = OmegaConf.to_container(cfg.training.model.params_model, resolve=True)
    params_training = OmegaConf.to_container(cfg.training.model.params_training, resolve=True)

    df_train = segment_database_(AprdbDataset(config=cfg.dataset_train), **feature_version)
    df_val = segment_database_(AprdbDataset(config=cfg.dataset_val), **feature_version)

    raw = pd.to_numeric(df_train["raw_prediction_1_5"], errors="coerce")
    subj_mean, subj_std = float(raw.mean()), float(raw.std())

    rows = []
    for name, (shuf_tr, shuf_te) in ORDER_CONDITIONS.items():
        for seed in seeds:
            set_seed(seed)
            rng = np.random.default_rng(1000 + seed)

            dtr = shuffle_within_review(df_train, rng) if shuf_tr else df_train
            dte = shuffle_within_review(df_val, rng) if shuf_te else df_val

            Xtr, Ltr, Ytr, _ = build_features(dtr[dtr["label"] == 0], dtr[dtr["label"] == 1],
                                              subj_mean, subj_std, feature_version=seg)
            Xte, Lte, Yte, _ = build_features(dte[dte["label"] == 0], dte[dte["label"] == 1],
                                              subj_mean, subj_std, feature_version=seg)

            train_dl = DataLoader(TensorDataset(Xtr, Ltr, Ytr), batch_size=batch_size, shuffle=True)
            val_dl = DataLoader(TensorDataset(Xte, Lte, Yte), batch_size=batch_size, shuffle=False)

            model = SimpleGRU(**params_model)
            train_model(model, train_dl, **params_training, device=device)
            y, p = _proba(model, val_dl, device)

            row = {"condition": name, "seed": seed,
                   "melange_train": shuf_tr, "melange_test": shuf_te}
            row.update(detection_metrics(y, p, threshold=0.5))
            rows.append(row)
            print(f"[{name} | seed {seed}]  AUC {row['auc']:.4f}  acc {row['acc']:.4f}  "
                  f"TPR@1%FPR {row['tpr@fpr1']:.4f}")

    runs = pd.DataFrame(rows)
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        runs.to_csv(save_dir / "order_runs.csv", index=False)
    return runs