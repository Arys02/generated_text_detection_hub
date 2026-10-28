# file: src/detector/subjectivity_aware_generation_estimator/training/transformer_train.py
from __future__ import annotations

import logging
import math
import random
from pathlib import Path
from typing import Optional, Tuple

import hydra
import joblib
import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import RocCurveDisplay, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, TensorDataset

from src.config import RESULT_DIR
from src.data.dataset_aprdb import AprdbDataset
from src.detector.subjectivity_aware_generation_estimator.model.SubjTransformer import SubjectivityTransformer
from src.detector.subjectivity_aware_generation_estimator.training.gru_features import (
    LABEL_CATS_1, LABEL_CATS_2, build_features,
)
# évaluation importée telle quelle : garantit des métriques strictement identiques au run GRU
from src.detector.subjectivity_aware_generation_estimator.training.gru_train import evaluate
from src.detector.subjectivity_aware_generation_estimator.training.linguistic_segmentation import segment_database_
from src.evaluation.eval_utils import generate_config_hash

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

logger = logging.getLogger(__name__)
load_dotenv()


# --------------------------------------------------------------------------- #
# Reproductibilité
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Préparation des données — DOIT rester identique au script GRU
# --------------------------------------------------------------------------- #
def build_split_tensors(cfg: DictConfig):
    """Reproduit exactement le pipeline de gru_train.train_gru jusqu'aux tenseurs.

    À factoriser dans un module commun importé par les deux scripts : toute
    divergence ici invalide la comparaison d'architectures.
    """
    ds_train = AprdbDataset(config=cfg.dataset_train)
    ds_val = AprdbDataset(config=cfg.dataset_val)

    feature_version = OmegaConf.to_container(cfg.training.model.feature_version, resolve=True)
    segmentation_version = feature_version["segmentation"]

    df_train = segment_database_(ds_train, **feature_version)
    df_val = segment_database_(ds_val, **feature_version)

    hwt_tr, mgt_tr = df_train[df_train["label"] == 0], df_train[df_train["label"] == 1]
    hwt_va, mgt_va = df_val[df_val["label"] == 0], df_val[df_val["label"] == 1]

    train_all = pd.concat([hwt_tr, mgt_tr])
    subj_mean = train_all["raw_prediction_1_5"].mean()
    subj_std = train_all["raw_prediction_1_5"].std()
    if not np.isfinite(subj_mean) or not np.isfinite(subj_std) or subj_std == 0:
        raise ValueError(f"statistiques de subjectivité dégénérées: mean={subj_mean}, std={subj_std}")

    Xtr, Ltr, Ytr, idtr = build_features(hwt_tr, mgt_tr, subj_mean, subj_std,
                                         feature_version=segmentation_version)
    Xva, Lva, Yva, idva = build_features(hwt_va, mgt_va, subj_mean, subj_std,
                                         feature_version=segmentation_version)

    # garde-fou : le one-hot est construit par np.eye[cat_idx] ; un label hors
    # vocabulaire produit un NaN d'indexation, pas une exception explicite
    if torch.isnan(Xtr).any() or torch.isnan(Xva).any():
        raise ValueError("NaN dans les features — label d'acte de langage hors du vocabulaire "
                         f"attendu pour feature_version={segmentation_version}")
    if Xtr.shape[-1] != Xva.shape[-1]:
        raise ValueError(f"dimension d'entrée incohérente: {Xtr.shape[-1]} vs {Xva.shape[-1]}")

    return (Xtr, Ltr, Ytr, idtr), (Xva, Lva, Yva, idva), float(subj_mean), float(subj_std), segmentation_version


def stratified_holdout(y: torch.Tensor, frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Split interne train/dev. Une ligne = une review, donc pas de fuite de groupe."""
    sss = StratifiedShuffleSplit(n_splits=1, test_size=frac, random_state=seed)
    return next(sss.split(np.zeros(len(y)), y.numpy()))


# --------------------------------------------------------------------------- #
# Optimisation
# --------------------------------------------------------------------------- #
def build_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    """AdamW avec exclusion du weight decay sur les biais et les LayerNorm.
    Pénaliser les gains de normalisation dégrade systématiquement les petits Transformers."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 or name.endswith(".cls") else decay).append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}], lr=lr
    )


def build_scheduler(opt, total_steps: int, warmup_ratio: float):
    warmup = max(1, int(warmup_ratio * total_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


@torch.no_grad()
def _auc(model, dl, device: str) -> float:
    model.eval()
    ys, ps = [], []
    for xb, lb, yb in dl:
        ps.append(torch.sigmoid(model(xb.to(device), lb)).cpu().numpy())
        ys.append(yb.numpy())
    return float(roc_auc_score(np.concatenate(ys), np.concatenate(ps)))

@torch.no_grad()
def _eval_pass(model, dl, device: str, crit) -> tuple[float, float]:
    """Perte moyenne et AUC sur un DataLoader, en une seule traversée."""
    model.eval()
    tot, ys, ps = 0.0, [], []
    for xb, lb, yb in dl:
        xb, yb = xb.to(device), yb.float().to(device)
        logits = model(xb, lb)
        tot += crit(logits, yb).item() * len(yb)
        ps.append(torch.sigmoid(logits).cpu().numpy())
        ys.append(yb.cpu().numpy())
    return tot / len(dl.dataset), float(roc_auc_score(np.concatenate(ys), np.concatenate(ps)))

def train_transformer_model(
    model: nn.Module,
    train_dl: DataLoader,
    dev_dl: Optional[DataLoader] = None,
    epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    grad_clip: float = 1.0,
    patience: Optional[int] = None,
    device: str = "cpu",
    log_every: int = 3,
) -> dict:
    """Retourne l'historique. Si dev_dl et patience sont fournis, restaure les
    poids du meilleur epoch (AUC dev) — sinon budget d'epochs fixe, parité GRU."""
    print(f"Train on device : {device} | params = {sum(p.numel() for p in model.parameters()):,}")
    model.to(device)
    crit = nn.BCEWithLogitsLoss()
    opt = build_optimizer(model, lr, weight_decay)
    sched = build_scheduler(opt, epochs * max(1, len(train_dl)), warmup_ratio)

    hist = {"loss": [], "dev_loss": [], "dev_auc": []}
    best_auc, best_state, bad = -np.inf, None, 0

    for ep in range(epochs):
        model.train()
        tot = 0.0
        for xb, lb, yb in train_dl:
            xb, yb = xb.to(device), yb.float().to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb, lb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            sched.step()
            tot += loss.item() * len(yb)

        tr_loss = tot / len(train_dl.dataset)
        hist["loss"].append(tr_loss)
        verbose = True

        if dev_dl is not None:
            dev_loss, auc = _eval_pass(model, dev_dl, device, crit)
            hist["dev_loss"].append(dev_loss)
            hist["dev_auc"].append(auc)

            if auc > best_auc:
                best_auc, bad = auc, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1

            if verbose:
                # gap = écart de généralisation ; sa dérivée signale l'overfit
                # plus tôt que dev_loss, qui peut stagner avant de remonter
                print(f"epoch {ep:2d}  loss {tr_loss:.4f}  dev_loss {dev_loss:.4f}  "
                      f"gap {dev_loss - tr_loss:+.4f}  dev_auc {auc:.4f}  "
                      f"lr {sched.get_last_lr()[0]:.2e}")

            if patience is not None and bad >= patience:
                print(f"early stopping at epoch {ep} (best dev_auc {best_auc:.4f})")
                break
        elif verbose:
            print(f"epoch {ep:2d}  loss {tr_loss:.4f}  lr {sched.get_last_lr()[0]:.2e}")
    if best_state is not None:
        model.load_state_dict(best_state)
        hist["best_dev_auc"] = best_auc
    return hist


# --------------------------------------------------------------------------- #
# Entraînement complet
# --------------------------------------------------------------------------- #
@hydra.main(config_path="../../../../conf", config_name="train", version_base="1.3")
def _train_transformer(cfg: DictConfig):
    train_transformer(cfg)


def train_transformer(cfg: DictConfig):
    logger.info(OmegaConf.to_yaml(cfg))
    config_hash = generate_config_hash(cfg, exclude=["hardware"])
    local_dir = Path(RESULT_DIR / f"{config_hash}")
    local_dir.mkdir(parents=True, exist_ok=True)

    seed = int(cfg.training.model.get("seed", 0))
    set_seed(seed)

    (Xtr, Ltr, Ytr, _), (Xva, Lva, Yva, _), subj_mean, subj_std, seg_v = build_split_tensors(cfg)

    input_dim = int(Xtr.shape[-1])
    max_len_data = int(max(Ltr.max().item(), Lva.max().item()))
    label_cats = LABEL_CATS_1 if seg_v == 1 else LABEL_CATS_2
    assert input_dim == len(label_cats) + 2, "feature layout incohérent avec block_to_sequence"

    params_model = OmegaConf.to_container(cfg.training.model.params_model, resolve=True)
    declared = params_model.pop("input_dim", None)
    if declared is not None and int(declared) != input_dim:
        raise ValueError(f"input_dim déclaré {declared} != dimension réelle {input_dim}")
    params_model["input_dim"] = input_dim
    params_model["max_len"] = max(int(params_model.get("max_len", 0)), max_len_data)

    batch_size = cfg.training.model["batch_size"]
    dev_frac = float(cfg.training.model.get("dev_frac", 0.0))

    if dev_frac > 0:
        tr_idx, dev_idx = stratified_holdout(Ytr, dev_frac, seed)
        train_dl = DataLoader(TensorDataset(Xtr[tr_idx], Ltr[tr_idx], Ytr[tr_idx]),
                              batch_size=batch_size, shuffle=True, drop_last=False)
        dev_dl = DataLoader(TensorDataset(Xtr[dev_idx], Ltr[dev_idx], Ytr[dev_idx]),
                            batch_size=batch_size, shuffle=False)
    else:
        train_dl = DataLoader(TensorDataset(Xtr, Ltr, Ytr), batch_size=batch_size, shuffle=True)
        dev_dl = None

    val_dl = DataLoader(TensorDataset(Xva, Lva, Yva), batch_size=batch_size, shuffle=False)

    model = SubjectivityTransformer(**params_model)
    hist = train_transformer_model(
        model, train_dl, dev_dl=val_dl,
        **OmegaConf.to_container(cfg.training.model.params_training, resolve=True),
        device=cfg.hardware.device,
    )

    artifact = {
        "hash": config_hash,
        "cfg": OmegaConf.to_yaml(cfg, resolve=True),
        "model": model,                       # chargement direct, compatible SageGRU
        "model_class": "SubjectivityTransformer",
        "model_kwargs": params_model,         # reconstruction robuste si le pickle casse
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "subj_mean": subj_mean,
        "subj_std": subj_std,
        "seed": seed,
        "history": hist,
        "torch_version": torch.__version__,
        "feature_spec": {                     # contrat d'interface features <-> modèle
            "segmentation_version": seg_v,
            "label_cats": list(label_cats),
            "columns": [*label_cats, "subj_z", "subj_valid"],
            "input_dim": input_dim,
            "max_len_train": max_len_data,
        },
    }
    joblib.dump(artifact, local_dir / "artifact.joblib")
    print(local_dir / "artifact.joblib")

    y_te, p_te = evaluate(model, val_dl, cfg.hardware.device)
    np.savez(local_dir / "transformer_result_1.npz", y_true=y_te, y_proba=p_te)

    fig, ax = plt.subplots(figsize=(6, 6))
    RocCurveDisplay.from_predictions(y_te, p_te, ax=ax, name="Transformer")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="aléatoire")
    ax.set_xlabel("Taux de faux positifs (FPR)")
    ax.set_ylabel("Taux de vrais positifs (TPR)")
    ax.set_title("Courbe ROC — Transformer")
    ax.legend()
    plt.tight_layout()
    fig.savefig(local_dir / "roc_transformer.png", dpi=150)
    plt.close(fig)
    return model, artifact


if __name__ == "__main__":
    _train_transformer()