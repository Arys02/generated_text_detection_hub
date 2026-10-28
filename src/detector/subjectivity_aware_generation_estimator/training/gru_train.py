# file : gru_train.py
import hydra
import joblib
import torch
import logging
import random

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

from pathlib import Path
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, f1_score
from torch.utils.data import TensorDataset, DataLoader

from src.config import RESULT_DIR
from src.data.dataset_aprdb import AprdbDataset
from src.data.dataset_factory import dataset_factory
from src.detector.subjectivity_aware_generation_estimator.features.pairing import pair_and_balance
from src.detector.subjectivity_aware_generation_estimator.model.SimpleGRU import SimpleGRU
from src.detector.subjectivity_aware_generation_estimator.training.gru_features import build_features
from src.evaluation.eval_utils import generate_config_hash
from src.detector.subjectivity_aware_generation_estimator.training.linguistic_segmentation import segment_database, \
    segment_database_
from src.detector.subjectivity_aware_generation_estimator.features.feature_spec import get_spec
from src.detector.subjectivity_aware_generation_estimator.features.tensor_builder import (
    build_dataset, fit_context,
)
from src.detector.subjectivity_aware_generation_estimator.training.linguistic_segmentation import (
    segment_database_spec,
)

logger = logging.getLogger(__name__)

load_dotenv()

LABEL_CATS = ["Question", "Neither", "Assertion"]
SCORE_CATS = [1.0, 2.0, 3.0, 4.0, 5.0]
SUBJ_IMPUTE = 0.0

META_COLS = ["conference", "year", "generator"]

# noms lisibles pour les rapports 3 classes (encodage label_3c de SemDetectDataset)
THREE_CLASS_NAMES = ["human", "rewrite", "ai"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _block_meta(df: pd.DataFrame, ids, labels) -> pd.DataFrame:
    """Une ligne par bloc, alignée sur l'ordre de build_dataset."""
    first = (df.drop_duplicates(["review-id", "label"])
               .set_index(["review-id", "label"])[META_COLS])
    return first.loc[list(zip(ids, labels))].reset_index()


def _check_val_split(cfg: DictConfig) -> None:
    """Un split de validation virtuel (val_fraction) n'est cohérent que si train et val
    partagent exactement le même tirage de papiers ; sinon des papiers fuient entre les deux."""
    tr, va = cfg.dataset_train, cfg.dataset_val
    if tr.get("name") != va.get("name"):
        return
    for key in ("val_fraction", "val_seed"):
        if tr.get(key) != va.get(key):
            raise ValueError(f"dataset_train.{key}={tr.get(key)!r} != dataset_val.{key}={va.get(key)!r} : "
                             f"le tirage des papiers de validation doit être identique")
    if tr.get("val_fraction") is not None and tr.get("split") == va.get("split"):
        raise ValueError(f"dataset_train et dataset_val sur le même split {tr.get('split')!r}")


def _load_blocks(ds_cfg: DictConfig, spec) -> pd.DataFrame:
    """Segmente puis équilibre. Un dataset déjà apparié en amont (balance=paired) n'est pas
    ré-apparié : pair_and_balance suppose un review-id partagé HWT/MGT, propre à APRDB."""
    df = segment_database_spec(dataset_factory(ds_cfg), spec)
    if ds_cfg.get("balance") == "paired":
        print(f"[{ds_cfg.get('name')} / split={ds_cfg.get('split')}] déjà apparié par le dataset : "
              f"pair_and_balance ignoré | blocs par label : "
              f"{df.drop_duplicates('review-id')['label'].value_counts().sort_index().to_dict()}")
        return df
    return pair_and_balance(df, seed=ds_cfg.seed)


def prepare_data(cfg: DictConfig) -> dict:

    spec = get_spec(cfg.training.model.feature_spec)
    _check_val_split(cfg)

    df_train = _load_blocks(cfg.dataset_train, spec)
    df_val = _load_blocks(cfg.dataset_val, spec)

    ctx = fit_context(df_train, spec)

    Xtr, Ltr, Ytr, idtr = build_dataset(df_train, spec, ctx)
    Xva, Lva, Yva, idva = build_dataset(df_val, spec, ctx)

    assert (Ltr > 0).all() and (Lva > 0).all(), "séquence de longueur nulle"
    assert Xtr.shape[-1] == Xva.shape[-1] == spec.input_dim

    return {"spec": spec, "ctx": ctx,
            "df_train": df_train, "df_val": df_val,
            "Xtr": Xtr, "Ltr": Ltr, "Ytr": Ytr, "idtr": idtr,
            "Xva": Xva, "Lva": Lva, "Yva": Yva, "idva": idva,
            "meta_va": _block_meta(df_val, idva, Yva.numpy())}


def train_model(model, train_dl, epochs=30, lr=1e-3, device="cpu", n_outputs=1):
    print(f"Train on device : {device}")
    model.to(device)
    # équilibre -> pas de pos_weight / class weights
    crit = nn.BCEWithLogitsLoss() if n_outputs == 1 else nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        model.train();
        tot = 0.0
        for xb, lb, yb in train_dl:
            xb = xb.to(device)
            yb = yb.float().to(device) if n_outputs == 1 else yb.long().to(device)
            opt.zero_grad()
            loss = crit(model(xb, lb), yb)
            loss.backward();
            opt.step()
            tot += loss.item() * len(yb)
        if ep % 3 == 0 or ep == epochs - 1:
            print(f"epoch {ep:2d}  loss {tot / len(train_dl.dataset):.4f}")


@torch.no_grad()
def evaluate(model, test_dl, device="cpu"):

    model.eval();
    ys, ps = [], []
    for xb, lb, yb in test_dl:
        proba = torch.sigmoid(model(xb.to(device), lb)).cpu().numpy()
        ps.append(proba)
        ys.append(yb.numpy())
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    pred = (p >= 0.5).astype(int)
    print("\nconfusion [[TN FP][FN TP]] :\n", confusion_matrix(y, pred))
    print(classification_report(y, pred, digits=3))
    print("AUC :", round(roc_auc_score(y, p), 4))
    return y, p  # <-- pour la ROC


@torch.no_grad()
def predict_proba(model, dl, device: str, n_outputs: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """n_outputs=1 : P(label=1), forme (N,) ; n_outputs=K : softmax, forme (N, K)."""
    model.eval()
    ys, ps = [], []
    for xb, lb, yb in dl:
        logits = model(xb.to(device), lb)
        proba = torch.sigmoid(logits) if n_outputs == 1 else torch.softmax(logits, dim=-1)
        ps.append(proba.cpu().numpy())
        ys.append(yb.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def stratified_auc(y: np.ndarray, p: np.ndarray, meta: pd.DataFrame,
                   by: list[str]) -> pd.DataFrame:
    """AUC par cellule. Une cellule mono-classe n'a pas d'AUC définie -> NaN."""
    d = meta.copy()
    d["y"], d["p"] = y, p
    rows = []
    for key, g in d.groupby(by, sort=True):
        auc = (roc_auc_score(g["y"], g["p"]) if g["y"].nunique() == 2 else np.nan)
        rows.append({**dict(zip(by, key if isinstance(key, tuple) else (key,))),
                     "n": len(g), "auc": auc})
    return pd.DataFrame(rows).sort_values("auc", na_position="last")


def auc_by_generator_vs_all_negatives(y: np.ndarray, p: np.ndarray, meta: pd.DataFrame) -> pd.DataFrame:
    """Positifs d'un générateur contre TOUS les négatifs. Nécessaire quand les négatifs
    n'ont pas de générateur propre (Sem-Detect : generator='human'), sinon chaque cellule
    par générateur est mono-classe."""
    d = meta.copy()
    d["y"], d["p"] = y, p
    neg = d[d["y"] == 0]
    rows = []
    for gen, pos in d[d["y"] == 1].groupby("generator", sort=True):
        part = pd.concat([pos, neg])
        rows.append({"generator": gen, "n_pos": len(pos), "n_neg": len(neg),
                     "auc": roc_auc_score(part["y"], part["p"])})
    return pd.DataFrame(rows).sort_values("auc")


@hydra.main(config_path="../../../../conf", config_name="train", version_base="1.3")
def _train_gru(cfg: DictConfig):
    train_gru(cfg)


def train_gru(cfg: DictConfig):
    logger.info(OmegaConf.to_yaml(cfg))
    config_hash = generate_config_hash(cfg, exclude=["hardware"])
    local_dir = Path(RESULT_DIR / f"{config_hash}")
    local_dir.mkdir(parents=True, exist_ok=True)

    seed = int(cfg.training.model.get("seed", 0))
    set_seed(seed)
    d = prepare_data(cfg)
    spec, ctx = d["spec"], d["ctx"]

    params_model = OmegaConf.to_container(cfg.training.model.params_model, resolve=True)
    declared = params_model.pop("n_features", None)
    if declared is not None and int(declared) != spec.input_dim:
        raise ValueError(f"n_features={declared} déclaré dans le YAML, mais la spec "
                         f"{spec.name!r} produit {spec.input_dim} colonnes")
    params_model["n_features"] = spec.input_dim

    # n_outputs absent des presets binaires existants -> 1 : kwargs et hash inchangés
    n_outputs = int(params_model.get("n_outputs", 1))
    n_classes_train = int(torch.unique(d["Ytr"]).numel())
    expected = 2 if n_outputs == 1 else n_outputs
    if n_classes_train != expected:
        raise ValueError(f"n_outputs={n_outputs} attend {expected} classes, "
                         f"le train en contient {n_classes_train}")
    if not set(torch.unique(d["Yva"]).tolist()) <= set(torch.unique(d["Ytr"]).tolist()):
        raise ValueError("la validation contient des labels absents du train")

    bs = cfg.training.model["batch_size"]
    train_dl = DataLoader(TensorDataset(d["Xtr"], d["Ltr"], d["Ytr"]), batch_size=bs, shuffle=True)
    val_dl = DataLoader(TensorDataset(d["Xva"], d["Lva"], d["Yva"]), batch_size=bs, shuffle=False)

    # reseed : pair_and_balance a consommé de l'aléatoire, l'initialisation des poids
    # doit dépendre du seed déclaré et de rien d'autre
    set_seed(seed)
    model = SimpleGRU(**params_model)
    print(f"params = {sum(p.numel() for p in model.parameters()):,}")
    train_model(model, train_dl,
                **OmegaConf.to_container(cfg.training.model.params_training, resolve=True),
                device=cfg.hardware.device, n_outputs=n_outputs)

    y, p = predict_proba(model, val_dl, cfg.hardware.device, n_outputs=n_outputs)

    if n_outputs == 1:
        auc = float(roc_auc_score(y, p))
        print("\n" + classification_report(y, (p >= 0.5).astype(int), digits=3))
        print("AUC globale :", round(auc, 4))

        strat = stratified_auc(y, p, d["meta_va"], ["conference", "year"])
        print("\nAUC par cellule :\n", strat.to_string(index=False))
        if d["meta_va"].loc[y == 0, "generator"].nunique() <= 1:
            strat_gen = auc_by_generator_vs_all_negatives(y, p, d["meta_va"])
            print("\nAUC par générateur (positifs du générateur vs tous les négatifs) :\n",
                  strat_gen.to_string(index=False))
        else:
            strat_gen = stratified_auc(y, p, d["meta_va"], ["generator"])
            print("\nAUC par générateur :\n", strat_gen.to_string(index=False))
        metrics = {"auc": auc}
    else:
        pred = p.argmax(axis=1)
        auc = float(roc_auc_score(y, p, multi_class="ovr", average="macro"))
        macro_f1 = float(f1_score(y, pred, average="macro"))
        names = THREE_CLASS_NAMES if n_outputs == 3 else None
        print("\n" + classification_report(y, pred, digits=3, target_names=names))
        print("confusion (lignes = vrai, colonnes = prédit) :\n", confusion_matrix(y, pred))
        print("Macro-F1 :", round(macro_f1, 4), "| AUC OvR macro :", round(auc, 4))
        metrics = {"auc": auc, "macro_f1": macro_f1}

    artifact = {
        "hash": config_hash,
        "seed": seed,
        "cfg": OmegaConf.to_yaml(cfg, resolve=True),
        # state_dict CPU + kwargs : survit à un renommage de classe ou à l'absence du GPU
        "model_class": "SimpleGRU",
        "model_kwargs": params_model,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "feature_spec": spec.to_dict(),  # contrat validé au chargement par Sage
        "ctx": ctx,
        "n_outputs": n_outputs,
        "auc": auc,
        "metrics": metrics,
        "torch_version": torch.__version__,
    }
    joblib.dump(artifact, local_dir / "artifact.joblib")
    np.savez(local_dir / "gru_result.npz", y_true=y, y_proba=p,
             **{c: d["meta_va"][c].to_numpy() for c in META_COLS})
    print("\n", local_dir / "artifact.joblib")
    return artifact


if __name__ == '__main__':
    _train_gru()