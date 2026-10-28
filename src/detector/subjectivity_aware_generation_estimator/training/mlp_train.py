import logging
from pathlib import Path

import hydra
import joblib
import pandas as pd
import numpy as np
import sklearn
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import RESULT_DIR
from src.data.dataset_aprdb import AprdbDataset
from src.detector.subjectivity_aware_generation_estimator.training.linguistic_segmentation import segment_database
from src.evaluation.eval_utils import generate_config_hash
import pickle

logger = logging.getLogger(__name__)


LABEL_CATS = ["Question", "Neither", "Assertion"]
SCORE_CATS = [1.0, 2.0, 3.0, 4.0, 5.0]

def block_to_features(block: pd.DataFrame) -> np.ndarray:
    n_b = len(block)  # dénominateur UNIQUE partagé par les deux sous-vecteurs

    # Bloc-1 : proportions des labels sur le bloc entier.
    # reindex force les 3 catégories dans l'ordre fixe ; fill_value=0
    # gère un label absent du bloc sans lever d'erreur.
    label_counts = (block["rule_based_SA-label"]
                    .value_counts()
                    .reindex(LABEL_CATS, fill_value=0))
    h1 = label_counts.to_numpy() / n_b

    # Bloc-2 : proportions des valeurs de score, MÊME dénominateur n_b.
    # Les None (lignes non-Assertion) sont ignorés par value_counts,
    # donc la somme de h2 vaut mécaniquement p_assert.
    score_counts = (block["rounded_prediction"]
                    .value_counts()
                    .reindex(SCORE_CATS, fill_value=0))
    h2 = score_counts.to_numpy() / n_b

    return np.concatenate([h1, h2])

def df_to_feature_matrix(df: pd.DataFrame):
    """Retourne (X, ids) : X de forme (n_blocs, 8), ids l'ordre des review-id."""
    ids, rows = [], []
    # sort=False : préserve l'ordre d'apparition, pas de tri implicite
    for rid, block in df.groupby("review-id", sort=False):
        ids.append(rid)
        rows.append(block_to_features(block))
    X = np.vstack(rows)            # (n_blocs, 8) — vstack garantit la forme 2D
    return X, np.array(ids)



@hydra.main(config_path="../../../../conf", config_name="train", version_base="1.3")
def _train_mlp(cfg: DictConfig):
    train_mlp(cfg)

def train_mlp(cfg : DictConfig) -> None:
    logger.info(OmegaConf.to_yaml(cfg))
    print("building config hash:")
    config_hash = generate_config_hash(cfg, exclude=['hardware'])

    local_dir = Path(RESULT_DIR / f"{config_hash}")
    local_dir.mkdir(parents=True, exist_ok=True)


    ds_train = AprdbDataset(config=cfg.dataset_train)
    df_train = segment_database(ds_train)

    hwt_df_train = df_train[df_train["label"] == 0]
    mgt_df_train = df_train[df_train["label"] == 1]

    # df_to_feature_matrix renvoie un TUPLE (X, ids) — il faut déballer.
    X_hwt, ids_hwt = df_to_feature_matrix(hwt_df_train)
    X_mgt, ids_mgt = df_to_feature_matrix(mgt_df_train)

    # Labels constants par source. np.full fixe dtype=int et la bonne longueur.
    y_hwt = np.full(len(X_hwt), 0, dtype=int)   # HWT → 0
    y_mgt = np.full(len(X_mgt), 1, dtype=int)   # MGT → 1

    # Dataset supervisé
    X = np.vstack([X_hwt, X_mgt])         # (n_hwt + n_mgt, 8)
    y = np.concatenate([y_hwt, y_mgt])


    ds_test = AprdbDataset(config=cfg.dataset_val)
    df_test = segment_database(ds_test)

    hwt_df_test = df_test[df_test["label"] == 0]
    mgt_df_test = df_test[df_test["label"] == 1]

    # df_to_feature_matrix renvoie un TUPLE (X, ids) — il faut déballer.
    X_hwt_test, ids_hwt_test = df_to_feature_matrix(hwt_df_test)
    X_mgt_test, ids_mgt_test = df_to_feature_matrix(mgt_df_test)

    # Labels constants par source. np.full fixe dtype=int et la bonne longueur.
    y_hwt_test = np.full(len(X_hwt_test), 0, dtype=int)   # HWT → 0
    y_mgt_test = np.full(len(X_mgt_test), 1, dtype=int)   # MGT → 1

    # Dataset supervisé
    X_test = np.vstack([X_hwt_test, X_mgt_test])         # (n_hwt + n_mgt, 8)
    y_test = np.concatenate([y_hwt_test, y_mgt_test])
    y_tr = y
    y_te = y_test
    # 2) STANDARDISATION fit sur le TRAIN seul, puis transform du test.

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(**cfg.training.model.params)),
    ])

    pipe.fit(X, y_tr)

    # 4) ÉVALUATION honnête sur le test.
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    print("itérations effectuées :", pipe.named_steps["clf"].n_iter_)
    print("\nmatrice de confusion [[TN FP][FN TP]] :")
    print(confusion_matrix(y_te, y_pred))
    print("\nclassification report :")
    print(classification_report(y_te, y_pred, digits=3))
    print("AUC :", round(roc_auc_score(y_te, y_proba), 4))

    result_file = f"mlp_result.npz"
    np.savez(local_dir / f"{result_file}", y_true=y_te, y_proba=y_proba)

    artifact = {
        "hash": config_hash,
        "cfg" : OmegaConf.to_yaml(cfg, resolve=True),
        "sklearn_version": sklearn.__version__,
        "pipeline": pipe
    }

    joblib.dump(artifact, local_dir / "artifact.joblib")
    print(local_dir / "artifact.joblib")
    #<CACHE_DIR>/results/<hash>/artifact.joblib
if __name__ == '__main__':
    _train_mlp()