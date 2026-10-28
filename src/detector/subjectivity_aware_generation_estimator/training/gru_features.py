#file : gru_features.py
import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence

LABEL_CATS_2 = ["Assertion", "Expressive", "Neither", 'Question', 'Suggestion']
LABEL_CATS_1 = ["Assertion", "Neither", 'Question']

from src.detector.subjectivity_aware_generation_estimator.features.feature_spec import (
    FeatureSpec, get_spec,
)

def _as_spec(feature_version) -> FeatureSpec:
    """Accepte une FeatureSpec, un nom de spec, ou un entier de segmentation (legacy)."""
    if isinstance(feature_version, FeatureSpec):
        return feature_version
    if isinstance(feature_version, str):
        return get_spec(feature_version)
    return get_spec("v1" if feature_version == 1 else "v2")


def block_to_sequence(block: pd.DataFrame, subj_mean=0., subj_std=0.,
                      order_col: str | None = None, feature_version=1) -> np.ndarray:
    spec = _as_spec(feature_version)
    if order_col is not None:
        block = block.sort_values(order_col)

    label_cats = spec.label_cats
    cat_idx = block["rule_based_SA-label"].map({c: i for i, c in enumerate(label_cats)})
    if cat_idx.isna().any():
        bad = sorted(set(block.loc[cat_idx.isna(), "rule_based_SA-label"]))
        raise ValueError(f"catégories hors vocabulaire {spec.name} : {bad}")
    onehot = np.eye(len(label_cats))[cat_idx.to_numpy(dtype=int)]

    raw = block["raw_prediction_1_5"].to_numpy(dtype=float)
    value = np.where(np.isnan(raw), 0.0, (raw - subj_mean) / subj_std)

    if not spec.include_valid:
        return np.column_stack([onehot, value])
    return np.column_stack([onehot, value, (~np.isnan(raw)).astype(float)])


def df_to_padded_batch(df, label, subj_mean=None, subj_std=None, order_col=None, feature_version=None):
    """Retourne (padded, lengths, labels, ids).
       padded  : (N_blocs, L_max, 5) float32
       lengths : (N_blocs,) int64  -- longueurs réelles, pour pack_padded_sequence
       labels  : (N_blocs,) int64  -- constante = label
       ids     : ordre des review-id (traçabilité)
    """
    seqs, lengths, ids = [], [], []
    for rid, block in df.groupby("review-id", sort=False):
        s = block_to_sequence(block, subj_mean, subj_std, order_col=order_col, feature_version=feature_version)
        seqs.append(torch.tensor(s, dtype=torch.float32))
        lengths.append(len(s))
        ids.append(rid)
    # pad_sequence avec batch_first : (N, L_max, 5), padding=0.0
    padded = pad_sequence(seqs, batch_first=True, padding_value=0.0)
    lengths = torch.tensor(lengths, dtype=torch.int64)
    labels = torch.full((len(seqs),), label, dtype=torch.int64)
    return padded, lengths, labels, np.array(ids)


def build_features(hwt, mgt, subj_mean, subj_std, order_col=None, feature_version = 1):
    """Assemble HWT(0)+MGT(1) en (padded, lengths, labels, ids) avec un L_max GLOBAL."""
    seqs, lengths, labels, ids = [], [], [], []
    for df, lab in [(hwt, 0), (mgt, 1)]:
        for rid, block in df.groupby("review-id", sort=False):
            s = block_to_sequence(block, subj_mean, subj_std, order_col, feature_version=feature_version)
            seqs.append(torch.tensor(s, dtype=torch.float32))
            lengths.append(len(s))
            labels.append(lab)
            ids.append(rid)
    padded = pad_sequence(seqs, batch_first=True, padding_value=0.0)  # L_max global
    return (padded,
            torch.tensor(lengths, dtype=torch.int64),
            torch.tensor(labels, dtype=torch.int64),
            np.array(ids))


def build_features_mono(df, subj_mean, subj_std, order_col=None, feature_version = 1):
    if "label" in df.columns and df.groupby("review-id")["label"].nunique().gt(1).any():
        raise ValueError(
            "review-id ambigu : plusieurs labels par id. build_features_mono fusionnerait "
            "les versions HWT et MGT en une seule séquence. Filtrer par label en amont."
        )
    """Séquences padées + longueurs + ids, sans labels (inférence)."""
    seqs, lengths, ids = [], [], []
    for rid, block in df.groupby("review-id", sort=False):
        s = block_to_sequence(block, subj_mean, subj_std, order_col, feature_version=feature_version)
        seqs.append(torch.tensor(s, dtype=torch.float32))
        lengths.append(len(s))
        ids.append(rid)
    padded = pad_sequence(seqs, batch_first=True, padding_value=0.0)
    return padded, torch.tensor(lengths, dtype=torch.int64), np.array(ids)
