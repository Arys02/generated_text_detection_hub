# file: src/detector/subjectivity_aware_generation_estimator/xai/normalized_segmentation.py
from __future__ import annotations

from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

from src.config import RESULT_DIR
from src.detector.subjectivity_aware_generation_estimator.training.linguistic_segmentation import (
    segment_dataframe_,
)
from src.detector.subjectivity_aware_generation_estimator.xai.text_normalization import normalize_dataframe
from src.evaluation.eval_utils import append_to_jsonl, flatten_config, generate_config_hash


def segment_database_normalized(dataset, segmentation: int = 1, subj_category=None,
                                device: str = "cuda:1") -> pd.DataFrame:
    """Identique à segment_database_, mais le texte brut est normalisé avant segmentation.

    La clé de cache porte le suffixe `_norm` : indispensable, sinon la fonction relit
    le CSV non normalisé et renvoie silencieusement les mêmes features.
    """
    if subj_category is None:
        subj_category = ["Assertion"]

    feature_hash = generate_config_hash(
        {"segmentation": segmentation, "subj_category": sorted(subj_category)}
    )[:8]
    local_dir = RESULT_DIR / "implicite_segmentation" / f"{dataset.get_hash()}_{feature_hash}_norm"
    local_dir.mkdir(parents=True, exist_ok=True)
    output_file_path = local_dir / "sentences_degrees_final.csv"

    if output_file_path.exists():
        print(f"📦 Loading normalized data from cache: {output_file_path}")
        return pd.read_csv(output_file_path)

    df_raw = dataset.to_pandas()
    df_norm = normalize_dataframe(df_raw, column="content")
    n_changed = int((df_raw["content"].fillna("") != df_norm["content"]).sum())
    print(f"normalisation : {n_changed}/{len(df_norm)} textes modifiés")

    df, n_assertions = segment_dataframe_(df_norm, device, segmentation, subj_category)
    df.to_csv(output_file_path, index=False, encoding="utf-8-sig")

    flat = flatten_config(OmegaConf.to_container(dataset.get_config()))
    append_to_jsonl(
        {"hash": dataset.get_hash(), **flat, "version": segmentation, "normalized": True},
        filepath=RESULT_DIR / "implicite_segmentation" / "mapping_hash_config.json",
    )
    print(f"✅ {output_file_path}  |  {n_assertions} phrases scorées")
    return df