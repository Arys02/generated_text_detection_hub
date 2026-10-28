# src/detector/sem_detect/feature_store.py
"""Cache disque reprenable des calculs coûteux de Sem-Detect.

Deux caches indépendants, chacun indexé par ce dont il dépend réellement :
  - emb/<clé> : embeddings des claims sémantiques  -> provenance des claims + modèle d'embedding
  - txt/<clé> : features textuelles (Mistral-7B)   -> provenance du texte + modèle + longueur max
Changer d'extracteur de claims ne recalcule donc pas les features textuelles.
Écriture atomique par fragments : un crash ne perd que le fragment en cours.
"""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.detector.sem_detect.semdetect.claim_extractor import (
    CLAIM_CATEGORIES, SEMANTIC_CATEGORIES, ClaimExtraction,
)
from src.detector.sem_detect.semdetect.feature_builder import MIN_EMBEDDINGS
from src.detector.sem_detect.semdetect.textual_features import TextualStats

logger = logging.getLogger(__name__)

STORE_VERSION = 2
DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_TEXTUAL_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"


def parse_claims(raw) -> ClaimExtraction:
    """JSON (str) ou dict -> ClaimExtraction. Clés inconnues : erreur, pas de perte silencieuse."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)) or raw == "":
        return ClaimExtraction()
    d = json.loads(raw) if isinstance(raw, str) else dict(raw)
    unknown = set(d) - set(CLAIM_CATEGORIES)
    if unknown:
        raise KeyError(f"catégories de claims inattendues : {sorted(unknown)} ; attendu {CLAIM_CATEGORIES}")
    return ClaimExtraction(**{c: [str(x) for x in (d.get(c) or [])] for c in CLAIM_CATEGORIES})


# ------------------------------------------------------------------ helpers disque
def _keyed_dir(parent: Path, config: dict) -> Path:
    key = hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
    d = Path(parent) / key
    d.mkdir(parents=True, exist_ok=True)
    cfg_path = d / "config.json"
    if not cfg_path.exists():
        cfg_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return d


def _load_shards(directory: Path, kind: str) -> dict:
    out: dict = {}
    for p in sorted(directory.glob(f"{kind}_*.joblib")):
        out.update(joblib.load(p))
    return out


def _write_shard(directory: Path, kind: str, data: dict) -> None:
    idx = len(list(directory.glob(f"{kind}_*.joblib")))
    path = directory / f"{kind}_{idx:05d}.joblib"
    tmp = path.with_suffix(".tmp")
    joblib.dump(data, tmp)
    tmp.replace(path)  # atomique : un fragment est complet ou absent


def _emb_config(claims_source: str, embedding_model: str) -> dict:
    return {"store_version": STORE_VERSION, "kind": "emb", "claims_source": claims_source,
            "embedding_model": embedding_model, "semantic_categories": list(SEMANTIC_CATEGORIES)}


def _txt_config(text_source: str, textual_model: str, textual_max_length: int) -> dict:
    return {"store_version": STORE_VERSION, "kind": "txt", "text_source": text_source,
            "textual_model": textual_model, "textual_max_length": textual_max_length}


class SemDetectFeatureStore:
    def __init__(
        self,
        root: Path,
        claims_source: str,
        text_source: str,
        device: str,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        textual_model: str = DEFAULT_TEXTUAL_MODEL,
        textual_max_length: int = 4096,
        shard_size: int = 256,
    ):
        if device == "cuda":
            raise ValueError("device explicite requis (ex. 'cuda:0') : 'cuda' active le mode multi-GPU de Sem-Detect")
        root = Path(root)
        self.embedding_model = embedding_model
        self.textual_model = textual_model
        self.textual_max_length = textual_max_length
        self.emb_dir = _keyed_dir(root / "emb", _emb_config(claims_source, embedding_model))
        self.txt_dir = _keyed_dir(root / "txt", _txt_config(text_source, textual_model, textual_max_length))

        self.device = device
        self.shard_size = shard_size
        self._embedder = None
        self._textual = None
        self._emb: Dict[str, Dict[str, Optional[np.ndarray]]] = _load_shards(self.emb_dir, "emb")
        self._txt: Dict[str, dict] = _load_shards(self.txt_dir, "txt")
        logger.info(f"Feature store : {len(self._emb)} embeddings ({self.emb_dir.name}), "
                    f"{len(self._txt)} stats textuelles ({self.txt_dir.name})")

    # ---------------------------------------------------------------- modèles
    @property
    def embedder(self):
        if self._embedder is None:
            from src.detector.sem_detect.semdetect.embedding_generator import EmbeddingGenerator
            self._embedder = EmbeddingGenerator(model_name=self.embedding_model, device=self.device)
        return self._embedder

    @property
    def textual_extractor(self):
        if self._textual is None:
            from src.detector.sem_detect.semdetect.textual_features import TextualFeatureExtractor
            self._textual = TextualFeatureExtractor(
                model_name=self.textual_model, max_length=self.textual_max_length, device=self.device,
            )
        return self._textual

    # ----------------------------------------------------------------- calcul
    def ensure_embeddings(self, df: pd.DataFrame, force_ids=None) -> None:
        """df : colonnes review-id, claim_extraction. Calcule les reviews absentes du cache, plus
        celles de force_ids (claims mises à jour sous la même provenance, ex. après une relance).
        Le nouveau fragment est écrit après les anciens : il l'emporte au rechargement."""
        force = set(force_ids or [])
        todo = (df.loc[~df["review-id"].isin(self._emb) | df["review-id"].isin(force),
                       ["review-id", "claim_extraction"]]
                  .drop_duplicates("review-id"))
        if todo.empty:
            return
        for start in tqdm(range(0, len(todo), self.shard_size), desc="Embeddings (fragments)"):
            chunk = todo.iloc[start:start + self.shard_size]
            texts, owners = [], []
            shard: Dict[str, Dict[str, list]] = {}
            for rid, raw in zip(chunk["review-id"], chunk["claim_extraction"]):
                claims = parse_claims(raw)
                shard[rid] = {cat: [] for cat in SEMANTIC_CATEGORIES}
                for cat in SEMANTIC_CATEGORIES:
                    for claim in getattr(claims, cat):
                        texts.append(claim)
                        owners.append((rid, cat))
            if texts:
                vecs = np.asarray(self.embedder.embed_texts(texts), dtype=np.float32)
                for (rid, cat), v in zip(owners, vecs):
                    shard[rid][cat].append(v)
            packed = {rid: {cat: (np.stack(v) if v else None) for cat, v in cats.items()}
                      for rid, cats in shard.items()}
            _write_shard(self.emb_dir, "emb", packed)
            self._emb.update(packed)


    def ensure_textual(self, df: pd.DataFrame) -> None:
        """df : colonnes review-id, content. Ne calcule que les reviews absentes du cache."""
        todo = df.loc[~df["review-id"].isin(self._txt), ["review-id", "content"]].drop_duplicates("review-id")
        if todo.empty:
            return
        n_shards = (len(todo) + self.shard_size - 1) // self.shard_size
        for i, start in enumerate(range(0, len(todo), self.shard_size)):
            chunk = todo.iloc[start:start + self.shard_size]
            stats = [self.textual_extractor.compute(t)
                     for t in tqdm(chunk["content"], desc=f"Textuel fragment {i + 1}/{n_shards}")]
            shard = {rid: asdict(s) for rid, s in zip(chunk["review-id"], stats)}
            _write_shard(self.txt_dir, "txt", shard)
            self._txt.update(shard)

    # ---------------------------------------------------------------- lecture
    def n_embeddings(self) -> int:
        return len(self._emb)

    def n_textual(self) -> int:
        return len(self._txt)

    def semantic(self, rid: str, min_embeddings: int = MIN_EMBEDDINGS) -> Optional[List[np.ndarray]]:
        """Embeddings des 3 catégories sémantiques, dans l'ordre de get_semantic_embeddings ;
        None si moins de min_embeddings (même règle que le code d'origine)."""
        cats = self._emb[rid]
        vecs = [v for cat in SEMANTIC_CATEGORIES if cats[cat] is not None for v in cats[cat]]
        return vecs if len(vecs) >= min_embeddings else None

    def textual(self, rid: str) -> TextualStats:
        return TextualStats(**self._txt[rid])


def migrate_legacy_store(root: Path, text_source: str) -> List[dict]:
    """Copie (sans supprimer) les fragments d'un cache v1 (embeddings et stats dans un même dossier)
    vers les deux caches v2. Idempotent : un cache v2 déjà peuplé n'est pas réécrit."""
    root = Path(root)
    report = []
    for legacy in sorted(p for p in root.iterdir() if p.is_dir() and (p / "config.json").exists()):
        cfg = json.loads((legacy / "config.json").read_text(encoding="utf-8"))
        if cfg.get("store_version") != 1:
            continue
        targets = {
            "emb": _keyed_dir(root / "emb", _emb_config(cfg["claims_source"], cfg["embedding_model"])),
            "txt": _keyed_dir(root / "txt", _txt_config(text_source, cfg["textual_model"], cfg["textual_max_length"])),
        }
        for kind, target in targets.items():
            shards = sorted(legacy.glob(f"{kind}_*.joblib"))
            if any(target.glob(f"{kind}_*.joblib")):
                report.append({"legacy": legacy.name, "kind": kind, "target": target.name, "action": "déjà migré"})
                continue
            for p in shards:
                shutil.copy2(p, target / p.name)
            report.append({"legacy": legacy.name, "kind": kind, "target": target.name,
                           "action": f"{len(shards)} fragments copiés"})
    return report