# src/data/dataset_semdetect.py
import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from datasets import Dataset as HFDataset
from datasets import concatenate_datasets, load_dataset
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset

import src.config  # noqa: F401  (effet de bord : chargement du .env, comme dans dataset_aprdb.py)
from src.evaluation.eval_utils import generate_config_hash

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REPO_ID = "Sem-Detect/ML_Conferences-Peer-Reviews"

# conférence -> préfixe des partitions HF (iclr_2021, neurips_2022, ...)
CONFERENCES: Dict[str, str] = {"ICLR": "iclr", "NeurIPS": "neurips"}
YEARS: List[str] = ["2021", "2022"]
# "val" est virtuel : papiers du train réservés via val_fraction
SPLITS: List[str] = ["train", "val", "test"]
GENERATORS: List[str] = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "deepseek.v3-v1:0",
    "qwen.qwen3-235b-a22b-2507-v1:0",
]

# Encodage 3 classes, ordre du papier : f(t, A_p) -> {0: human, 1: LLM-refined, 2: AI}
CLASS_TO_3C: Dict[str, int] = {"human": 0, "rewrite": 1, "ai": 2}

# None = classe exclue du dataset
LABEL_MODES: Dict[str, Dict[str, Optional[int]]] = {
    "human_vs_ai": {"human": 0, "rewrite": None, "ai": 1},
    "ai_vs_rest": {"human": 0, "rewrite": 0, "ai": 1},
    "human_vs_llm": {"human": 0, "rewrite": 1, "ai": 1},
    "three_class": {"human": 0, "rewrite": 1, "ai": 2},
}
BALANCE_MODES = {"uniform", "none", "paired"}

# Appariement IA <-> humaine : même papier et même note (les reviews IA sont générées à note imposée)
PAIR_KEYS = ["conference", "year", "submission_id", "rating"]

# Incrémenter à chaque changement de logique de chargement -> invalide les caches existants
_LOADER_VERSION = 3

_STR_COLS = [
    "paper_id", "submission_id", "conference", "paper_title", "review_id",
    "og_review_id", "author", "class", "rating", "clean_review", "split",
]


def _as_list(x: Any) -> Optional[List[str]]:
    if x is None:
        return None
    if isinstance(x, (str, int)):
        return [str(x)]
    return [str(v) for v in x]


def _validate_subset(values: List[str], allowed: List[str], name: str) -> List[str]:
    unknown = sorted(set(values) - set(allowed))
    if unknown:
        raise ValueError(f"{name} inconnu(s) : {unknown}. Valeurs admises : {allowed}")
    return values


def _stable_unit(key: str, seed: int) -> float:
    """Tirage uniforme déterministe dans [0, 1). md5 et non hash() : ce dernier est
    randomisé par PYTHONHASHSEED, donc non reproductible entre processus."""
    h = hashlib.md5(f"{seed}:{key}".encode()).hexdigest()
    return int(h[:15], 16) / float(16 ** 15)


class SemDetectDataset(Dataset):
    """
    Sem-Detect (Duarte et al., 2026, arXiv:2605.21713) : reviews ICLR/NeurIPS 2021-2022,
    classes human / rewrite (LLM-refined) / ai.

    Items : dict avec au minimum
      content (str), label (int, selon label_mode), label_3c (int, 0/1/2), review-id (str, unique)
    """

    def __init__(
        self,
        config,
        use_tokenizer=None,
        cache_dir: Optional[str] = None,
        no_cache: bool = False,
    ):
        cache_dir = cache_dir if cache_dir is not None else os.getenv("CACHE_DIR")
        self.config = config
        cfg: Dict[str, Any] = (
            OmegaConf.to_container(config, resolve=True) if isinstance(config, DictConfig) else dict(config)
        )
        self.cfg = cfg

        self.repo_id: str = cfg.get("repo_id", REPO_ID)
        self.revision: Optional[str] = cfg.get("revision")

        self.splits = _validate_subset(_as_list(cfg.get("split", "test")), SPLITS, "split")
        self.conferences = _validate_subset(
            _as_list(cfg.get("conference")) or list(CONFERENCES), list(CONFERENCES), "conference"
        )
        self.years = _validate_subset(_as_list(cfg.get("date")) or list(YEARS), YEARS, "date")

        self.generators = _as_list(cfg.get("generator"))
        if self.generators is not None:
            _validate_subset(self.generators, GENERATORS, "generator")

        self.label_mode: str = cfg.get("label_mode", "human_vs_ai")
        if self.label_mode not in LABEL_MODES:
            raise ValueError(f"label_mode inconnu : {self.label_mode!r}. Valeurs admises : {list(LABEL_MODES)}")

        self.balance: str = cfg.get("balance", "none")
        if self.balance not in BALANCE_MODES:
            raise ValueError(f"balance inconnu : {self.balance!r}. Valeurs admises : {sorted(BALANCE_MODES)}")
        if self.balance == "paired" and self.label_mode not in ("human_vs_ai", "three_class"):
            logger.warning(
                f"balance=paired équilibre les classes réelles (label_3c), pas `label` : "
                f"avec label_mode={self.label_mode!r}, les labels binaires seront déséquilibrés (2:1)."
            )

        self.subset_size: Optional[int] = cfg.get("subset_size")
        if self.subset_size is not None and (not isinstance(self.subset_size, int) or self.subset_size <= 0):
            raise ValueError(f"subset_size doit être un entier > 0 ou null, reçu {self.subset_size!r}")

        self.val_fraction: Optional[float] = cfg.get("val_fraction")
        if self.val_fraction is not None and not (0.0 < float(self.val_fraction) < 1.0):
            raise ValueError(f"val_fraction doit être dans ]0, 1[ ou null, reçu {self.val_fraction!r}")
        if "val" in self.splits and self.val_fraction is None:
            raise ValueError("split 'val' demandé mais val_fraction est null.")
        self.val_seed = cfg.get("val_seed", 0)
        if not isinstance(self.val_seed, int) or isinstance(self.val_seed, bool):
            raise TypeError(f"val_seed doit être un entier, reçu {self.val_seed!r}")

        self.seed = cfg.get("seed", 0)
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise TypeError(f"dataset seed doit être un entier, reçu {self.seed!r} ({type(self.seed).__name__})")

        self.no_duplicate: bool = cfg.get("no_duplicate", True)
        self.keep_meta: bool = cfg.get("keep_meta", True)
        self.with_paper: bool = cfg.get("with_paper", False)
        self.keep_claims: bool = cfg.get("keep_claims", False)
        self.tokenizer = use_tokenizer

        self.exclude_ids_path: Optional[str] = cfg.get("exclude_ids_path")
        self.exclude_ids, exclude_digest = self._load_exclude_ids(self.exclude_ids_path)

        # ---- cache ----
        if cache_dir is None:
            raise EnvironmentError("CACHE_DIR n'est pas défini (variable d'environnement ou argument cache_dir).")
        self.cache_dir = Path(cache_dir)
        tokenizer_id = (
            None if use_tokenizer is None
            else getattr(use_tokenizer, "name_or_path", type(use_tokenizer).__name__)
        )
        # Le digest du fichier d'exclusion entre dans la clé : modifier la liste invalide le cache,
        # même si le chemin reste identique.
        self.cache_id = generate_config_hash(
            {
                **cfg,
                "_loader_version": _LOADER_VERSION,
                "_tokenizer": tokenizer_id,
                "_exclude_digest": exclude_digest,
            }
        )
        self.cache_path = self.cache_dir / "dataset_cache" / f"dataset_{self.cache_id}.pt"
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        if self.cache_path.exists() and not no_cache:
            logger.info(f"📦 Loading Sem-Detect dataset from cache: {self.cache_path}")
            self.data = torch.load(self.cache_path, weights_only=False)
        else:
            logger.info(f"Loading Sem-Detect dataset from {self.repo_id} (revision={self.revision})")
            self.data = self._build()
            torch.save(self.data, self.cache_path)

    # ------------------------------------------------------------------ API
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def get_hash(self) -> str:
        return self.cache_id

    def get_config(self):
        return self.config

    def to_pandas(self) -> pd.DataFrame:
        return self.data.to_pandas()

    # ------------------------------------------------------------ pipeline
    def _hf_splits(self) -> List[str]:
        return [f"{CONFERENCES[c]}_{y}" for c in self.conferences for y in self.years]

    def _load_config(self, name: str) -> pd.DataFrame:
        parts = [
            load_dataset(self.repo_id, name, split=s, revision=self.revision)
            for s in self._hf_splits()
        ]
        return concatenate_datasets(parts).to_pandas()

    def _build(self) -> HFDataset:
        df = self._load_config("reviews")
        n_raw = len(df)

        # Normalisation des types : `year` peut être inféré en datetime côté HF
        df["year"] = df["year"].astype(str).str[:4]
        for col in _STR_COLS:
            if col in df.columns:
                df[col] = df[col].fillna("").astype(str)

        unknown_classes = set(df["class"].unique()) - set(CLASS_TO_3C)
        if unknown_classes:
            raise ValueError(f"Classes inattendues dans le dataset : {sorted(unknown_classes)}")

        # ---- intégrité des identifiants (avant tout filtre) ----
        df = self._resolve_id_collisions(df)

        # ---- split de validation virtuel (niveau papier, déterministe, indépendant des filtres) ----
        if self.val_fraction is not None:
            is_val_paper = df["submission_id"].map(
                lambda s: _stable_unit(f"val:{s}", self.val_seed) < self.val_fraction
            )
            df.loc[df["split"].eq("train") & is_val_paper, "split"] = "val"
            logger.info(
                f"Split validation (val_fraction={self.val_fraction}, val_seed={self.val_seed}) : "
                f"papiers par split = {df.groupby('split')['submission_id'].nunique().to_dict()}"
            )

        # ---- filtres ----
        df = df[df["split"].isin(self.splits)]
        if self.generators is not None:
            df = df[(df["class"] == "human") | df["author"].isin(self.generators)]

        # ---- labels ----
        df = df.copy()
        df["label_3c"] = df["class"].map(CLASS_TO_3C).astype("int64")
        df["label"] = df["class"].map(LABEL_MODES[self.label_mode])
        df = df[df["label"].notna()].copy()
        df["label"] = df["label"].astype("int64")

        # ---- contenu ----
        df["content"] = df["clean_review"]
        df = df[df["content"].str.strip().ne("")]

        # ---- identifiants globaux : review_id n'est unique qu'au sein d'un papier ----
        prefix = df["conference"] + "-" + df["year"] + "-" + df["submission_id"]
        df["review-id"] = prefix + "-" + df["review_id"]
        df["source-review-id"] = np.where(df["og_review_id"].ne(""), prefix + "-" + df["og_review_id"], "")
        n_dup_ids = int(df["review-id"].duplicated().sum())
        if n_dup_ids:
            raise ValueError(f"{n_dup_ids} identifiants `review-id` dupliqués après construction de la clé composite.")

        df["subset"] = df["split"]
        df["generator"] = df["author"]
        df["forum_id"] = df["submission_id"]

        # ---- déduplication (même politique qu'APRDB : sur le texte) ----
        if self.no_duplicate:
            n_before = len(df)
            df = df.drop_duplicates(subset="content", keep="first")
            logger.info(f"Déduplication : {n_before - len(df)} doublons supprimés")

        # ---- exclusion explicite (ex. décontamination vs données d'entraînement d'un détecteur) ----
        if self.exclude_ids:
            mask = df["review-id"].isin(self.exclude_ids)
            logger.info(
                f"Exclusion ({self.exclude_ids_path}) : {int(mask.sum())} lignes retirées "
                f"sur {len(self.exclude_ids)} ids listés\n"
                f"{df.loc[mask].groupby(['class', 'conference', 'year']).size()}"
            )
            df = df.loc[~mask]

        # ---- échantillonnage ----
        df = self._sample(df)

        # ---- jointure papiers (après échantillonnage : merge plus léger) ----
        if self.with_paper:
            df = self._attach_papers(df)

        # ---- tokenisation optionnelle (parité avec AprdbDataset) ----
        if self.tokenizer is not None:
            df["content"] = df["content"].map(self.tokenizer.encode)

        cols = ["content", "label", "label_3c", "review-id"]
        if self.keep_meta:
            cols += [
                "subset", "generator", "class", "conference", "year", "paper_id",
                "submission_id", "forum_id", "rating", "source-review-id",
            ]
        if "pair-id" in df.columns:
            cols.append("pair-id")
        if self.keep_claims:
            df["claim_extraction"] = df["claim_extraction"].fillna("").astype(str)
            cols.append("claim_extraction")
        if self.with_paper:
            cols += [c for c in ("paper_content", "paper_appendix", "forum_id") if c not in cols]

        df = df[cols].reset_index(drop=True)
        self._log_stats(df, n_raw)
        return HFDataset.from_pandas(df, preserve_index=False)

    @staticmethod
    def _load_exclude_ids(path: Optional[str]):
        if path is None:
            return frozenset(), None
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"exclude_ids_path introuvable : {p}")
        raw = p.read_bytes()
        ids = frozenset(line.strip() for line in raw.decode("utf-8").splitlines() if line.strip())
        return ids, hashlib.sha256(raw).hexdigest()

    @staticmethod
    def _resolve_id_collisions(df: pd.DataFrame) -> pd.DataFrame:
        """
        1. Supprime les doublons stricts (toutes colonnes identiques) : sans perte.
        2. Exclut les collisions (même clé composite, contenus différents) : origine ambiguë.
        3. Exclut les rewrites dont la review humaine source est une collision : lien og_review_id ambigu.
        Appliqué AVANT les filtres split/label pour un traitement identique en train et en test.
        """
        n0 = len(df)
        df = df.drop_duplicates().copy()
        n_strict = n0 - len(df)

        prefix = df["conference"] + "-" + df["year"] + "-" + df["submission_id"]
        key = prefix + "-" + df["review_id"]
        colliding = key.duplicated(keep=False)

        bad_human_keys = set(key[colliding & df["class"].eq("human")])
        source_key = prefix + "-" + df["og_review_id"]
        orphan_rewrite = (
            df["class"].eq("rewrite")
            & df["og_review_id"].ne("")
            & source_key.isin(bad_human_keys)
        )

        drop = colliding | orphan_rewrite
        if n_strict or drop.any():
            dropped = df.loc[drop].assign(
                reason=np.where(colliding[drop], "collision", "rewrite_orphelin")
            )
            logger.warning(
                f"Identifiants : {n_strict} doublons stricts supprimés, "
                f"{int(drop.sum())} lignes exclues pour ambiguïté :\n"
                f"{dropped.groupby(['split', 'class', 'reason']).size()}"
            )
        return df.loc[~drop].copy()

    def _sample(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            raise ValueError("Aucune review ne matche les filtres (split/conference/date/generator/label_mode).")

        if self.balance == "paired":
            df = self._pair(df)

        elif self.balance == "uniform":
            counts = df["label"].value_counts().sort_index()
            n_classes = len(counts)
            n_per_class = int(counts.min())
            if self.subset_size is not None:
                requested = self.subset_size // n_classes
                if requested > n_per_class:
                    logger.warning(
                        f"subset_size={self.subset_size} demande {requested}/classe, "
                        f"mais la plus petite classe n'en a que {n_per_class} : plafonnement."
                    )
                n_per_class = min(n_per_class, requested)
            if n_per_class == 0:
                raise ValueError("Équilibrage uniforme impossible : au moins une classe est vide après sous-échantillonnage.")
            df = df.groupby("label").sample(n=n_per_class, random_state=self.seed)

        elif self.subset_size is not None and self.subset_size < len(df):
            # Stratification proportionnelle : conserve la distribution naturelle des labels
            frac = self.subset_size / len(df)
            df = df.groupby("label").sample(frac=frac, random_state=self.seed)

        return df.sample(frac=1.0, random_state=self.seed).reset_index(drop=True)

    def _pair(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Un groupe par review humaine pivot, un membre par classe réelle présente :
          - rewrite : une réécriture de CETTE review (source-review-id), modèle tiré de façon déterministe ;
          - ai      : une review IA du même papier et de la même note, sans remise.
        Les humaines sans partenaire complet sont écartées. Colonne `pair-id` = review-id du pivot.
        """
        present = set(df["class"])
        if "human" not in present:
            raise ValueError("balance=paired exige la classe human (pivot de l'appariement).")

        def ordered(frame: pd.DataFrame) -> pd.DataFrame:
            keys = frame["review-id"].map(lambda r: _stable_unit(r, self.seed))
            return frame.assign(_k=keys).sort_values(["_k", "review-id"]).drop(columns="_k")

        human = ordered(df[df["class"] == "human"])
        n_human0 = len(human)
        picks: Dict[str, Dict[str, str]] = {}

        # rewrite d'abord : une humaine sans réécriture ne doit pas consommer une review IA
        if "rewrite" in present:
            rw = ordered(df[df["class"] == "rewrite"])
            pools = {k: list(g["review-id"]) for k, g in rw.groupby("source-review-id", sort=False)}
            picks["rewrite"] = {h: pools[h][0] for h in human["review-id"] if h in pools}
            human = human[human["review-id"].isin(picks["rewrite"])]
        n_after_rw = len(human)

        if "ai" in present:
            ai = ordered(df[df["class"] == "ai"])
            pools = {k: list(g["review-id"]) for k, g in ai.groupby(PAIR_KEYS, sort=False)}
            ai_picks: Dict[str, str] = {}
            for h, key in zip(human["review-id"], human[PAIR_KEYS].itertuples(index=False, name=None)):
                pool = pools.get(key)
                if pool:
                    ai_picks[h] = pool.pop(0)
            picks["ai"] = ai_picks
            human = human[human["review-id"].isin(ai_picks)]
        n_after_ai = len(human)

        if self.subset_size is not None:
            n_groups = self.subset_size // len(present)
            if n_groups == 0:
                raise ValueError(f"subset_size={self.subset_size} < nombre de classes ({len(present)}).")
            if n_groups > len(human):
                logger.warning(f"subset_size demande {n_groups} groupes, seuls {len(human)} disponibles.")
            human = human.iloc[:n_groups]  # ordre déjà tiré de façon déterministe

        pivots = human["review-id"].tolist()
        parts = [human.assign(**{"pair-id": pivots})]
        for cls, cls_picks in picks.items():
            partner_to_pivot = {cls_picks[h]: h for h in pivots}
            part = df[(df["class"] == cls) & df["review-id"].isin(partner_to_pivot)].copy()
            part["pair-id"] = part["review-id"].map(partner_to_pivot)
            parts.append(part)
        out = pd.concat(parts, ignore_index=True)

        per_group = out.groupby(["pair-id", "class"]).size()
        if not (per_group == 1).all() or out.groupby("pair-id")["class"].nunique().ne(len(present)).any():
            raise AssertionError("appariement incohérent : chaque groupe doit avoir exactement un membre par classe")

        logger.info(
            f"Appariement : {len(pivots)} groupes | humaines initiales {n_human0}, "
            f"sans réécriture {n_human0 - n_after_rw}, sans IA même papier+note {n_after_rw - n_after_ai}\n"
            f"générateurs retenus :\n{out[out['class'] != 'human'].groupby(['class', 'generator']).size()}"
        )
        return out

    def _attach_papers(self, df: pd.DataFrame) -> pd.DataFrame:
        papers = self._load_config("papers")
        papers = papers[["submission_id", "parsed_pdf_content", "parsed_appendix_content"]].copy()
        for col in papers.columns:
            papers[col] = papers[col].fillna("").astype(str)
        papers = papers.drop_duplicates(subset="submission_id").rename(
            columns={"parsed_pdf_content": "paper_content", "parsed_appendix_content": "paper_appendix"}
        )

        df = df.merge(papers, on="submission_id", how="left", validate="many_to_one")
        n_missing = int(df["paper_content"].isna().sum())
        if n_missing:
            logger.warning(f"{n_missing} reviews sans papier correspondant (submission_id non trouvé).")
        df["paper_content"] = df["paper_content"].fillna("")
        df["paper_appendix"] = df["paper_appendix"].fillna("")
        return df

    def _log_stats(self, df: pd.DataFrame, n_raw: int) -> None:
        logger.info(
            f"Sem-Detect | split={self.splits} conf={self.conferences} years={self.years} "
            f"label_mode={self.label_mode} balance={self.balance} | {len(df)}/{n_raw} reviews"
        )
        label_3c_names = {v: k for k, v in CLASS_TO_3C.items()}
        crosstab = pd.crosstab(df["label_3c"].map(label_3c_names), df["label"])
        logger.info(f"Répartition class x label :\n{crosstab}")


if __name__ == "__main__":
    ds = SemDetectDataset(
        config={
            "split": "test",
            "conference": None,
            "date": None,
            "generator": None,
            "label_mode": "human_vs_ai",
            "balance": "none",
            "subset_size": None,
            "val_fraction": None,
            "val_seed": 0,
            "no_duplicate": True,
            "keep_meta": True,
            "with_paper": False,
            "keep_claims": False,
            "exclude_ids_path": None,
            "seed": 0,
        },
        no_cache=True,
    )
    print(ds)
    print(len(ds))
    print({k: (v[:80] if isinstance(v, str) else v) for k, v in ds[0].items()})