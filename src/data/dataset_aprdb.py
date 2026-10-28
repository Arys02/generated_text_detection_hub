#src/data/dataset_aprdb.py
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any

import torch
from huggingface_hub import list_repo_files, hf_hub_download
from datasets import load_dataset, concatenate_datasets
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from tqdm import tqdm

import src.config as config
import os

import logging

from src.detector.cristal_anchor.fetch_academic_paper import fetch_pdf_from_review_id_v2, get_forum_id_from_review_v2
from src.evaluation.eval_utils import generate_config_hash

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Subset = ["calibration", "test", "extended"]
Generator = ["gpt4o", "claude", "gemini", "llama", "qwen"]
Journal = {
    "ICLR": ["2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024"],
    "NeurIPS": ["2016", "2017", "2018", "2019", "2021", "2022", "2023", "2024"]
}

# Colonnes qu'on ne veut PAS mélanger dans le texte
DEFAULT_EXCLUDE_COLS = {
    "label",
    "custom_id", "PaperId", "review-id", "paper_id", "forum",
    "rating", "confidence", "overall score", "confidence score",
}


@dataclass
class FilterDataCleaning:
    no_duplicate: bool = False


class AprdbDataset(Dataset):
    def __init__(self,
                 config,
                 filter_data_cleaning: FilterDataCleaning = None,
                 use_tokenizer=None,
                 cache_dir: str = os.getenv("CACHE_DIR"),
                 no_cache: bool = False,
                 ):

        self.repo_id: str = "IntelLabs/AI-Peer-Review-Detection-Benchmark"

        self.config = config
        print(f"Loading APRDB dataset from {self.repo_id} with config: {config}")

        ## handling cache loading
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_id = generate_config_hash(
            OmegaConf.to_container(config, resolve=True) if not isinstance(config, dict) else dict(config)
    )
        self.cache_path = self.cache_dir / "dataset_cache" / f"dataset_{self.cache_id}.pt"



        ## handling pdf
        self.with_pdf = self.config['with_pdf']

        self.subset = config['split']
        self.conference = config['conference']
        self.tokenizer = use_tokenizer
        self.subset_size = config.get('subset_size', None)

        seed = self.config.get("seed")
        if self.subset_size is not None and not isinstance(seed, int):
            raise TypeError(f"dataset seed doit être un entier, reçu {seed!r} ({type(seed).__name__})")

        self.generator = config.get('generator', None)
        if config['date'] is None:
            self.date = ["2016", "2017", "2018", "2019", "2020", "2021"]
        else:
            self.date = config['date']
        self.keep_meta = config['keep_meta']

        if self.cache_path.exists() and not no_cache:
            print(f"📦 Loading dataset from cache: {self.cache_path}")
            self.data = torch.load(self.cache_path, weights_only=False)
        else:
            data = self._load_all_reviews()

            if filter_data_cleaning is None:
                filter_data_cleaning = FilterDataCleaning(no_duplicate=True)
            cleaned_data = self._data_cleaning(data, filter_data_cleaning)

            # if self.with_pdf:
            #     print("fetching pdfs")
            #     cleaned_data = self._attach_pdf_paths(cleaned_data)
            self.data = cleaned_data
            torch.save(self.data, self.cache_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def get_hash(self):
        return self.cache_id

    def get_config(self):
        return self.config

    def _attach_pdf_paths_local(self, ds):
        """
        Option A — résolution 100% locale, aucun appel API.
        Garde uniquement les reviews dont le review-id est dans le mapping cache
        ET dont le PDF {forum_id}.pdf existe sur disque.
        forum_id / pdf_path sont peuplés depuis le cache local.
        """
        cache_root = Path(os.getenv("CACHE_DIR"))
        mapping_path = cache_root / "review_paper_mapping.json"
        pdf_dir = cache_root / "papers_pdf"

        # 1. Index local, chargé une seule fois
        with mapping_path.open(encoding="utf-8") as f:
            review2forum: dict[str, str] = json.load(f)
        pdf_on_disk = {p.stem for p in pdf_dir.glob("*.pdf")}  # {forum_id}

        def resolve(review_id):
            fid = review2forum.get(review_id)
            if fid is None or fid not in pdf_on_disk:
                return None, None
            return fid, str(pdf_dir / f"{fid}.pdf")

        # 2. Filtrage : ne garder que les reviews entièrement résolues localement
        def keep_fn(ex):
            fid, _ = resolve(ex["review-id"])
            return fid is not None

        n_before = len(ds)
        ds = ds.filter(keep_fn)
        logger.info(f"Filtre local PDF : {len(ds)}/{n_before} reviews conservées (0 appel API)")

        # 3. Enrichissement forum_id / pdf_path depuis le cache
        def enrich(ex):
            fid, path = resolve(ex["review-id"])
            ex["forum_id"] = fid
            ex["pdf_path"] = path
            return ex

        ds = ds.map(enrich)

        return ds

    def _attach_pdf_paths(self, ds):
        """
        Enrichit ds avec :
          - PaperIdFull = f"{conference}-{year}-{PaperId}"
          - forum_id    = ID OpenReview du papier (None si non résolu)
          - pdf_path    = chemin local du PDF (None si non résolu)
        Résolution une seule fois par papier unique, puis broadcast.
        """
        df = ds.to_pandas()

        df["PaperIdFull"] = (
                df["conference"].astype(str) + "-"
                + df["year"].astype(str) + "-"
                + df["PaperId"].astype(str)
        )

        representatives = (
            df.dropna(subset=["review-id"])
            .groupby("PaperIdFull", as_index=False)
            .agg(review_id_repr=("review-id", "first"))
        )

        forum_ids, paths = [], []
        n_fail_forum, n_fail_pdf = 0, 0
        for rid in tqdm(representatives["review_id_repr"], total=len(representatives["review_id_repr"])):
            fid = get_forum_id_from_review_v2(rid)
            path = fetch_pdf_from_review_id_v2(rid) if fid else None
            forum_ids.append(fid)
            paths.append(path)
            n_fail_forum += (fid is None)
            n_fail_pdf += (path is None)

        representatives["forum_id"] = forum_ids
        representatives["pdf_path"] = paths

        logger.info(
            f"OpenReview resolution: forum_id {len(representatives) - n_fail_forum}/{len(representatives)} | "
            f"pdf {len(representatives) - n_fail_pdf}/{len(representatives)}"
        )

        df = df.merge(
            representatives[["PaperIdFull", "forum_id", "pdf_path"]],
            on="PaperIdFull",
            how="left",
        )

        from datasets import Dataset as HFDataset
        return HFDataset.from_pandas(df, preserve_index=False)

    @staticmethod
    def _list_csv_files(repo_id: str, subset: str) -> List[str]:
        files = list_repo_files(repo_id=repo_id, repo_type="dataset")
        prefix = f"{subset}/"
        csvs = [f for f in files if f.startswith(prefix) and f.endswith(".csv")]
        csvs.sort()
        if not csvs:
            raise ValueError(f"Aucun CSV trouvé sous {prefix}. Vérifie repo_id/subset.")
        return csvs

    @staticmethod
    def _parse_meta_from_path(path: str):
        # Ex: "test/gpt4o/ICLR2022.test.gpt-4o.csv"
        parts = path.split("/")
        generator = parts[1] if len(parts) >= 2 else None

        m = re.search(r"(ICLR|NeurIPS)(\d{4})", path)
        conference = m.group(1) if m else None
        year = m.group(2) if m else None

        return generator, conference, year

    @staticmethod
    def _data_cleaning(data, config):
        if config.no_duplicate:
            seen = set()
            keep_idx = []

            for i, x in enumerate(data["content"]):
                s = "" if x is None else str(x)
                if s in seen:
                    continue
                seen.add(s)
                keep_idx.append(i)
            data = data.select(keep_idx)
        return data

    def to_pandas(self) -> pd.DataFrame:
        rows = []
        for i in range(len(self)):
            item = self[i]
            rows.append(item)
        return pd.DataFrame(rows)

    def _load_all_reviews(self) -> Dataset:
        ## TODO save all reviews and automatic load when subset already exits instead of loading everything
        if isinstance(self.subset, list):
            subsets = []
            for subset in self.subset:
                subsets.append(self._list_csv_files(self.repo_id, subset))
            csv_files = [i for r in subsets for i in r]

        else:
            csv_files = self._list_csv_files(self.repo_id, self.subset)

        datasets = []

        for fp in csv_files:
            generator, conference, year = self._parse_meta_from_path(fp)
            if self.conference is not None and conference not in self.conference:
                continue
            if self.date is not None and year not in self.date:
                continue

            local_path = hf_hub_download(  # Hub cache, dédup par commit-hash, pas de re-DL si commit inchangé
                repo_id=self.repo_id,
                filename=fp,
                repo_type="dataset",
            )
            ds = load_dataset("csv", data_files={"data": local_path}, split="data")

            # ds = load_dataset(
            #     "csv",
            #     data_files={"data": f"hf://datasets/{self.repo_id}/{fp}"},
            #     split="data",
            # )

            def build_content(ex):
                txt = ex.get("review")
                if not (isinstance(txt, str) and txt.strip()):
                    txt = ex.get("main_review")
                if self.tokenizer is not None:
                    txt = self.tokenizer.encode(txt)
                ex["content"] = txt
                return ex

            ds = ds.map(build_content)

            keep = ["label", "content"]

            if getattr(self, "keep_meta", False):
                def add_meta(ex, _fp=fp, _g=generator, _c=conference, _y=year, _subset=self.subset):
                    ex["subset"] = _subset
                    ex["generator"] = _g
                    ex["conference"] = _c
                    ex["year"] = _y
                    ex["file"] = _fp
                    return ex

                ds = ds.map(add_meta)
                keep += ["subset", "generator", "conference", "PaperId", "year", "file", "review-id"]

            ds = ds.select_columns([c for c in keep if c in ds.column_names])
            datasets.append(ds)

        if not datasets:
            raise ValueError("Aucun CSV ne matche tes filtres (conference/date).")

        dataset_final = concatenate_datasets(datasets)

        if self.with_pdf:
            print("fetching pdfs")
            dataset_final = self._attach_pdf_paths_local(dataset_final)

        if self.subset_size is not None:
            ds_pos = dataset_final.filter(lambda x: x['label'] == 1)
            ds_neg = dataset_final.filter(lambda x: x['label'] == 0)

            # 2. Déterminer le nombre d'échantillons par classe
            # On prend la moitié de la taille demandée, ou le maximum dispo si une classe est trop petite
            half_size = self.subset_size // 2
            n_pos = min(half_size, len(ds_pos))
            n_neg = min(half_size, len(ds_neg))

            # 3. Sélectionner les sous-ensembles
            # On mélange d'abord pour ne pas prendre toujours les mêmes "premiers" exemples
            subset_pos = ds_pos.shuffle(seed=self.config['seed']).select(range(n_pos))
            subset_neg = ds_neg.shuffle(seed=self.config['seed']).select(range(n_neg))

            # 4. Fusionner et mélanger le résultat final
            dataset_final = concatenate_datasets([subset_pos, subset_neg]).shuffle(seed=self.config['seed'])

        return dataset_final


if __name__ == "__main__":
    dataset = AprdbDataset(config={"split": "test", "conference": None, "date": None, "keep_meta": True})

    print(dataset)
    print(len(dataset))
