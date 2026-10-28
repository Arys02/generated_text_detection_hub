# file: src/detector/subjectivity_aware_generation_estimator/sage.py

import os
from abc import ABC
from dataclasses import asdict
from pathlib import Path
from typing import Union

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from src.config import RESULT_DIR
from src.detector.base import BaseDetector
from src.detector.subjectivity_aware_generation_estimator.features.feature_spec import SPECS, FeatureSpec
from src.detector.subjectivity_aware_generation_estimator.features.tensor_builder import (
    REVIEW_ID, build_sequences,
)
from src.detector.subjectivity_aware_generation_estimator.model.artifact_io import load_model_from_artifact
from src.detector.subjectivity_aware_generation_estimator.training.gru_features import build_features_mono
from src.detector.subjectivity_aware_generation_estimator.training.linguistic_segmentation import segment_texts
from src.detector.subjectivity_aware_generation_estimator.training.mlp_train import df_to_feature_matrix

# colonnes d'identifiant acceptées en sortie de segment_texts, par ordre de priorité
_ID_CANDIDATES = (REVIEW_ID, "text_id", "id", "doc_id")
# clés ajoutées par FeatureSpec.to_dict() en plus des champs du dataclass
_SPEC_DERIVED_KEYS = ("name", "input_dim", "column_names", "spec_hash")


class ModelBase(ABC):
    def __init__(self):
        pass

    def evaluate(self, texts: Union[str, list[str]]):
        pass


class SageMLP(ModelBase):
    def __init__(self, hash: str):
        super().__init__()
        self.hash = hash
        self.source_model_path = Path(RESULT_DIR / f"{hash}")
        self.artifacts = joblib.load(str(self.source_model_path / "artifact.joblib"))
        self.pipe = self.artifacts['pipeline']

        assert self.artifacts["sklearn_version"] == sklearn.__version__, \
            f"version mismatch: {self.artifacts['sklearn_version']} vs {sklearn.__version__}"

    def evaluate(self, texts: Union[str, list[str]]):
        if isinstance(texts, str):
            texts = [texts]
        df_texts = segment_texts(texts)
        feature_matrix_texts, _ = df_to_feature_matrix(df_texts)
        return self.pipe.predict(feature_matrix_texts)


class SageGRU(ModelBase):
    """
    Trois générations d'artefacts :
      - legacy        : {'model', 'subj_mean', 'subj_std', ...}            -> chemin d'origine inchangé
      - intermédiaire : {'state_dict', 'feature_spec', 'ctx': 'ctx', ...}  -> ctx reconstruit depuis artifact['cfg']
      - courant       : {'state_dict', 'feature_spec', 'ctx': dict, ...}
    """

    def __init__(self, hash: str, cfg, device: str, save_segmented: bool = True):
        super().__init__()
        self.hash = hash
        self.cfg = cfg
        self.device = torch.device(device)
        self.save_segmented = save_segmented
        self.source_model_path = Path(RESULT_DIR / f"{hash}")
        print(f"Loading model from {self.source_model_path}...")
        self.artifacts = joblib.load(str(self.source_model_path / "artifact.joblib"))

        if "state_dict" in self.artifacts:
            self._init_spec_format()
        elif "model" in self.artifacts:
            self._init_legacy_format()
        else:
            raise KeyError(f"Format d'artefact inconnu, clés : {sorted(self.artifacts)}")

    # ------------------------------------------------------------ chargement
    def _init_legacy_format(self) -> None:
        self.model = self.artifacts['model']
        self.model.to(self.device)
        self.subj_mean = self.artifacts['subj_mean']
        self.subj_std = self.artifacts['subj_std']
        print("subj_mean", self.subj_mean, type(self.subj_mean))
        print("subj_std", self.subj_std, type(self.subj_std))
        self.format = "legacy"

    def _init_spec_format(self) -> None:
        art = self.artifacts
        spec_dict = art.get("feature_spec") or {}
        self.spec = self._resolve_spec(spec_dict)

        declared = self.cfg.training.model.get("feature_spec")
        if declared is not None and declared not in (self.spec.name, spec_dict.get("name")):
            raise ValueError(
                f"preset déclare feature_spec={declared!r}, artefact entraîné avec "
                f"{spec_dict.get('name')!r} (résolu en {self.spec.name!r})"
            )

        fv = self.cfg.training.model.get("feature_version")
        if fv is not None and (
                int(fv["segmentation"]) != self.spec.segmentation
                or sorted(fv["subj_category"]) != sorted(self.spec.subj_category)
        ):
            raise ValueError(
                f"feature_version du preset ({OmegaConf.to_container(fv)}) incohérente avec la spec "
                f"{self.spec.segmentation_key()}"
            )

        ctx = art.get("ctx")
        if not isinstance(ctx, dict):
            print(f"⚠️ ctx non sauvegardé dans l'artefact (type={type(ctx).__name__}) : "
                  f"reconstruction depuis la config d'entraînement stockée (artifact['cfg']).")
            ctx = self._rebuild_ctx()
        if tuple(ctx["label_cats"]) != tuple(self.spec.label_cats):
            raise ValueError(f"ctx.label_cats={ctx['label_cats']} != spec.label_cats={self.spec.label_cats}")
        self.ctx = ctx

        n_features = art["model_kwargs"].get("n_features")
        if n_features != self.spec.input_dim:
            raise ValueError(f"modèle n_features={n_features} != spec.input_dim={self.spec.input_dim}")
        self.model = load_model_from_artifact(art, str(self.device))

        self.format = "spec"
        print(f"spec={self.spec.name} (hash {self.spec.hash()}) | ctx train : "
              f"subj_mean={ctx['subj_mean']:.6f}, subj_std={ctx['subj_std']:.6f}")

    @staticmethod
    def _resolve_spec(spec_dict: dict) -> FeatureSpec:
        """Par nom si le hash concorde ; sinon par structure (spec renommée depuis l'entraînement)."""
        name, saved_hash = spec_dict.get("name"), spec_dict.get("spec_hash")
        if name in SPECS and SPECS[name].hash() == saved_hash:
            return SPECS[name]

        saved = {k: v for k, v in spec_dict.items() if k not in _SPEC_DERIVED_KEYS}
        matches = []
        for candidate in SPECS.values():
            cand = asdict(candidate)
            cand.pop("name")
            if cand == saved:
                matches.append(candidate)
        if len(matches) == 1:
            print(f"⚠️ spec {name!r} (hash {saved_hash}) absente ou modifiée dans le registre : "
                  f"résolue par structure en {matches[0].name!r}")
            return matches[0]
        raise ValueError(
            f"spec de l'artefact introuvable : name={name!r}, hash={saved_hash}, "
            f"{len(matches)} correspondance(s) structurelle(s) dans le registre {sorted(SPECS)}"
        )

    def _rebuild_ctx(self) -> dict:
        """Recalcule fit_context sur le train EXACT de l'artefact (config stockée, seed incluse).
        Imports locaux : chemin rare, évite de charger la chaîne d'entraînement à chaque inférence."""
        from src.data.dataset_factory import dataset_factory
        from src.detector.subjectivity_aware_generation_estimator.features.pairing import pair_and_balance
        from src.detector.subjectivity_aware_generation_estimator.features.tensor_builder import fit_context
        from src.detector.subjectivity_aware_generation_estimator.training.linguistic_segmentation import (
            segment_database_spec,
        )

        cfg_yaml = self.artifacts.get("cfg")
        if not isinstance(cfg_yaml, str) or not cfg_yaml.strip():
            raise ValueError("artifact['cfg'] absent : impossible de reconstruire ctx, ré-entraîner le modèle.")
        train_cfg = OmegaConf.create(cfg_yaml)

        df_train = pair_and_balance(
            segment_database_spec(dataset_factory(train_cfg.dataset_train), self.spec),
            seed=train_cfg.dataset_train.seed,
        )
        return fit_context(df_train, self.spec)

    # ------------------------------------------------------------- inférence
    @torch.no_grad()
    def evaluate(self, texts: Union[str, list[str]]):
        if isinstance(texts, str):
            texts = [texts]
        texts = list(texts)
        if self.format == "legacy":
            return self._evaluate_legacy(texts)
        return self._evaluate_spec(texts)

    def _feature_version(self):
        """feature_version du preset, transmise telle quelle à segment_texts (comme à l'origine)."""
        fv = self.cfg.training.model.get("feature_version")
        if fv is not None:
            return fv
        return {"segmentation": self.spec.segmentation, "subj_category": list(self.spec.subj_category)}

    def _segment(self, texts: list[str], feature_version) -> pd.DataFrame:
        df_texts = segment_texts(texts, **feature_version)
        ####### SAVING FOR MORE INFO ############
        if self.save_segmented:
            save_path = Path(RESULT_DIR / "evaluation_batch" / "evaluation_segmented_txt.jsonl")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            df_texts.to_json(save_path, orient="records", lines=True, force_ascii=False,
                             mode="a" if save_path.exists() else "w")
        #########################################
        return df_texts

    def _forward(self, X: torch.Tensor, L: torch.Tensor) -> np.ndarray:
        self.model.eval()
        dl = DataLoader(TensorDataset(X, L), batch_size=64, shuffle=False)
        ps = []
        for xb, lb in dl:
            logits = self.model(xb.to(self.device), lb)  # longueurs sur CPU (pack_padded_sequence)
            ps.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(ps).ravel()  # P(mgt) par review

    def _evaluate_legacy(self, texts: list[str]) -> np.ndarray:
        feature_version = self.cfg.training.model.feature_version
        df_texts = self._segment(texts, feature_version)
        Xte, Lte, ids = build_features_mono(df_texts, self.subj_mean, self.subj_std, order_col=None,
                                            feature_version=feature_version['segmentation'])
        return self._forward(Xte, Lte)

    def _evaluate_spec(self, texts: list[str]) -> np.ndarray:
        df = self._segment(texts, self._feature_version())

        id_col = next((c for c in _ID_CANDIDATES if c in df.columns), None)
        if id_col is None:
            raise KeyError(f"segment_texts ne renvoie aucune colonne d'id parmi {_ID_CANDIDATES} ; "
                           f"colonnes : {list(df.columns)}")
        # à l'inférence, un bloc = un texte : generator/label ne doivent pas scinder les blocs
        df = df.drop(columns=[c for c in ("generator", "label") if c in df.columns])
        if id_col != REVIEW_ID:
            df = df.rename(columns={id_col: REVIEW_ID})

        self.spec.validate(df)
        X, L, ids = build_sequences(df, self.spec, self.ctx, order_col=None)
        return self._align(self._forward(X, L), ids, len(texts))

    @staticmethod
    def _align(proba: np.ndarray, ids: np.ndarray, n_texts: int) -> np.ndarray:
        """Remet les scores dans l'ordre des textes d'entrée. Échoue plutôt que de décaler."""
        ids = np.asarray(ids)
        if ids.ndim == 2:  # tolère l'ancien _iter_blocks (1-tuples sous pandas >= 2)
            ids = ids[:, 0]
        try:
            pos = ids.astype(np.int64)
        except (TypeError, ValueError) as e:
            raise ValueError(f"ids de segment_texts non convertibles en positions : {ids[:5]}") from e
        if len(pos) != len(proba):
            raise ValueError(f"{len(pos)} ids pour {len(proba)} scores")
        if (pos < 0).any() or (pos >= n_texts).any() or len(np.unique(pos)) != len(pos):
            raise ValueError(f"ids hors de [0, {n_texts}) ou dupliqués : {pos[:10]}")

        out = np.full(n_texts, np.nan)
        out[pos] = proba
        missing = np.flatnonzero(np.isnan(out))
        if missing.size:
            raise ValueError(f"{missing.size} texte(s) sans segment (positions {missing[:10].tolist()}) : "
                             f"aucun score, alignement avec les labels impossible")
        return out


class Sage(BaseDetector):
    def __init__(self, config,
                 ):
        super(Sage, self).__init__(config)
        self.params = config['params']
        self.cache_dir = os.getenv('CACHE_DIR')
        self.model_meta = self.params['model_meta']
        self.model_hash = self.params['model_hash']
        self.model = self.build_model(self.model_hash, self.model_meta)

    def build_model(self, model_hash, model_meta) -> ModelBase:
        if model_meta.name == "mlp":
            return SageMLP(model_hash)
        if model_meta.name == "gru":
            return SageGRU(model_hash, model_meta, device=self.params['device'])

        raise ValueError('Unknown model{}'.format(model_meta.name))

    def predict(self, texts: Union[str, list[str]]):
        return self.model.evaluate(texts)