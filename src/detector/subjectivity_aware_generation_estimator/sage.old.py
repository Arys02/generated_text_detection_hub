#file sage.py

import os
from abc import ABC
from pathlib import Path
from typing import Union

import joblib
import sklearn
import torch
from setuptools.sandbox import save_path
from torch.utils.data import DataLoader, TensorDataset

from src.config import RESULT_DIR
from src.detector.base import BaseDetector
from src.detector.subjectivity_aware_generation_estimator.training.linguistic_segmentation import segment_texts
from src.detector.subjectivity_aware_generation_estimator.training.mlp_train import df_to_feature_matrix

from src.detector.subjectivity_aware_generation_estimator.training.gru_features import df_to_padded_batch, build_features_mono

import numpy as np


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
    def __init__(self, hash: str, cfg,  device:str):
        super().__init__()
        self.hash = hash
        self.cfg = cfg
        self.device = torch.device(device)
        self.source_model_path = Path(RESULT_DIR / f"{hash}")
        print(f"Loading model from {self.source_model_path}...")
        self.artifacts = joblib.load(str(self.source_model_path / "artifact.joblib"))
        self.model = self.artifacts['model']
        self.model.to(self.device)
        self.subj_mean = self.artifacts['subj_mean']
        self.subj_std = self.artifacts['subj_std']
        print("subj_mean", self.subj_mean, type(self.subj_mean))
        print("subj_std", self.subj_std, type(self.subj_std))

    @torch.no_grad()
    def evaluate(self, texts: Union[str, list[str]]):
        if isinstance(texts, str):
            texts = [texts]

        segmentation_version = self.cfg.training.model.feature_version['segmentation']

        df_texts = segment_texts(texts, **self.cfg.training.model.feature_version)
        ####### SAVING FOR MORE INFO ############
        save_path = RESULT_DIR / "evaluation_batch" / f"evaluation_segmented_txt.jsonl"
        save_path = Path(save_path)
        df_texts.to_json(save_path, orient="records", lines=True, force_ascii=False,
                                 mode="a" if save_path.exists() else "w")

        #########################################

        Xte, Lte, ids = build_features_mono(df_texts, self.subj_mean, self.subj_std, order_col=None, feature_version=segmentation_version)

        test_dl = DataLoader(TensorDataset(Xte, Lte), batch_size=64, shuffle=False)

        self.model.eval()
        ps = []
        for xb, lb in test_dl:
            logits = self.model(xb.to(self.device), lb)
            ps.append(torch.sigmoid(logits).cpu().numpy())

        proba = np.concatenate(ps).ravel()  # P(mgt) par review, aligné sur ids
        return proba



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
