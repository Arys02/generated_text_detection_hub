# src/detector/subjectivity_aware_generation_estimator/model/artifact_io.py
from typing import Any, Dict

import torch

from src.detector.subjectivity_aware_generation_estimator.model.SimpleGRU import SimpleGRU

_MODEL_CLASSES = {"SimpleGRU": SimpleGRU}


def load_model_from_artifact(artifact: Dict[str, Any], device: str) -> torch.nn.Module:
    """Reconstruit le modèle depuis le format produit par gru_train.train_gru."""
    missing = {"model_class", "model_kwargs", "state_dict"} - set(artifact)
    if missing:
        raise KeyError(
            f"Artefact au format inattendu, clés manquantes : {sorted(missing)}. "
            f"Clés présentes : {sorted(artifact)}"
        )
    cls_name = artifact["model_class"]
    if cls_name not in _MODEL_CLASSES:
        raise ValueError(f"model_class inconnue : {cls_name!r}. Connues : {sorted(_MODEL_CLASSES)}")

    model = _MODEL_CLASSES[cls_name](**artifact["model_kwargs"])
    model.load_state_dict(artifact["state_dict"])
    return model.to(device).eval()