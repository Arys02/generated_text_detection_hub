#src/evaluation/run_eval.py
import json
from pathlib import Path

import hydra
import pandas as pd
from hydra import compose
from omegaconf import DictConfig, OmegaConf

import logging

import src.config
from src.config import RESULT_DIR
from src.data.dataloader import data_loader

from src.data.dataset_aprdb import AprdbDataset
from src.data.dataset_factory import dataset_factory
from src.data.dataset_openturingbench import get_data_loader
from src.detector.detector_factory import detector_factory
from src.evaluation.eval_engine import Evaluator
import torch
import os

from src.evaluation.eval_utils import generate_config_hash, flatten_config, append_to_jsonl

logger = logging.getLogger(__name__)


@hydra.main(config_path="../../conf", config_name="eval", version_base="1.3")
def main(cfg: DictConfig):
    logger.info(OmegaConf.to_yaml(cfg))
    resolved = OmegaConf.to_container(cfg, resolve=True)
    flat_config = flatten_config(resolved)
    experiment_hash = {"hash": generate_config_hash(resolved)}
    preset = cfg.get("train_preset")  # .get() sur DictConfig : safe même en struct mode

    train_config_hash = ""
    train_cfg_out = {}

    print("#################### Printing train config ####################")
    if preset is not None:
        train_cfg = compose(
            config_name="train",
            overrides=[f"experiment/train={preset}"],
        )
        print(OmegaConf.to_yaml(train_cfg))
        print("building config hash:")
        train_config_hash = generate_config_hash(train_cfg, exclude=["hardware"])
        cfg.detector.params.model_hash = train_config_hash
        # Résolution dans le contexte de la config d'entraînement AVANT la greffe :
        # une interpolation absolue (${dataset_train.x}) ne se résout plus une fois
        # déplacée sous detector.params.model_meta. Le hash ci-dessus est inchangé.
        train_cfg_resolved = OmegaConf.create(OmegaConf.to_container(train_cfg, resolve=True))
        cfg.detector.params.model_meta = train_cfg_resolved
        train_cfg_out = flatten_config(OmegaConf.to_container(train_cfg, resolve=True))
        experiment_hash['hash'] += train_config_hash
    print("#################### END Printing train config ####################")

    # TODO abstraire ça avec un dataset factory liked directement au fichiers de confs
    ds_test = dataset_factory(cfg.dataset)
    logger.info(f"Dataset '{cfg.dataset.name}' : {len(ds_test)} exemples (hash={ds_test.get_hash()})")

    dataloader = data_loader(cfg.data_loader, ds_test)

    model = detector_factory(cfg.detector)
    evaluator = Evaluator(
        model=model,
        dataloader=dataloader,
        device=cfg.hardware.device,
        type=cfg.detector['type']
    )
    # TODO tracker ce que le model prends en ram
    # TODO tracker le temps que le model prends
    # TODO tracker plus de donnée gpu
    metrics = evaluator.run()

    # TODO better architecture for that
    resolved = OmegaConf.to_container(cfg, resolve=True)
    flat_config = flatten_config(resolved)

    final_record = {**experiment_hash, **flat_config, **metrics, **train_cfg_out}

    append_to_jsonl(final_record, filepath=RESULT_DIR / "results.jsonl")

if __name__ == "__main__":
    main()
