#evaluation/eval_utils.py
import hashlib
import json
import os
from copy import deepcopy
from typing import Dict, Any

from omegaconf import OmegaConf


def generate_config_hash(cfg, exclude=None) -> str:
    """Hash MD5 déterministe d'une config, certains champs exclus de l'identité."""
    # to_container + resolve : on hashe les VALEURS, pas la représentation objet.
    # default=str devient un vrai filet de secours, plus un masque.
    cfg_dict = OmegaConf.to_container(cfg, resolve=True) if not isinstance(cfg, dict) else deepcopy(cfg)
    if exclude is not None:
        _drop_paths(cfg_dict, exclude)  # cf. message précédent
    config_str = json.dumps(cfg_dict, sort_keys=True, default=str).encode("utf-8")
    return hashlib.md5(config_str).hexdigest()


def flatten_config(d: dict, parent_key: str = '', sep: str = '.') -> dict:
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_config(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _drop_paths(container: dict, dotted_paths: list[str]) -> None:
    """Supprime in-place les clés désignées par des chemins pointés (ex: 'training.model.output_dir')."""
    for path in dotted_paths:
        *parents, leaf = path.split(".")
        node = container
        for p in parents:
            if not isinstance(node, dict) or p not in node:
                node = None
                break
            node = node[p]
        if isinstance(node, dict):
            node.pop(leaf, None)

def append_to_jsonl(record: dict, filepath: str = "results.jsonl"):
    print(filepath)
    if not os.path.exists(filepath):
        with open(filepath, 'w', encoding='utf-8') as f:
            # json.dumps convertit le dictionnaire en une chaîne JSON sur une seule ligne
            f.write(json.dumps(record) + '\n')
    else:
        with open(filepath, 'a', encoding='utf-8') as f:
            # json.dumps convertit le dictionnaire en une chaîne JSON sur une seule ligne
            f.write(json.dumps(record) + '\n')