# src/data/dataset_factory.py
from src.data.dataset_aprdb import AprdbDataset
from src.data.dataset_semdetect import SemDetectDataset


def dataset_factory(config, **kwargs):
    """
    Instancie le dataset à partir de `config.name` (groupe Hydra `dataset`).
    kwargs : transmis tels quels au constructeur (use_tokenizer, cache_dir, no_cache, ...).
    """
    name = config["name"]

    if name == "aprdb":
        dataset = AprdbDataset(config=config, **kwargs)
    elif name == "sem_detect":
        dataset = SemDetectDataset(config=config, **kwargs)
    else:
        raise ValueError("Unknown dataset {}".format(name))
    return dataset