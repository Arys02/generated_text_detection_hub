import json
import os
from collections import defaultdict

from src.config import DATA_DIR

huggingface_config = {
    # Only required for private models from Huggingface (e.g. LLaMA models)
    "TOKEN": os.environ.get("HF_TOKEN", None)
}


def from_pretrained(cache_dir, cls, model_name, kwargs):
    # use local model if it exists
    local_path = os.path.join(cache_dir, 'local.' + model_name.replace("/", "_"))
    if os.path.exists(local_path):
        return cls.from_pretrained(local_path, **kwargs)
    return cls.from_pretrained(model_name, **kwargs, cache_dir=cache_dir)


def load_data(input_file):
    data_file = f"{DATA_DIR}/{input_file}.json"
    with open(data_file, "r") as fin:
        data = json.load(fin)
        print(f"Raw data loaded from {data_file}")
    return data


def merge_dicts_of_lists(dataset_list) -> dict:
    """
    将一系列 dict(键→list) 合并为一个 dict，
    同一个键对应的 list 会被 extend 到一起。
    """
    merged = defaultdict(list)
    for d in dataset_list:
        for key, value in d.items():
            # 如果 value 本身是 list，则 extend；否则 append
            if isinstance(value, list):
                merged[key].extend(value)
            else:
                merged[key].append(value)
    return dict(merged)


def load_training_data(train_dataset_list):
    dataset_list = []
    for data_name in train_dataset_list:
        dataset = load_data(data_name)
        dataset_list.append(dataset)
    ## combine training data
    train_data = merge_dicts_of_lists(dataset_list)
    return train_data
