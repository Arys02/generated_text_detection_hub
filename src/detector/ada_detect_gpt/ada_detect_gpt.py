import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Union, List, Dict

from scipy.stats import norm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM
import torch
from torch import nn
import scipy.interpolate as si

import src.config
from src.config import DATA_DIR
from src.detector.ada_detect_gpt.nuisance_func_human import BSplineTheory
from src.detector.ada_detect_gpt.nuisance_func import BSplineTwoSample
from src.detector.base import BaseDetector
import re
import numpy as np
import torch
import tqdm

from src.evaluation.eval_utils import generate_config_hash

huggingface_config = {
    # Only required for private models from Huggingface (e.g. LLaMA models)
    "TOKEN": os.environ.get("HF_TOKEN", None)
}

WFUNC_REGISTRY = {
    "BSplineTwoSample": BSplineTwoSample,
    "BSplineTheory":    BSplineTheory,
}


def get_classification_stat(logits_ref, logits_score, labels, w_func, shift_value=None):
    assert logits_ref.shape[0] == 1
    assert logits_score.shape[0] == 1
    assert labels.shape[0] == 1
    if logits_ref.size(-1) != logits_score.size(-1):
        # print(f"WARNING: vocabulary size mismatch {logits_ref.size(-1)} vs {logits_score.size(-1)}.")
        vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref = logits_ref[:, :, :vocab_size]
        logits_score = logits_score[:, :, :vocab_size]

    labels = labels.unsqueeze(-1) if labels.ndim == logits_score.ndim - 1 else labels
    #ORIGINAL
    # lprobs_score = w_func(torch.log_softmax(logits_score, dim=-1))
    # probs_ref = torch.softmax(logits_ref, dim=-1)
    # log_likelihood = lprobs_score.gather(dim=-1, index=labels).squeeze(-1)
    # mean_ref = (probs_ref * lprobs_score).sum(dim=-1)
    # var_ref = (probs_ref * torch.square(lprobs_score)).sum(dim=-1) - torch.square(mean_ref)
    #FIX
    lprobs_score = w_func(torch.log_softmax(logits_score.float(), dim=-1))
    probs_ref = torch.softmax(logits_ref.float(), dim=-1)
    log_likelihood = lprobs_score.gather(dim=-1, index=labels).squeeze(-1)
    mean_ref = (probs_ref * lprobs_score).sum(dim=-1)
    # forme stable : Var[X] = E[(X - E[X])^2]
    centered = lprobs_score - mean_ref.unsqueeze(-1)
    var_ref = (probs_ref * centered.square()).sum(dim=-1).clamp_min(0.)

    ########
    L = torch.tensor(var_ref.shape[0])
    stat = (log_likelihood.sum(dim=-1) - mean_ref.sum(dim=-1) - shift_value * L) / var_ref.sum(dim=-1).sqrt()
    stat = stat.mean()
    return stat.item()


class AdaDetectGPT(BaseDetector):
    def __init__(self, config,
                 scoring_model_name = "tiuuae/falcon-7b",
                 sampling_model_name = "tiuuae/falcon-7b"):
        super(AdaDetectGPT, self).__init__(config)
        self.config = config['params']
        self.cache_dir = os.getenv('CACHE_DIR')
        # Chargement unique en mémoire
        print("Initialisation des modèles AdaDetectGPT en VRAM...")

        print(f"Loading Scoring Tokenizer: {self.config['scoring_model_name']}")
        self.scoring_tokenizer = self.from_pretrained(AutoTokenizer, self.config['scoring_model_name'],
                                                      {"token": huggingface_config['TOKEN']})

        print(f"Loading Scoring Model: {self.config['scoring_model_name']}")
        self.scoring_model = self.from_pretrained(AutoModelForCausalLM, self.config['scoring_model_name'],
                                                  {"token": huggingface_config['TOKEN']}).to(
            self.config.device)

        self.scoring_model.eval()

        if self.scoring_tokenizer.pad_token is None:
            self.scoring_tokenizer.pad_token = self.scoring_tokenizer.eos_token
            self.scoring_tokenizer.pad_token_id = self.scoring_tokenizer.eos_token_id

        if self.config['sampling_model_name'] != self.config['scoring_model_name']:

            print(f"Loading Sampling Model: {self.config['sampling_model_name']}")
            self.sampling_model = self.from_pretrained(AutoModelForCausalLM, self.config['sampling_model_name'],
                                                       {"token": huggingface_config["TOKEN"]}).to(
                self.config.device)

            print(f"Loading Sampling Tokenizer: {self.config['sampling_model_name']}")
            self.sampling_tokenizer = self.from_pretrained(AutoTokenizer, self.config['sampling_model_name'],
                                                           {"token": huggingface_config["TOKEN"]})
            self.scoring_model.eval()
            self.same_scoring_sampling = False
        else:
            self.same_scoring_sampling = True
            self.sampling_model = self.scoring_model
            self.sampling_tokenizer = self.scoring_tokenizer

        self.shift_value = torch.zeros(1).to(self.config.device)

            ## TODO make it with factory for better check

        print("Loading Witness function")
        if self.config['w_func'] == 'identity':
            self.w_func = nn.Identity()
            self.beta = None
        else:
            bspline_config = {"start": -32, "end": 0, "n_bases": 7, "spline_order": 2, "intercept": 1}
            if self.config['w_func'] == 'pretrained':
                self.w_func = BSplineTwoSample(bspline_config, self.config.device)
                self.w_func.beta_hat = torch.tensor(
                    [0.0, -0.011333, -0.037667, -0.056667, -0.281667, -0.592, 0.157833, 0.727333, ]).to(
                    self.config.device)
            else:

                self.cache_id = generate_config_hash(
                    {"dataset": self.config['train_dataset'], "seed": self.config['seed'],
                     "w_function": self.config['w_func']})

                self.cache_path = Path(self.cache_dir) / f"ada_detect_gpt_wfunc_weight_{self.cache_id}.pth"

                if self.cache_path.exists():
                    ckpt = torch.load(self.cache_path, map_location=self.config.device, weights_only=True)
                    cls = WFUNC_REGISTRY[ckpt["class_name"]]
                    self.w_func = cls(ckpt["bspline_args"], self.config.device)
                    self.w_func.load_state_dict(ckpt["state_dict"])
                    self.w_func._fitted = True


                else:
                    if isinstance(self.config['train_dataset'], str):
                        self.config['train_dataset'] = [self.config['train_dataset']]
                    train_data = self.load_training_data(self.config['train_dataset'])

                    human_token_list = [
                        self.scoring_tokenizer(x, return_tensors="pt", padding=True,
                                               return_token_type_ids=False).to(
                            self.config.device) for x in train_data['original']]


                    if self.config['w_func'] == 'bspline':
                        machine_token_list = [
                            self.scoring_tokenizer(x, return_tensors="pt", padding=True, return_token_type_ids=False).to(
                                self.config.device) for x in train_data['sampled']]

                        self.w_func = BSplineTwoSample(bspline_config, self.config.device)
                        self.w_func.fit(
                            human_token_list,
                            machine_token_list,
                            self.scoring_model,
                            {"device": self.config.device})

                    elif self.config['w_func'] == 'bspline_theory':
                        self.w_func = BSplineTheory(bspline_config, machine_text=False)
                        self.w_func.fit(
                            human_token_list,
                            self.scoring_model,
                            {"device": self.config.device})
                    else:
                        raise ValueError

                    torch.save({
                        "class_name": type(self.w_func).__name__,
                        "bspline_args": bspline_config,
                        "state_dict": self.w_func.state_dict(),
                    }, self.cache_path)

            self.beta = self.w_func.beta_hat.detach().cpu().tolist()

        self.shift_value = torch.zeros(1).to(self.config.device)
        random.seed(self.config['seed'])
        torch.manual_seed(self.config['seed'])
        np.random.seed(self.config['seed'])

    def get_meta(self):
        return None

    @staticmethod
    def load_data(input_file):
        data_file = f"{DATA_DIR}/{input_file}.json"
        with open(data_file, "r") as fin:
            data = json.load(fin)
            print(f"Raw data loaded from {data_file}")
        return data

    @staticmethod
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

    @staticmethod
    def load_training_data(train_dataset_list):
        dataset_list = []
        for data_name in train_dataset_list:
            dataset = AdaDetectGPT.load_data(data_name)
            dataset_list.append(dataset)
        ## combine training data
        train_data = AdaDetectGPT.merge_dicts_of_lists(dataset_list)
        return train_data

    ## potentiel -> utils
    def from_pretrained(self, cls, model_name, kwargs):
        # use local model if it exists
        local_path = os.path.join(self.cache_dir, 'local.' + model_name.replace("/", "_"))
        if os.path.exists(local_path):
            return cls.from_pretrained(local_path, **kwargs)
        return cls.from_pretrained(model_name, **kwargs, cache_dir=self.cache_dir)

    def load_mask_tokenizer(self, model_name, max_length):
        tokenizer = self.from_pretrained(AutoTokenizer, model_name, {'model_max_length': max_length})
        return tokenizer

    def predict(self, texts: Union[List[str] | str]):
        if isinstance(texts, str):
            texts = [texts]

        device = self.config.device

        use_analytic = self.config.get('discrepancy_analytic', True)
        print(use_analytic)
        criterion_fn = get_classification_stat

        results = []

        for text in texts:
            tokenized = self.scoring_tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                return_token_type_ids=False,
                truncation=True,
            ).to(self.config['device'])

            labels = tokenized.input_ids[:, 1:]

            with torch.no_grad():
                logits_score = self.scoring_model(**tokenized).logits[:, :-1]
                if self.same_scoring_sampling:
                    logits_ref = logits_score
                else:
                    tokenized = self.sampling_tokenizer(
                        text,
                        return_tensors="pt",
                        padding=True,
                        return_token_type_ids=False,
                        truncation=True,
                    ).to(self.config['device'])
                    assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
                    logits_ref = self.sampling_model(**tokenized).logits[:, :-1]
                original_crit = criterion_fn(logits_ref, logits_score, labels, self.w_func, self.shift_value)
            prob = 1 - norm.cdf(original_crit)
            results.append(prob)

        return results
