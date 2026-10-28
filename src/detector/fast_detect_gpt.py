import json
import os
from typing import Union, List, Dict

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM
import torch

import src.config
from src.detector.base import BaseDetector
import re
import numpy as np
import torch
import tqdm



class FastDetectGPT(BaseDetector):
    def __init__(self, config):
        super(FastDetectGPT, self).__init__(config)
        self.config = config['params']
        self.cache_dir = os.getenv('CACHE_DIR')
        # Chargement unique en mémoire
        print("Initialisation des modèles FastDetectGPT en VRAM...")

        self.scoring_model = self.from_pretrained(AutoModelForCausalLM, self.config['scoring_model_name'], {}).to(self.config.device)
        self.scoring_tokenizer = self.from_pretrained(AutoTokenizer, self.config['scoring_model_name'], {})
        self.scoring_model.eval()

        if self.scoring_tokenizer.pad_token is None:
            self.scoring_tokenizer.pad_token = self.scoring_tokenizer.eos_token
            self.scoring_tokenizer.pad_token_id = self.scoring_tokenizer.eos_token_id

        self.sampling_model= self.from_pretrained(AutoModelForCausalLM, self.config['sampling_model_name'], {}).to(
            self.config.device)
        self.sampling_tokenizer= self.from_pretrained(AutoTokenizer, self.config['sampling_model_name'], {})
        self.scoring_model.eval()

        if self.sampling_tokenizer is not None and self.sampling_tokenizer.pad_token is None:
            self.sampling_tokenizer.pad_token = self.sampling_tokenizer.eos_token
            self.sampling_tokenizer.pad_token_id = self.sampling_tokenizer.eos_token_id


        self.use_analytic = self.config['discrepancy_analytic']

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

    def get_meta(self):
        return None

    def save_data(self, output_file, args, data):
        # write args to file
        args_file = f"{output_file}.args.json"
        with open(args_file, "w") as fout:
            json.dump(args.__dict__, fout, indent=4)
            print(f"Args written into {args_file}")

        # write the data to a json file in the save folder
        data_file = f"{output_file}.raw_data.json"
        with open(data_file, "w") as fout:
            json.dump(data, fout, indent=4)
            print(f"Raw data written into {data_file}")

    def load_data(self, input_file):
        data_file = f"{input_file}.raw_data.json"
        with open(data_file, "r") as fin:
            data = json.load(fin)
            print(f"Raw data loaded from {data_file}")
        return data

    def get_samples(self, logits, labels):
        assert logits.shape[0] == 1
        assert labels.shape[0] == 1
        nsamples = 10000
        lprobs = torch.log_softmax(logits, dim=-1)
        distrib = torch.distributions.categorical.Categorical(logits=lprobs)
        samples = distrib.sample([nsamples]).permute([1, 2, 0])
        return samples

    def get_likelihood(self, logits, labels):
        assert logits.shape[0] == 1
        assert labels.shape[0] == 1
        labels = labels.unsqueeze(-1) if labels.ndim == logits.ndim - 1 else labels
        lprobs = torch.log_softmax(logits, dim=-1)
        log_likelihood = lprobs.gather(dim=-1, index=labels)
        return log_likelihood.mean(dim=1)

    def get_sampling_discrepancy(self, logits_ref, logits_score, labels):
        assert logits_ref.shape[0] == 1
        assert logits_score.shape[0] == 1
        assert labels.shape[0] == 1
        if logits_ref.size(-1) != logits_score.size(-1):
            # print(f"WARNING: vocabulary size mismatch {logits_ref.size(-1)} vs {logits_score.size(-1)}.")
            vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
            logits_ref = logits_ref[:, :, :vocab_size]
            logits_score = logits_score[:, :, :vocab_size]

        samples = self.get_samples(logits_ref, labels)
        log_likelihood_x = self.get_likelihood(logits_score, labels)
        log_likelihood_x_tilde = self.get_likelihood(logits_score, samples)
        miu_tilde = log_likelihood_x_tilde.mean(dim=-1)
        sigma_tilde = log_likelihood_x_tilde.std(dim=-1)
        discrepancy = (log_likelihood_x.squeeze(-1) - miu_tilde) / sigma_tilde

        discrepancy = torch.nan_to_num(discrepancy, nan=-1e4)

        return discrepancy.item()

    def get_sampling_discrepancy_analytic(self, logits_ref, logits_score, labels):
        assert logits_ref.shape[0] == 1
        assert logits_score.shape[0] == 1
        assert labels.shape[0] == 1
        if logits_ref.size(-1) != logits_score.size(-1):
            # print(f"WARNING: vocabulary size mismatch {logits_ref.size(-1)} vs {logits_score.size(-1)}.")
            vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
            logits_ref = logits_ref[:, :, :vocab_size]
            logits_score = logits_score[:, :, :vocab_size]

        labels = labels.unsqueeze(-1) if labels.ndim == logits_score.ndim - 1 else labels
        lprobs_score = torch.log_softmax(logits_score, dim=-1)
        probs_ref = torch.softmax(logits_ref, dim=-1)
        log_likelihood = lprobs_score.gather(dim=-1, index=labels).squeeze(-1)
        mean_ref = (probs_ref * lprobs_score).sum(dim=-1)
        var_ref = (probs_ref * torch.square(lprobs_score)).sum(dim=-1) - torch.square(mean_ref)
        discrepancy = (log_likelihood.sum(dim=-1) - mean_ref.sum(dim=-1)) / var_ref.sum(dim=-1).sqrt()
        discrepancy = discrepancy.mean()

        discrepancy = torch.nan_to_num(discrepancy, nan=-1e4)

        return discrepancy.item()

    def predict(self, texts: Union[List[str] | str]):
        if isinstance(texts, str):
            texts = [texts]

        use_analytic = self.config.get('discrepancy_analytic', True)
        print(use_analytic)
        criterion_fn = (
            self.get_sampling_discrepancy_analytic
            if use_analytic
            else self.get_sampling_discrepancy
        )

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
                # FastDetectGPT en mode self-detection (ref == score)
                # si tu charges un sampling_model séparé, substitue ici
                logits_ref = logits_score

                discrepancy = criterion_fn(logits_ref, logits_score, labels)

            prob = torch.sigmoid(torch.tensor(discrepancy)).item()
            results.append(prob)

        return results
