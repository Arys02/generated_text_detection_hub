import json
import os
from abc import ABC
from typing import Union, List, Dict

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM
import torch

import src.config
from src.detector.base import BaseDetector
import re
import numpy as np
import torch
import tqdm

# define regex to match all <extra_id_*> tokens, where * is an integer
pattern = re.compile(r"<extra_id_\d+>")


class DetectGPT(BaseDetector):
    def __init__(self, config):
        super(DetectGPT, self).__init__(config)
        self.config = config['params']
        self.cache_dir = os.getenv('CACHE_DIR')
        # Chargement unique en mémoire
        print("Initialisation des modèles DetectGPT en VRAM...")
        self.args = type('Args', (), dict(self.config))()
        if not hasattr(self.args, 'device'):
            self.args.device = self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

        self.mask_model = self.load_mask_model(self.config['mask_filling_model_name'], self.args.device)
        self.mask_tokenizer = self.load_mask_tokenizer(self.config['mask_filling_model_name'], 512)

        self.scoring_model = self.from_pretrained(AutoModelForCausalLM, self.config['scoring_model_name'], {}).to(
            self.args.device)
        self.scoring_tokenizer = self.from_pretrained(AutoTokenizer, self.config['scoring_model_name'], {})
        self.scoring_model.eval()
        self.mask_model.eval()

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

    def tokenize_and_mask(self, text, span_length, pct, ceil_pct=False):
        buffer_size = 1
        tokens = text.split(' ')
        mask_string = '<<<mask>>>'

        n_spans = pct * len(tokens) / (span_length + buffer_size * 2)
        if ceil_pct:
            n_spans = np.ceil(n_spans)
        n_spans = int(n_spans)

        n_masks = 0
        while n_masks < n_spans:
            start = np.random.randint(0, len(tokens) - span_length)
            end = start + span_length
            search_start = max(0, start - buffer_size)
            search_end = min(len(tokens), end + buffer_size)
            if mask_string not in tokens[search_start:search_end]:
                tokens[start:end] = [mask_string]
                n_masks += 1

        # replace each occurrence of mask_string with <extra_id_NUM>, where NUM increments
        num_filled = 0
        for idx, token in enumerate(tokens):
            if token == mask_string:
                tokens[idx] = f'<extra_id_{num_filled}>'
                num_filled += 1
        assert num_filled == n_masks, f"num_filled {num_filled} != n_masks {n_masks}"
        text = ' '.join(tokens)
        return text

    def get_meta(self):
        return None

    def count_masks(self, texts):
        return [len([x for x in text.split() if x.startswith("<extra_id_")]) for text in texts]

    def replace_masks(self, args, mask_model, mask_tokenizer, texts):
        n_expected = self.count_masks(texts)
        stop_id = mask_tokenizer.encode(f"<extra_id_{max(n_expected)}>")[0]
        tokens = mask_tokenizer(texts, return_tensors="pt", padding=True).to(args.device)
        outputs = mask_model.generate(**tokens, max_length=150, max_new_tokens=None,
                                      do_sample=True, top_p=args.mask_top_p,
                                      num_return_sequences=1, eos_token_id=stop_id)
        return mask_tokenizer.batch_decode(outputs, skip_special_tokens=False)

    def extract_fills(self, texts):
        # remove <pad> from beginning of each text
        texts = [x.replace("<pad>", "").replace("</s>", "").strip() for x in texts]

        # return the text in between each matched mask token
        extracted_fills = [pattern.split(x)[1:-1] for x in texts]

        # remove whitespace around each fill
        extracted_fills = [[y.strip() for y in x] for x in extracted_fills]

        return extracted_fills

    def apply_extracted_fills(self, masked_texts, extracted_fills):
        # split masked text into tokens, only splitting on spaces (not newlines)
        tokens = [x.split(' ') for x in masked_texts]

        n_expected = self.count_masks(masked_texts)

        # replace each mask token with the corresponding fill
        for idx, (text, fills, n) in enumerate(zip(tokens, extracted_fills, n_expected)):
            if len(fills) < n:
                tokens[idx] = []
            else:
                for fill_idx in range(n):
                    text[text.index(f"<extra_id_{fill_idx}>")] = fills[fill_idx]

        # join tokens back into text
        texts = [" ".join(x) for x in tokens]
        return texts

    def perturb_texts_(self, args, mask_model, mask_tokenizer, texts, ceil_pct=False):
        span_length = args.span_length
        pct = args.pct_words_masked
        masked_texts = [self.tokenize_and_mask(x, span_length, pct, ceil_pct) for x in texts]
        raw_fills = self.replace_masks(args, mask_model, mask_tokenizer, masked_texts)
        extracted_fills = self.extract_fills(raw_fills)
        perturbed_texts = self.apply_extracted_fills(masked_texts, extracted_fills)

        # Handle the fact that sometimes the model doesn't generate the right number of fills and we have to try again
        attempts = 1
        while '' in perturbed_texts:
            idxs = [idx for idx, x in enumerate(perturbed_texts) if x == '']
            print(f'WARNING: {len(idxs)} texts have no fills. Trying again [attempt {attempts}].')
            masked_texts = [self.tokenize_and_mask(x, span_length, pct, ceil_pct) for idx, x in enumerate(texts) if
                            idx in idxs]
            raw_fills = self.replace_masks(args, mask_model, mask_tokenizer, masked_texts)
            extracted_fills = self.extract_fills(raw_fills)
            new_perturbed_texts = self.apply_extracted_fills(masked_texts, extracted_fills)
            for idx, x in zip(idxs, new_perturbed_texts):
                perturbed_texts[idx] = x
            attempts += 1
        return perturbed_texts

    def load_mask_model(self, model_name, device):
        # mask filling t5 model
        print(f'Loading mask filling model {model_name}...')
        mask_model = self.from_pretrained(AutoModelForSeq2SeqLM, model_name, {})
        mask_model = mask_model.to(device)
        return mask_model

    def perturb_texts(self, args, mask_model, mask_tokenizer, texts, ceil_pct=False):
        chunk_size = 10
        outputs = []
        for i in range(0, len(texts), chunk_size):
            outputs.extend(
                self.perturb_texts_(args, mask_model, mask_tokenizer, texts[i:i + chunk_size], ceil_pct=ceil_pct))
        return outputs

    # Get the log likelihood of each text under the base_model
    def get_ll(self, args, scoring_model, scoring_tokenizer, text):
        with torch.no_grad():
            tokenized = scoring_tokenizer(text, return_tensors="pt", return_token_type_ids=False).to(args.device)
            labels = tokenized.input_ids
            return -scoring_model(**tokenized, labels=labels).loss.item()

    def get_lls(self, args, scoring_model, scoring_tokenizer, texts):
        return [self.get_ll(args, scoring_model, scoring_tokenizer, text) for text in texts]

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

    def generate_perturbs(self, args):
        n_perturbations = args.n_perturbations
        name = f'perturbation_{n_perturbations}'
        # load model
        mask_model = self.load_mask_model(args.mask_filling_model_name, args.device)
        mask_model.eval()
        try:
            n_positions = mask_model.config.n_positions
        except AttributeError:
            n_positions = 512
        mask_tokenizer = self.load_mask_tokenizer(args.mask_filling_model_name, n_positions)

        # load data
        data = self.load_data(args.dataset_file)
        n_samples = len(data["sampled"])

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        # generate perturb samples
        perturbs = []
        for idx in tqdm.tqdm(range(n_samples), desc=f"Perturb text"):
            original_text = data["original"][idx]
            sampled_text = data["sampled"][idx]
            # perturb
            p_sampled_text = self.perturb_texts(args, mask_model, mask_tokenizer,
                                                [sampled_text for _ in range(n_perturbations)])
            p_original_text = self.perturb_texts(args, mask_model, mask_tokenizer,
                                                 [original_text for _ in range(n_perturbations)])
            assert len(
                p_sampled_text) == n_perturbations, f"Expected {n_perturbations} perturbed samples, got {len(p_sampled_text)}"
            assert len(
                p_original_text) == n_perturbations, f"Expected {n_perturbations} perturbed samples, got {len(p_original_text)}"
            # result
            perturbs.append({
                "original": original_text,
                "sampled": sampled_text,
                "perturbed_sampled": p_sampled_text,
                "perturbed_original": p_original_text
            })

        self.save_data(f'{args.dataset_file}.{args.mask_filling_model_name}.{name}', args, perturbs)

    def predict(self, texts: Union[List[str] | str]):
        if isinstance(texts, str):
            texts = [texts]

        max_words = 300
        truncated_texts = [" ".join(text.split()[:max_words]) for text in texts]

        probs = []
        for text in truncated_texts:
            # 1. Générer les perturbations
            perturbed_texts = self.perturb_texts(self.args, self.mask_model, self.mask_tokenizer,
                                                 [text] * self.config.get('n_perturbations', 10))

            # 2. Scorer l'original et les perturbations
            ll_orig = self.get_ll(self.args, self.scoring_model, self.scoring_tokenizer, text)
            lls_pert = self.get_lls(self.args, self.scoring_model, self.scoring_tokenizer, perturbed_texts)

            # 3. Calcul de la métrique (Z-score)
            mu_pert = np.mean(lls_pert)
            std_pert = np.std(lls_pert) if np.std(lls_pert) > 0 else 1e-8
            z_score = (ll_orig - mu_pert) / std_pert

            # 4. Conversion en probabilité (Non-calibrée)
            prob = 1 / (1 + np.exp(-z_score))
            probs.append(float(prob))

        return probs
