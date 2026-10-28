## file : linguistic_segmentation.py
import re
from pathlib import Path
from typing import Union, List

import numpy as np
import time


import hydra
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm
from src.config import DATA_DIR, RESULT_DIR

from hydra import initialize, compose
from omegaconf import DictConfig, OmegaConf
from src.data.dataset_aprdb import AprdbDataset
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, PreTrainedTokenizerBase, PreTrainedModel
import os


from src.detector.subjectivity_aware_generation_estimator.linguistic_utils import segment_text
from src.evaluation.eval_utils import generate_config_hash, append_to_jsonl, flatten_config
from ast import literal_eval

load_dotenv()

# Dossier des modèles : SAGE_MODELS_DIR, sinon <repo>/models (installé par scripts/fetch_sage_models.py)
MODELS_ROOT = Path(os.getenv("SAGE_MODELS_DIR") or Path(__file__).resolve().parents[4] / "models")
# GPU de la segmentation : SAGE_SEGMENTATION_DEVICE, sinon cuda:1 s'il y a deux GPU, cuda:0 sinon, cpu sans GPU
SEGMENTATION_DEVICE = os.getenv("SAGE_SEGMENTATION_DEVICE") or (
    "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0" if torch.cuda.is_available() else "cpu")
SUBJECTIVITY_MODEL_DIR = MODELS_ROOT / "roberta_subjectivity_regressor"    # 1-5 regression
SPEECH_ACT_MODEL_DIR = MODELS_ROOT / "roberta_speech_act_classifier"

SPEECH_ACT_COL = "rule_based_SA-label"
ASSERTION_LABEL = "Assertion"

SUBJECTIVITY_ID2LABEL = {
    1: "WORLD",
    2: "WORLD+VALUES",
    3: "VALUES",
    4: "TASTE+VALUES",
    5: "TASTE",
}


# --- Shared helper ------------------------------------------------------------

def load_sequence_classifier(
        model_dir: Path, device: torch.device
) -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = (
        AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
        .to(device)
        .eval()
    )
    return tokenizer, model

def explode_into_sentences(df: pd.DataFrame) -> pd.DataFrame:
    """One row per non-empty sentence; original columns are duplicated."""
    tqdm.pandas(desc="Segmenting reviews")
    df = df.copy()
    df["sentences"] = df["content"].progress_apply(segment_text)
    df = df.explode("sentences", ignore_index=True)
    df["sentences"] = df["sentences"].str.strip()
    # fillna: documents yielding an empty sentence list explode into NaN
    return df[df["sentences"].fillna("") != ""].reset_index(drop=True)

def label_speech_acts_rules(df: pd.DataFrame) -> pd.DataFrame:
    """version=1 -- rule-based classification."""
    df[SPEECH_ACT_COL] = df["sentences"].apply(classify)
    return df


def label_speech_acts_model(
        df: pd.DataFrame, device: torch.device, batch_size: int = 1024
) -> pd.DataFrame:
    """version=2 -- model-based classification (roberta_speech_act_classifier)."""
    tokenizer, model = load_sequence_classifier(SPEECH_ACT_MODEL_DIR, device)
    id2label = model.config.id2label

    sentences: list[str] = df["sentences"].tolist()
    preds: list[int] = []
    with torch.no_grad():
        for start in tqdm(
                range(0, len(sentences), batch_size), desc="Speech-act classification"
        ):
            batch = sentences[start: start + batch_size]
            inputs = tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=512
            ).to(device)
            preds.extend(model(**inputs).logits.argmax(dim=-1).cpu().tolist())

    df[SPEECH_ACT_COL] = [id2label[p] for p in preds]
    return df


def denormalize_to_1_5(raw: float) -> float:
    """Inverse of the (y - 1) / 4 target normalization applied at training time."""
    return raw * 4 + 1


def score_subjectivity(df: pd.DataFrame, device: torch.device, subjectivity_label) -> tuple[pd.DataFrame, int]:
    """Regression on Assertion rows only; predictions written back in place."""
    tokenizer, model = load_sequence_classifier(SUBJECTIVITY_MODEL_DIR, device)

    df["raw_prediction_1_5"] = None
    df["rounded_prediction"] = None
    df["predicted_label"] = None

    assertions = df.loc[df[SPEECH_ACT_COL].isin(subjectivity_label), "sentences"]

    for idx, text in tqdm(assertions.items(), total=len(assertions), desc="Subjectivity scoring"):
        inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)
        with torch.no_grad():
            raw = model(**inputs).logits.squeeze().item()

        score = denormalize_to_1_5(raw)
        rounded = min(5, max(1, round(score)))

        df.at[idx, "raw_prediction_1_5"] = score
        df.at[idx, "rounded_prediction"] = rounded
        df.at[idx, "predicted_label"] = SUBJECTIVITY_ID2LABEL[rounded]

    return df, len(assertions)


def score_subjectivity_optimized(df: pd.DataFrame, device: torch.device, subjectivity_label, batch_size: int = 512) -> tuple[
    pd.DataFrame, int]:
    """Regression on Assertion rows only; predictions written back in place via batching."""
    tokenizer, model = load_sequence_classifier(SUBJECTIVITY_MODEL_DIR, device)

    # Initialisation de la structure
    for col in ["raw_prediction_1_5", "rounded_prediction", "predicted_label"]:
        if col not in df.columns:
            df[col] = None

    # Création d'un masque booléen pour cibler directement la donnée
    mask = df[SPEECH_ACT_COL].isin(subjectivity_label)
    assertions = df.loc[mask, "sentences"].tolist()
    num_assertions = len(assertions)

    if num_assertions == 0:
        return df, 0

    raw_scores = []

    # Inférence par batchs
    for i in tqdm(range(0, num_assertions, batch_size), desc="Subjectivity scoring"):
        batch_texts = assertions[i: i + batch_size]

        # padding=True est indispensable pour batcher des séquences de tailles différentes
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to(device)

        with torch.no_grad():
            # .flatten() est plus sûr que .squeeze() pour éviter un crash si batch_size == 1
            logits = model(**inputs).logits.flatten()

        raw_scores.extend(logits.cpu().tolist())

    # Post-processing sous forme de list comprehension
    scores = [denormalize_to_1_5(r) for r in raw_scores]
    roundeds = [min(5, max(1, round(s))) for s in scores]
    labels = [SUBJECTIVITY_ID2LABEL[r] for r in roundeds]

    # Assignation en bloc (Bulk assignment)
    df.loc[mask, "raw_prediction_1_5"] = scores
    df.loc[mask, "rounded_prediction"] = roundeds
    df.loc[mask, "predicted_label"] = labels

    return df, num_assertions

def segment_dataframe_(
        df: pd.DataFrame, device: str = "cuda:0", version: int = 1, subj_category=None
) -> tuple[pd.DataFrame, int]:
    """Full pipeline. `version` selects the stage-2 strategy: 1=rules, 2=model."""
    if subj_category is None:
        subj_category = ['Assertion']
    dev = torch.device(device)

    df = explode_into_sentences(df)

    subjectivity_label = subj_category

    if version == 1:
        df = label_speech_acts_rules(df)
    elif version == 2:
        df = label_speech_acts_model(df, dev)
    else:
        raise ValueError(f"Unknown version {version!r} (expected 1 or 2)")

    return score_subjectivity_optimized(df, dev, subjectivity_label)


def segment_database_(dataset, segmentation: int = 1, subj_category=None) -> pd.DataFrame:
    if subj_category is None:
        subj_category = ['Assertion']
    feature_hash = generate_config_hash(
        {"segmentation": segmentation, "subj_category": sorted(subj_category)}
    )[:8]

    cfg = dataset.get_config()
    flat_config = flatten_config(OmegaConf.to_container(cfg))
    experiment_hash = dataset.get_hash()
    print(experiment_hash)

    local_dir = RESULT_DIR / "implicite_segmentation" / f"{experiment_hash}_{feature_hash}"


    local_dir.mkdir(parents=True, exist_ok=True)
    output_file_path = local_dir / "sentences_degrees_final.csv"

    if output_file_path.exists():
        print(f"📦 Loading data from cache: {output_file_path}")
        df = pd.read_csv(output_file_path)
        return df

    # NOTE: result caching intentionally disabled
    # (previously: early return if output_file_path exists).

    df, n_assertions = segment_dataframe_(dataset.to_pandas(), SEGMENTATION_DEVICE, segmentation, subj_category)

    df.to_csv(output_file_path, index=False, encoding="utf-8-sig")
    append_to_jsonl(
        {"hash": experiment_hash, **flat_config, "version": segmentation},
        filepath=RESULT_DIR / "implicite_segmentation" / "mapping_hash_config.json",
    )

    print(f"\n✅ Predictions saved to {output_file_path}")
    print(f"📊 Processed {n_assertions} Assertion sentences")
    return df



#--------------------------------------------------------------------------------
MAPPING = {
        1: "WORLD",
        2: "WORLD+VALUES",
        3: "VALUES",
        4: "TASTE+VALUES",
        5: "TASTE"
    }

MODEL_SEGMENTATION_PATH = str(MODELS_ROOT) + "/"


def segment_database(dataset, version=1):
    cfg = dataset.get_config()
    flat_config = OmegaConf.to_container(cfg)
    experiment_hash = dataset.get_hash()
    flat_config = flatten_config(flat_config)

    print(experiment_hash)
    local_dir = RESULT_DIR / f"implicite_segmentation" / f"{experiment_hash}{version}"
    local_dir.mkdir(parents=True, exist_ok=True)

    output_file_path = local_dir / f"sentences_degrees_final.csv"
 #   if output_file_path.exists():
 #       print(f"📦 Loading data from cache: {output_file_path}")
 #       df = pd.read_csv(output_file_path)
 #       return df

    df: pd.DataFrame = dataset.to_pandas()
    df, df_assertion_len = segment_dataframe(df, SEGMENTATION_DEVICE, version)



    df.to_csv(output_file_path, index=False, encoding="utf-8-sig")
    append_to_jsonl({"hash": experiment_hash, **flat_config, "version": version},
                    filepath=RESULT_DIR / f"implicite_segmentation" / "mapping_hash_config.json")

    print(f"\n✅ Predictions saved to {output_file_path}")
    print(f"📊 Processed {df_assertion_len} Assertion sentences")
    return df


def classify(sentence):
    question_words = r"^(why|what|how|when|where|who|which|is|are|do|does|did|can|could|would|should)\b"
    s = str(sentence).strip()

    if not s:
        return "Neither"

    s_low = s.lower()

    # --- Neither: formatting / metadata / fragments ---
    if len(s.split()) <= 2:
        return "Neither"

    if re.match(r"^(figure|table|section|chapter|source|url|http|www)\b", s_low):
        return "Neither"

    if s.startswith("[") and s.endswith("]"):
        return "Neither"

    if s_low in {"yes", "no", "ok", "okay"}:
        return "Neither"

    # --- Question detection ---
    if s.endswith("?"):
        return "Question"

    # wh-questions
    if re.match(question_words, s_low):
        return "Question"

    # auxiliary inversion (common in English questions)
    if re.match(r"^(is|are|do|does|did|can|could|would|should|will|won't|didn't)\b", s_low):
        return "Question"

    # --- Default ---
    return "Assertion"

def segmentation_df_v1_optimized(df, device, batch_size: int = 256, max_length: int = 512):
    model_name = "roberta_subjectivity_regressor"

    # --- Segmentation en phrases (inchangé, CPU-bound) ---
    df['sentences'] = df['content'].progress_apply(segment_text)
    df = df.explode('sentences', ignore_index=True)
    df['sentences'] = df['sentences'].str.strip()
    df = df[df['sentences'] != ''].reset_index(drop=True)

    df["rule_based_SA-label"] = df["sentences"].apply(classify)

    hf_token = os.getenv("HF_TOKEN")
    model_path = Path(MODEL_SEGMENTATION_PATH) / model_name
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=hf_token, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, token=hf_token, local_files_only=True,
        torch_dtype=torch.bfloat16,          # fp16 si GPU pré-Ampere
    ).to(device)
    model.eval()

    df['raw_prediction_1_5'] = None
    df['rounded_prediction'] = None
    df['predicted_label'] = None

    mask = df['rule_based_SA-label'] == "Assertion"
    texts = df.loc[mask, 'sentences'].tolist()
    orig_idx = df.index[mask].to_numpy()

    # --- Tri par longueur : réduit drastiquement le padding inutile ---
    order = np.argsort([len(t) for t in texts])
    texts_sorted = [texts[i] for i in order]

    preds = np.empty(len(texts_sorted), dtype=np.float32)

    with torch.inference_mode():
        for start in tqdm(range(0, len(texts_sorted), batch_size)):
            batch = texts_sorted[start:start + batch_size]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True,               # padding dynamique au max du batch
            ).to(device)
            logits = model(**inputs).logits.squeeze(-1)
            preds[start:start + len(batch)] = logits.float().cpu().numpy()

    # --- Restauration de l'ordre original + post-traitement vectorisé ---
    preds_unsorted = np.empty_like(preds)
    preds_unsorted[order] = preds

    pred_1_5 = preds_unsorted * 4 + 1
    rounded = np.clip(np.rint(pred_1_5), 1, 5).astype(int)

    df.loc[orig_idx, 'raw_prediction_1_5'] = pred_1_5
    df.loc[orig_idx, 'rounded_prediction'] = rounded
    df.loc[orig_idx, 'predicted_label'] = [MAPPING[r] for r in rounded]

    return df, mask.sum()

def segmentation_df_v1(df, device):
    model_name = f"roberta_subjectivity_regressor"
    print(f"segmenting text into sentences")
    df['sentences'] = df['content'].progress_apply(segment_text)
    # print(type(df['sentences_full'].iloc[0]))
    # df['sentences'] = df['sentences_full'].apply(literal_eval)

    df_expanded = df.explode('sentences', ignore_index=True)
    # The 'sentences' column now contains individual strings (not lists)
    df_expanded['sentences'] = df_expanded['sentences'].str.strip()
    # Drop any rows where sentence is empty
    df_expanded = df_expanded[df_expanded['sentences'] != '']
    # Save or inspect

    df = df_expanded
    df["rule_based_SA-label"] = df["sentences"].apply(classify)

    hf_token = os.getenv("HF_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained( Path(MODEL_SEGMENTATION_PATH) / model_name, token=hf_token,
                                              local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(Path(MODEL_SEGMENTATION_PATH) / model_name, token=hf_token,
                                                               local_files_only=True).to(
        device)

    if 'sentences' not in df.columns:
        raise ValueError("CSV file must contain a 'sentence' column")

    # --- NEW: ensure label column exists ---
    if 'rule_based_SA-label' not in df.columns:
        raise ValueError("CSV file must contain a 'label' column")

    # Add prediction columns
    df['raw_prediction_1_5'] = None
    df['rounded_prediction'] = None
    df['predicted_label'] = None

    # --- NEW: filter only Assertion rows ---
    df_assertion = df[df['rule_based_SA-label'] == "Assertion"].copy()

    for index, row in tqdm(df_assertion.iterrows(), total=len(df_assertion)):
        text = row['sentences']

        inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        prediction = outputs.logits.squeeze().item()

        # Convert back to 1–5 scale
        prediction_1_5 = prediction * 4 + 1

        rounded_prediction = round(prediction_1_5)
        rounded_prediction = max(1, min(5, rounded_prediction))

        mapped_label = MAPPING[rounded_prediction]

        # write back to original df using index
        df.at[index, 'raw_prediction_1_5'] = prediction_1_5
        df.at[index, 'rounded_prediction'] = rounded_prediction
        df.at[index, 'predicted_label'] = mapped_label
    return df, len(df_assertion)

def segmentation_df_v2(df, device):
    model_classification_name = f"roberta_speech_act_classifier"
    model_name = "roberta_subjectivity_regressor"
    model_classification_path = MODEL_SEGMENTATION_PATH / model_classification_name
    df['sentences'] = df['content'].progress_apply(segment_text)

    tokenizer = AutoTokenizer.from_pretrained(model_classification_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_classification_path)

    id2label = model.config.id2label
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"device for segmentation model : {device}")

    # --- Read data ---
    sentences = df["sentences"].fillna("").astype(str).tolist()

    BATCH_SIZE = 1024
    total_sentences = len(sentences)
    all_preds = []

    # Start timing
    start_time = time.time()

    with torch.no_grad():
        for i in range(0, total_sentences, BATCH_SIZE):
            batch = sentences[i:i + BATCH_SIZE]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=-1).cpu().tolist()
            all_preds.extend(preds)

            # Print progress every 20 batches (or at least once)
            if i % (BATCH_SIZE * 20) == 0 or i + BATCH_SIZE >= total_sentences:
                processed = min(i + BATCH_SIZE, total_sentences)
                elapsed = time.time() - start_time
                # Estimate remaining time
                if i > 0:  # avoid division by zero at first iteration
                    batches_done = i // BATCH_SIZE + 1
                    avg_time_per_batch = elapsed / batches_done
                    remaining_batches = (total_sentences - processed) / BATCH_SIZE
                    eta_seconds = avg_time_per_batch * remaining_batches
                    # Format ETA
                    if eta_seconds > 60:
                        eta_min = int(eta_seconds // 60)
                        eta_sec = int(eta_seconds % 60)
                        eta_str = f"{eta_min} min {eta_sec} sec"
                    else:
                        eta_str = f"{int(eta_seconds)} sec"
                else:
                    eta_str = "estimating..."

                print(f"Processed {processed}/{total_sentences} sentences, ETA: {eta_str}")

    # --- Save results ---
    df["rule_based_SA-label"] = [id2label[p] for p in all_preds]
    model_name = f"roberta_subjectivity_regressor"
    print(f"segmenting text into sentences")

    hf_token = os.getenv("HF_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(Path(MODEL_SEGMENTATION_PATH) / model_name, token=hf_token,
                                              local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(Path(MODEL_SEGMENTATION_PATH) / model_name,
                                                               token=hf_token,
                                                               local_files_only=True).to(
        device)

    if 'sentences' not in df.columns:
        raise ValueError("CSV file must contain a 'sentence' column")

    # --- NEW: ensure label column exists ---
    if 'rule_based_SA-label' not in df.columns:
        raise ValueError("CSV file must contain a 'label' column")

    # Add prediction columns
    df['raw_prediction_1_5'] = None
    df['rounded_prediction'] = None
    df['predicted_label'] = None

    # --- NEW: filter only Assertion rows ---
    df_assertion = df[df['rule_based_SA-label'] == "Assertion"].copy()

    for index, row in tqdm(df_assertion.iterrows(), total=len(df_assertion)):
        text = row['sentences']

        inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        prediction = outputs.logits.squeeze().item()

        # Convert back to 1–5 scale
        prediction_1_5 = prediction * 4 + 1

        rounded_prediction = round(prediction_1_5)
        rounded_prediction = max(1, min(5, rounded_prediction))

        mapped_label = MAPPING[rounded_prediction]

        # write back to original df using index
        df.at[index, 'raw_prediction_1_5'] = prediction_1_5
        df.at[index, 'rounded_prediction'] = rounded_prediction
        df.at[index, 'predicted_label'] = mapped_label
    return df, len(df_assertion)


def segment_dataframe(df, device = "cuda:0", version=1):
    tqdm.pandas(desc="Segmenting reviews")
    if version == 1:
        return segmentation_df_v1_optimized(df, device)
    elif version == 2:
        return segmentation_df_v2(df, device)


def segment_texts(texts: Union[str, List[str]], segmentation=1, subj_category=None) -> pd.DataFrame:
    if isinstance(texts, str):
        texts = [texts]

    df = pd.DataFrame({'content': texts})
    df['review-id'] = range(len(df))
    df, _ = segment_dataframe_(df, SEGMENTATION_DEVICE, version=segmentation, subj_category=subj_category)
    return df


# linguistic_segmentation.py — remplace segment_database_ et segment_texts

from src.detector.subjectivity_aware_generation_estimator.features.feature_spec import FeatureSpec


def _segmentation_dir(dataset_hash: str, spec: FeatureSpec) -> Path:
    """Clé de cache = identité du dataset + partie de la spec qui affecte la segmentation.
    subj_category en fait partie : deux specs v2 qui scorent des catégories différentes
    produisent des CSV différents et ne doivent pas partager de répertoire."""
    key = generate_config_hash(spec.segmentation_key())[:8]
    return RESULT_DIR / "implicite_segmentation" / f"{dataset_hash}_{key}"


def segment_database_spec(dataset, spec: FeatureSpec) -> pd.DataFrame:
    dataset_hash = dataset.get_hash()
    local_dir = _segmentation_dir(dataset_hash, spec)
    local_dir.mkdir(parents=True, exist_ok=True)
    output_file_path = local_dir / "sentences_degrees_final.csv"

    if output_file_path.exists():
        print(f"📦 Loading data from cache: {output_file_path}")
        return pd.read_csv(output_file_path)

    df, n_assertions = segment_dataframe_(
        dataset.to_pandas(), SEGMENTATION_DEVICE,
        version=spec.segmentation,
        subj_category=list(spec.subj_category),
    )

    df.to_csv(output_file_path, index=False, encoding="utf-8-sig")
    append_to_jsonl(
        {"dataset_hash": dataset_hash, "spec_name": spec.name, "spec_hash": spec.hash(),
         **flatten_config(OmegaConf.to_container(dataset.get_config())),
         **spec.segmentation_key()},
        filepath=RESULT_DIR / "implicite_segmentation" / "mapping_hash_config.json",
    )

    print(f"\n✅ Predictions saved to {output_file_path}")
    print(f"📊 Processed {n_assertions} scored sentences")
    return df


def segment_texts_spec(texts: Union[str, List[str]], spec: FeatureSpec) -> pd.DataFrame:
    """Chemin d'inférence. review-id = index dans `texts`, pour le réalignement aval."""
    if isinstance(texts, str):
        texts = [texts]
    df = pd.DataFrame({"content": texts})
    df["review-id"] = range(len(df))
    df, _ = segment_dataframe_(
        df, SEGMENTATION_DEVICE,
        version=spec.segmentation,
        subj_category=list(spec.subj_category),
    )
    return df