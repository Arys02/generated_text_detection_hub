from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
import os

load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

# data
DATA_DIR = Path(os.getenv("DATA_DIR"))
DATA_OTB = DATA_DIR / "kaggle_dataset" / 'news_category_dataset_cleaned.parquet'

HYDRA_CONFIG_DIR = PROJ_ROOT / "conf"
DATA_FOLDER = PROJ_ROOT / "data"
# cache
CACHE_DIR = Path(os.getenv("CACHE_DIR"))

# src
SRC_DIR = PROJ_ROOT / "src"


# results
RESULT_DIR = CACHE_DIR / "results"


# Huggingface Dataset
os.environ['HF_HOME'] = os.getenv("HF_HOME")

HF_OTB_DATASET_LINK = "hf://datasets/MLNTeam-Unical/OpenTuringBench/"
SPLITS_OTB_ID  = {'train': 'in-domain/train.json', 'val': 'in-domain/val.json', 'test': 'in-domain/test.json'}
SPLITS_OTB_ID_var = {'mid_temperature': 'in-domain-variations/mid_temperature.json', 'high_temperature': 'in-domain-variations/high_temperature.json', 'larger_models': 'in-domain-variations/larger_models.json', 'model_rewrite': 'in-domain-variations/model_rewrite.json', 'human_revise': 'in-domain-variations/human_revise.json', 'human_continue': 'in-domain-variations/human_continue.json'}

# finetuning
DATA_FINETUNING = DATA_DIR / "train_human_style.jsonl"


