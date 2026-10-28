# %%
from hydra import initialize, compose
from omegaconf import DictConfig
import os

from src.data.dataloader import data_loader
from src.data.dataset_aprdb import AprdbDataset

import src.config
from src.config import CACHE_DIR

with initialize(version_base=None, config_path="../../../conf"):
    cfg: DictConfig = compose(config_name="config",overrides=["+experiment/dataset/aprbd@dataset=aprdb_tt_test_all","experiment/data_loader=1_4_true"])

os.environ.setdefault('HF_HOME', os.path.expanduser("~/.cache/huggingface"))
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

print(cfg)
# %%
ds_test = AprdbDataset(config=cfg.dataset)
len(ds_test)
# %% [markdown]
# ### getting dataframe conversion
# %%
import pandas as pd
rows = []
for i in range(len(ds_test)):
    item = ds_test[i]
    rows.append(item)
df = pd.DataFrame(rows)
# %%
## loading human only
df_hwt_only = df[df['label'] == 0]
len(df_hwt_only)


from tqdm import tqdm
from src.detector.cristal_anchor.fetch_academic_paper import *
paper_source_paths = []
failed_review_ids = []
for idx, batch in tqdm(df_hwt_only.iterrows(), total=len(df_hwt_only)):
    batch_id = batch['review-id']
    review_path = fetch_pdf_from_review_id(batch_id)

    if review_path:
        print(batch_id)
        paper_source_paths.append((batch_id, review_path, batch['PaperId']))
    else:
        failed_review_ids.append((batch_id, review_path, batch['PaperId']))

# %%
import pickle

with open('data.pkl', 'wb') as f:
    pickle.dump({'ok': paper_source_paths, 'failed': failed_review_ids}, f)
# %%
print(len(paper_source_paths))
paper_source_paths
