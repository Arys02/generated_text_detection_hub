from typing import Dict, Any

import torch
from torch.utils.data import DataLoader, Dataset

from pathlib import Path
import pandas as pd
import src.config as config

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OpenTuringBenchDataset(Dataset):
    # split : in_domain(test | val | train') | 'mid_temperature' | 'high_temperature' | 'larger_models' | 'model_rewrite' | 'human_revise' | 'human_continue'
    # dataset : 'TT' or 'AA'
    def __init__(self,
                 split: str = 'test',
                 task: str = 'TT'):
        self.split = split
        self.task = task
        self.seed = 42
        self.with_human_data = False


        self.data = self._load_data()
        self.label_map = self._create_label_map()

    def _load_data(self):
        mgt_df = pd.read_json(config.HF_OTB_DATASET_LINK + config.SPLITS_OTB_ID[self.split])

        hwt_df = pd.DataFrame()

        if self.with_human_data:

            if self.task == 'TT':
                hwt_size = mgt_df.shape[0]
            else:
                hwt_size = mgt_df.shape[0] // mgt_df['model'].unique().size

            hwt_df = self._generate_hwt_dataset(hwt_size)
        data = pd.concat([hwt_df, mgt_df])
        return data

    def _generate_hwt_dataset(self, size) -> pd.DataFrame:
        ds = pd.read_parquet(config.DATA_HWT_PATH)
        ds = ds.sample(frac=1, random_state=self.seed).groupby('category').head(1000)
        ds = ds.sample(frac=1, random_state=self.seed).head(size)
        return ds

    def _create_label_map(self) -> Dict[str, int]:
        unique_models = sorted(self.data['model'].unique())
        return {name: idx for idx, name in enumerate(unique_models)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict[str, Any]:
        row = self.data.iloc[idx]
        content = row['content']
        model_name = row['model']
        label_tt = 0 if model_name == 'human' else 1
        label_tt_t = torch.tensor(label_tt, dtype=torch.long)
        label_aa_t = torch.tensor(self.label_map.get(model_name, -1), dtype=torch.long)


        return {
            'content': content,
            'model_name': model_name,
            'label_tt': label_tt_t,
            'label_aa': label_aa_t,
        }



def get_data_loader(config: Dict) -> DataLoader:
    dataset = OpenTuringBenchDataset(
        split=config['split'],
        task=config['dataset'],
    )

    return DataLoader(
        dataset,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        pin_memory=True if torch.cuda.is_available() else False
    )


if __name__ == "__main__":
    dummy_config = {
        'dataset': 'TT',
        'batch_size': 5,
        'split': 'test',
        'num_workers': 1,
    }

    loader = get_data_loader(config=dummy_config)
    batch = next(iter(loader))
    print(f"Batch keys: {batch.keys()}")
    print(f"Text sample: {batch['content'][0]}")
    print(f"TT Label (0=Human): {batch['label_tt'][0]}")

