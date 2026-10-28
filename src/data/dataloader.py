#src/data/dataloader.py
import src.config as config

import torch
from torch.utils.data import DataLoader, Dataset

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def data_loader(config, dataset):
    return DataLoader(
        dataset,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        pin_memory=True if torch.cuda.is_available() else False
    )
