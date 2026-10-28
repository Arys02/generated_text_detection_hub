from abc import ABC
from typing import List, Union, Dict

import numpy as np
import torch
from torch import nn

from src.detector.base import BaseDetector

class RandomBaseDetector(nn.Module, BaseDetector):
    def predict(self, texts: Union[List[str] | str]) -> list[int]:
        return self.forward(texts)

    def __init__(self, config):
        super(RandomBaseDetector, self).__init__()
        self.config = config
        self.num_aa_classes = config.get("num_aa_classes", 2)

    def forward(self, texts: Union[List[str] | str]) -> list[int]:
        self.eval()
        self.to(self.config.get("device"))
        batch_size = len(texts)
        device = "cuda" if torch.cuda.is_available() else 'cpu'

        logits_tt = torch.randn(batch_size, 1).to(device)
        #logits_aa = torch.randn(batch_size, self.num_aa_classes).to(device)
        probs = torch.sigmoid(logits_tt).detach().cpu().numpy()
        #probs = np.where(probs>= 0.5, 1, 0)

        return probs.tolist()
