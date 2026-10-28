import torch
import torch.nn as nn
from abc import ABC, abstractmethod

from typing import List, Dict, Any, Union


class BaseDetector(ABC):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config

    # @abstractmethod
    # def forward(self, x: List[str]) -> Dict[str, torch.Tensor]:
    #     pass


    @abstractmethod
    def predict(self, texts: Union[List[str] | str]):
        pass

    def get_meta(self):
        return None
