import json
import os
from abc import ABC, abstractmethod
from pathlib import Path

from openai import OpenAI
from sentence_transformers import SentenceTransformer
import numpy as np



def embbeding_model_factory(model_name):
    if model_name == 'all-MiniLM-L6-v2':
        return SentenceTransformerWrapper('all-MiniLM-L6-v2')
    elif model_name == 'text-embedding-3-small':
        return OpenAiEmbeddingModel("text-embedding-3-small")
    else:
        raise NotImplementedError(f"Model {model_name} not implemented")

class EmbeddingModel(ABC):
    @abstractmethod
    def compare(self, val1, val2):
        pass

    @abstractmethod
    def encode(self, val1):
        pass

    @abstractmethod
    def decode(self, val1):
        pass

class OpenAiEmbeddingModel(EmbeddingModel):
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        dimensions: int | None = None,
        api_key_env: str = "OPEN_AI_API_KEY",
        cache_dir: str = os.getenv("CACHE_DIR"),
    ):
        self._client = OpenAI(api_key=os.environ[api_key_env])
        self._model = model
        self._dimensions = dimensions  # None -> pleine dim (1536)
        # TODO for enabling cache
        # self.cache_file = Path(cache_dir)  / f"embedding_cache" / f"{model}.json"
        # with self.cache_file.open(encoding="utf-8") as f:
        #     self.review2embed : dict[str, str] = json.load(f)

    def encode(self, val1: str | list[str]) -> np.ndarray:
        single = isinstance(val1, str)
        texts = [val1] if single else list(val1)
        if not texts:
            return (np.zeros(self._out_dim(), dtype=np.float64) if single
                    else np.empty((0, self._out_dim()), dtype=np.float64))
        # garde-fou : chaîne vide -> API 400. On la remplace ou on lève.
        if any(not t or not t.strip() for t in texts):
            raise ValueError(f"encode: input contient une chaîne vide/blanche : {texts!r}")

        # TODO if embedding is too costy
        # texts_ = []
        # embds = []
        # for val in texts:
        #     embd = self.review2embed.get(val)
        #     if embd is not None:
        #         embds.append(embd)



        kwargs = {"input": texts, "model": self._model}
        if self._dimensions is not None:
            kwargs["dimensions"] = self._dimensions
        resp = self._client.embeddings.create(**kwargs)
        embs = np.array([d.embedding for d in resp.data], dtype=np.float64)
        return embs[0] if isinstance(val1, str) else embs

    def _out_dim(self):
        return self._dimensions if self._dimensions is not None else 1536

    def compare(self, val1, val2):
        # val1, val2 : ndarray déjà encodés (ou str -> on encode par commodité)
        a = self._as_embeddings(val1)
        b = self._as_embeddings(val2)
        a = np.atleast_2d(a)
        b = np.atleast_2d(b)
        a = a / np.linalg.norm(a, axis=1, keepdims=True)
        b = b / np.linalg.norm(b, axis=1, keepdims=True)
        return a @ b.T

    def _as_embeddings(self, v):
        if isinstance(v, np.ndarray):
            return v
        return self.encode(v)  # str ou list[str]

    def decode(self, val1):
        raise NotImplementedError(
            "L'embedding n'est pas inversible : projection non-bijective vers R^d."
        )



class SentenceTransformerWrapper(EmbeddingModel):
    def __init__(self, embedder):
        self.embedder = SentenceTransformer(f"sentence-transformers/{embedder}")

    def compare(self, val1, val2):
        embedded_source = self.embedder.encode(val1)
        embedded_target = self.embedder.encode(val2)
        return self.embedder.similarity(embedded_source, embedded_target)


    def encode(self, val1):
        return self.embedder.encode(val1)


    def decode(self, val1):
        return self.embedder.decode(val1)
