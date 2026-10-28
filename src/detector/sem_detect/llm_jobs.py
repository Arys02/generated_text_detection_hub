# src/detector/sem_detect/llm_jobs.py
"""Tâches LLM locales (vLLM, API compatible OpenAI) avec cache JSONL reprenable.

- LocalReviewGenerator : reviews de référence, prompt de semdetect.review_generator (importé)
- LocalReviewCleaner   : nettoyage, prompt de semdetect.review_cleaner (importé)
Clé de cache = tâche + modèle servi + paramètres de décodage + empreinte du prompt.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
from openai import OpenAI
from tqdm import tqdm

from src.detector.sem_detect.semdetect.review_cleaner import _CLEANING_SYSTEM_PROMPT, _TRIPLE_BRACES

DEFAULT_VLLM_BASE_URL = os.getenv("VLLM_API_BASE", "http://127.0.0.1:8000/v1")  # surchargeable par VLLM_API_BASE
from src.detector.sem_detect.semdetect.review_generator import RATING_SCALE, _ai_review_system_prompt

logger = logging.getLogger(__name__)


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


class CachedChatJob:
    job_name = "job"

    def __init__(
        self,
        cache_root: Path,
        served_model: str,
        base_url: str = DEFAULT_VLLM_BASE_URL,
        decoding: Optional[dict] = None,
        extra_body: Optional[dict] = None,
        max_workers: int = 8,
        timeout: float = 1800.0,
        max_attempts: int = 2,
    ):
        self.client = OpenAI(base_url=base_url, api_key="EMPTY", timeout=timeout)
        self.base_url = base_url.rstrip("/")
        self.served_model = served_model
        self.decoding = dict(decoding or {})
        self.extra_body = dict(extra_body or {})
        self.max_workers = max_workers
        self.max_attempts = max_attempts

        self.params = {"job": self.job_name, "served_model": served_model, "decoding": self.decoding,
                       "extra_body": self.extra_body, "prompt_sha": self.prompt_fingerprint()}
        self.source_tag = f"{self.job_name}:{served_model}:{_sha(self.params)[:10]}"

        cache_dir = Path(cache_root) / "llm_jobs" / self.job_name
        cache_dir.mkdir(parents=True, exist_ok=True)
        stem = self.source_tag.replace(":", "_").replace("/", "_")
        self.cache_path = cache_dir / f"{stem}.jsonl"
        params_path = cache_dir / f"{stem}.params.json"
        if not params_path.exists():
            params_path.write_text(json.dumps(self.params, indent=2), encoding="utf-8")

        self._lock = threading.Lock()
        self._cache: Dict[str, dict] = self._load()
        logger.info(f"{self.job_name} [{served_model}] cache {self.cache_path.name} : "
                    f"{sum(r['status'] == 'ok' for r in self._cache.values())} ok")

    # --------------------------------------------------- à spécialiser
    def prompt_fingerprint(self) -> str:
        raise NotImplementedError

    def build_messages(self, item: dict) -> List[dict]:
        raise NotImplementedError

    def prepare(self, item: dict) -> dict:
        return item

    def parse(self, text: str, item: dict) -> dict:
        return {"text": text.strip()}

    # ------------------------------------------------------------ cache
    def _load(self) -> Dict[str, dict]:
        out: Dict[str, dict] = {}
        if self.cache_path.exists():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    out[rec["id"]] = rec  # la dernière tentative l'emporte
        return out

    def _append(self, rec: dict) -> None:
        with self._lock:
            with self.cache_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
            self._cache[rec["id"]] = rec

    # ------------------------------------------------------------- appel
    def _work(self, item: dict) -> dict:
        last_err: Optional[str] = None
        for attempt in range(1, self.max_attempts + 1):
            t0 = time.perf_counter()
            try:
                prepared = self.prepare(item)
                resp = self.client.chat.completions.create(
                    model=self.served_model, messages=self.build_messages(prepared),
                    extra_body=self.extra_body or None, **self.decoding,
                )
                choice = resp.choices[0]
                text = choice.message.content or ""
                if not text.strip():
                    raise ValueError("réponse vide")
                out = self.parse(text, prepared)
            except Exception as e:  # consigné, pas propagé
                last_err = f"{type(e).__name__}: {e}"
                continue
            return {"id": item["id"], "status": "ok", **out, "finish_reason": choice.finish_reason,
                    "attempts": attempt, "seconds": round(time.perf_counter() - t0, 2),
                    "prompt_tokens": resp.usage.prompt_tokens, "completion_tokens": resp.usage.completion_tokens,
                    **prepared.get("_meta", {})}
        return {"id": item["id"], "status": "error", "error": last_err, "attempts": self.max_attempts}

    # -------------------------------------------------------------- API
    def run(self, items: List[dict], desc: Optional[str] = None) -> pd.DataFrame:
        todo = [it for it in items if self._cache.get(it["id"], {}).get("status") != "ok"]
        if todo:
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                futures = [pool.submit(self._work, it) for it in todo]
                for fut in tqdm(as_completed(futures), total=len(futures),
                                desc=desc or f"{self.job_name} ({self.served_model})"):
                    self._append(fut.result())
        return self.collect([it["id"] for it in items])

    def collect(self, ids: List[str]) -> pd.DataFrame:
        """État du cache pour ces ids, sans appel au LLM (status 'missing' si jamais traité)."""
        return pd.DataFrame([self._cache.get(i, {"id": i, "status": "missing"}) for i in ids])


class LocalReviewGenerator(CachedChatJob):
    """generate_ai_review du code d'origine, servi localement. Troncature en fin de papier si le texte
    dépasse le contexte, comptée avec le tokenizer du modèle servi et consignée."""
    job_name = "review_generation"
    TEMPLATE_MARGIN = 256  # gabarit de chat (tokens spéciaux, rôles) : marge prudente

    def __init__(self, cache_root: Path, served_model: str, base_url: str = DEFAULT_VLLM_BASE_URL,
                 max_workers: int = 8, temperature: float = 0.9, top_p: float = 0.9, max_tokens: int = 3072, **kw):
        super().__init__(cache_root, served_model, base_url,
                         decoding={"temperature": temperature, "top_p": top_p, "max_tokens": max_tokens},
                         max_workers=max_workers, **kw)
        self.root_url = self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url
        self._max_model_len: Optional[int] = None

    def prompt_fingerprint(self) -> str:
        return _sha(_ai_review_system_prompt("{score}"))[:12]

    def max_model_len(self) -> int:
        if self._max_model_len is None:
            r = requests.get(f"{self.base_url}/models", timeout=10)
            r.raise_for_status()
            model = next(m for m in r.json()["data"] if m["id"] == self.served_model)
            self._max_model_len = int(model["max_model_len"])
        return self._max_model_len

    def _tokenize(self, text: str) -> List[int]:
        r = requests.post(f"{self.root_url}/tokenize", timeout=120,
                          json={"model": self.served_model, "prompt": text, "add_special_tokens": False})
        r.raise_for_status()
        return r.json()["tokens"]

    def _detokenize(self, tokens: List[int]) -> str:
        r = requests.post(f"{self.root_url}/detokenize", timeout=120,
                          json={"model": self.served_model, "tokens": tokens})
        r.raise_for_status()
        return r.json()["prompt"]

    def prepare(self, item: dict) -> dict:
        score = item["score"]
        if score not in RATING_SCALE:
            raise ValueError(f"note {score!r} hors échelle {RATING_SCALE}")
        overhead = (len(self._tokenize(_ai_review_system_prompt(score)))
                    + len(self._tokenize("Paper Content:\n")) + self.TEMPLATE_MARGIN)
        budget = self.max_model_len() - self.decoding["max_tokens"] - overhead
        tokens = self._tokenize(item["paper_text"])
        meta = {"paper_tokens": len(tokens), "truncated": len(tokens) > budget}
        text = item["paper_text"]
        if meta["truncated"]:
            text = self._detokenize(tokens[:budget])
            meta["kept_tokens"] = budget
        return {**item, "paper_text": text, "_meta": meta}

    def build_messages(self, item: dict) -> List[dict]:
        return [{"role": "system", "content": _ai_review_system_prompt(item["score"])},
                {"role": "user", "content": f"Paper Content:\n{item['paper_text']}"}]


class LocalReviewCleaner(CachedChatJob):
    """clean_review du code d'origine, servi localement (température 0, triples accolades)."""
    job_name = "review_cleaning"

    def __init__(self, cache_root: Path, served_model: str, base_url: str = DEFAULT_VLLM_BASE_URL,
                 max_workers: int = 32, max_tokens: int = 3072, enable_thinking: bool = False, **kw):
        super().__init__(cache_root, served_model, base_url,
                         decoding={"temperature": 0.0, "max_tokens": max_tokens},
                         extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
                         max_workers=max_workers, **kw)

    def prompt_fingerprint(self) -> str:
        return _sha(_CLEANING_SYSTEM_PROMPT)[:12]

    def build_messages(self, item: dict) -> List[dict]:
        return [{"role": "system", "content": _CLEANING_SYSTEM_PROMPT},
                {"role": "user", "content": item["text"]}]

    def parse(self, text: str, item: dict) -> dict:
        m = _TRIPLE_BRACES.search(text)
        return {"text": m.group(1).strip() if m else text.strip(), "braces": bool(m)}