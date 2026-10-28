# src/detector/sem_detect/claims_llm.py
"""Extraction des claims Sem-Detect par un LLM servi localement (vLLM, API compatible OpenAI).

Identique à semdetect.claim_extractor.extract_claims : prompt système (importé, non recopié),
message utilisateur, température, max_tokens, relance si 0 claim. Seule différence : la sortie
est contrainte par un JSON Schema (équivalent du schéma Gemini d'origine) -> JSON valide garanti.
Cache JSONL reprenable ; clé = modèle servi + hash du prompt + paramètres de décodage par défaut.
Les relances ciblées (retry_failed) consignent leurs paramètres de remplacement dans l'enregistrement.
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
from typing import Dict, List, Optional, Tuple

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

DEFAULT_VLLM_BASE_URL = os.getenv("VLLM_API_BASE", "http://127.0.0.1:8000/v1")  # surchargeable par VLLM_API_BASE

from src.detector.sem_detect.semdetect.claim_extractor import CLAIM_CATEGORIES, _CLAIM_EXTRACTION_PROMPT

logger = logging.getLogger(__name__)

CLAIMS_JSON_SCHEMA = {
    "type": "object",
    "properties": {c: {"type": "array", "items": {"type": "string"}} for c in CLAIM_CATEGORIES},
    "required": list(CLAIM_CATEGORIES),
    "additionalProperties": False,
}


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


class LocalClaimExtractor:
    def __init__(
        self,
        cache_root: Path,
        served_model: str,
        base_url: str = DEFAULT_VLLM_BASE_URL,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        max_workers: int = 16,
        timeout: float = 900.0,
        enable_thinking: bool = False,
        max_attempts: int = 2,
    ):
        self.client = OpenAI(base_url=base_url, api_key="EMPTY", timeout=timeout)
        self.served_model = served_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_workers = max_workers
        self.enable_thinking = enable_thinking
        self.max_attempts = max_attempts

        self.params = {
            "served_model": served_model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "enable_thinking": enable_thinking,
            "prompt_sha": _sha(_CLAIM_EXTRACTION_PROMPT)[:12],
            "schema_sha": _sha(CLAIMS_JSON_SCHEMA)[:12],
        }
        # tag de provenance : sert aussi de `claims_source` pour le feature store
        self.source_tag = f"vllm:{served_model}:{_sha(self.params)[:10]}"

        cache_dir = Path(cache_root) / "claims_llm"
        cache_dir.mkdir(parents=True, exist_ok=True)
        stem = self.source_tag.replace(":", "_").replace("/", "_")
        self.cache_path = cache_dir / f"{stem}.jsonl"
        params_path = cache_dir / f"{stem}.params.json"
        if not params_path.exists():
            params_path.write_text(json.dumps(self.params, indent=2), encoding="utf-8")

        self._lock = threading.Lock()
        self._cache: Dict[str, dict] = self._load()
        logger.info(f"Claims cache {self.cache_path} : {sum(r['status'] == 'ok' for r in self._cache.values())} ok")

    # ------------------------------------------------------------ cache
    def _load(self) -> Dict[str, dict]:
        out: Dict[str, dict] = {}
        if self.cache_path.exists():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    out[rec["review-id"]] = rec  # la dernière tentative l'emporte
        return out

    def _append(self, rec: dict) -> None:
        with self._lock:
            with self.cache_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
            self._cache[rec["review-id"]] = rec

    def records(self) -> pd.DataFrame:
        """État courant du cache (dernière tentative par review), sans appel au LLM."""
        if not self._cache:
            return pd.DataFrame()
        return pd.DataFrame(list(self._cache.values())).set_index("review-id")

    # ------------------------------------------------------------- appel
    def _call(self, text: str, max_tokens: int, extra_sampling: Optional[dict]) -> Tuple[Dict[str, List[str]], dict]:
        body = {"chat_template_kwargs": {"enable_thinking": self.enable_thinking}}
        if extra_sampling:
            body.update(extra_sampling)
        resp = self.client.chat.completions.create(
            model=self.served_model,
            messages=[
                {"role": "system", "content": _CLAIM_EXTRACTION_PROMPT},
                {"role": "user", "content": f"Review Text:\n\n{text}"},
            ],
            temperature=self.temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_schema",
                             "json_schema": {"name": "claim_extraction", "schema": CLAIMS_JSON_SCHEMA}},
            extra_body=body,
        )
        choice = resp.choices[0]
        if choice.finish_reason == "length":
            raise ValueError(f"sortie tronquée à max_tokens={max_tokens}")
        data = json.loads(choice.message.content)
        claims = {c: [str(x) for x in (data.get(c) or [])] for c in CLAIM_CATEGORIES}
        usage = {"prompt_tokens": resp.usage.prompt_tokens, "completion_tokens": resp.usage.completion_tokens}
        return claims, usage

    def _work(self, rid: str, text: str, max_tokens: Optional[int] = None,
              extra_sampling: Optional[dict] = None) -> dict:
        effective_max = max_tokens or self.max_tokens
        overrides = {k: v for k, v in {"max_tokens": max_tokens, **(extra_sampling or {})}.items() if v is not None}
        last_err: Optional[str] = None
        for attempt in range(1, self.max_attempts + 1):
            t0 = time.perf_counter()
            try:
                claims, usage = self._call(text, effective_max, extra_sampling)
            except Exception as e:  # erreur réseau, JSON, troncature : consignée, pas propagée
                last_err = f"{type(e).__name__}: {e}"
                continue
            n_claims = sum(len(v) for v in claims.values())
            if n_claims == 0 and attempt < self.max_attempts:  # même règle que le pipeline d'origine
                last_err = "0 claims"
                continue
            rec = {"review-id": rid, "status": "ok", "claims": claims, "attempts": attempt,
                   "seconds": round(time.perf_counter() - t0, 2), **usage}
            if overrides:
                rec["overrides"] = overrides
            return rec
        rec = {"review-id": rid, "status": "error", "claims": None, "attempts": self.max_attempts, "error": last_err}
        if overrides:
            rec["overrides"] = overrides
        return rec

    def _run(self, todo: List[Tuple[str, str]], max_tokens: Optional[int] = None,
             extra_sampling: Optional[dict] = None, desc: str = "") -> None:
        if not todo:
            return
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = [pool.submit(self._work, rid, txt, max_tokens, extra_sampling) for rid, txt in todo]
            for fut in tqdm(as_completed(futures), total=len(futures), desc=desc or f"Claims ({self.served_model})"):
                self._append(fut.result())

    # -------------------------------------------------------------- API
    def collect(self, df: pd.DataFrame) -> pd.DataFrame:
        """État du cache pour les reviews de df, sans appel au LLM.
        Retourne review-id, claim_extraction (JSON, '' si échec), claims_status."""
        rows = df[["review-id"]].drop_duplicates("review-id")
        recs = [self._cache.get(rid, {"status": "missing"}) for rid in rows["review-id"]]
        return pd.DataFrame({
            "review-id": rows["review-id"].to_numpy(),
            "claim_extraction": [json.dumps(r["claims"], ensure_ascii=False) if r.get("status") == "ok" else ""
                                 for r in recs],
            "claims_status": [r.get("status", "missing") for r in recs],
        })

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """df : colonnes review-id, content. N'appelle le LLM que pour les reviews absentes du cache
        ou en erreur, avec les paramètres par défaut."""
        rows = df[["review-id", "content"]].drop_duplicates("review-id")
        todo = [(rid, txt) for rid, txt in zip(rows["review-id"], rows["content"])
                if self._cache.get(rid, {}).get("status") != "ok"]
        self._run(todo)
        return self.collect(df)

    def retry_failed(self, df: pd.DataFrame, max_tokens: Optional[int] = None,
                     extra_sampling: Optional[dict] = None) -> pd.DataFrame:
        """Relance UNIQUEMENT les reviews en erreur, avec des paramètres de remplacement consignés
        dans l'enregistrement (champ `overrides`). La clé du cache n'est pas modifiée."""
        rows = df[["review-id", "content"]].drop_duplicates("review-id")
        todo = [(rid, txt) for rid, txt in zip(rows["review-id"], rows["content"])
                if self._cache.get(rid, {}).get("status") == "error"]
        logger.info(f"relance de {len(todo)} échec(s) avec max_tokens={max_tokens}, extra={extra_sampling}")
        self._run(todo, max_tokens=max_tokens, extra_sampling=extra_sampling, desc="Relance des échecs")
        return self.collect(df)