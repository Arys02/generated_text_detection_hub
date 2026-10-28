# Sem-Detect

Detector from the Sem-Detect project (Duarte et al., arXiv 2605.21713; dataset `Sem-Detect/ML_Conferences-Peer-Reviews`) embedded in this repository, plus an adaptation layer written for the project's experiments. It is **not** registered in `detector_factory` and is not evaluated by `run_eval`: it is used from notebooks.

## Licence and provenance

- `VENDORED.md` records the upstream source (`github.com/avduarte333/Sem-Detect`, commit `d97175a513fab1aa97b31cc9a06874bad79a21c0`) and the file-level boundary: the upstream part is `semdetect/` (17 files) and `models/lightgbm_best_model.pkl` (loaded by `semdetect/classifier.py`). Since the upstream repository has no licence, this part is **not versioned** here (git-ignored): install it with `python -m scripts.fetch_semdetect` and verify it with `--check` (SHA-256 manifest `semdetect.sha256`). The copy used for the article was byte-identical to a fresh install (checked 2026-10-21). Do not refactor, clean or reformat it.
- The upstream repository has no licence file (stated by the project author), so no licence can be quoted for this code and it is not redistributed (see the licence section of the root README).
- By content, the adaptation layer (versioned) is `claims_llm.py`, `feature_store.py`, `llm_jobs.py`, `references.py`, `text_cleaning.py` and `notebook/`; the upstream part is the fetched `semdetect/` and `models/lightgbm_best_model.pkl`. Citation: `duarte2026semdetect` in `doc/references.bib`.

## Upstream pipeline (`semdetect/pipeline.py`)

For a target review and its paper: (1) generate `k` reference AI reviews for the same paper, matched on the rating, with LLM providers (`providers/`: Bedrock, Gemini, LiteLLM, OpenAI); (2) clean all reviews; (3) extract claims; (4) embed the claims; (5) compute textual features of the target review; (6) build a 9-feature vector; (7) classify (`classifier.py`, LightGBM) into `ai`, `rewrite` or `human`. Modules: `claim_extractor`, `embedding_generator`, `feature_builder`, `textual_features`, `review_cleaner`, `review_generator`, `report`, `config`.

## Adaptation layer

| File | Role (from its docstring) |
|---|---|
| `claims_llm.py` | extracts claims with a locally served LLM (vLLM, OpenAI-compatible API) using the upstream prompt, a JSON-schema-constrained output, a resumable JSONL cache keyed by model, prompt hash and decoding parameters, and retries when zero claims are returned |
| `llm_jobs.py` | local LLM tasks with a resumable JSONL cache: `LocalReviewGenerator` (reference reviews with the upstream prompt) and `LocalReviewCleaner` |
| `feature_store.py` | resumable disk cache of expensive computations: `emb/<key>` for claim embeddings and `txt/<key>` for textual features (Mistral-7B), atomic fragment writes |
| `references.py` | protocols matching a target review to AI references, feature matrix and probabilities (`paper_protocol_references`, `PAIR_KEYS = conference, year, submission_id, rating`, `MIN_SOURCES = 3`) |
| `text_cleaning.py` | deterministic cleaners applied identically to all reviews: `clean_a` (upstream `strip_markdown`), `clean_b` (also strips list markers) |

`src/data/dataset_semdetect.py` only loads the dataset (`SemDetectDataset`, `label_mode`, `balance`, `val_fraction`; the config `conf/dataset/sem_detect.yaml` pins the Hugging Face revision).

## Experiments (notebooks in `src/detector/sem_detect/notebook/`)

From project notes, not re-verified against the notebooks: E0 reproduction with the dataset's own claims and references (`semdetect_repro`); E1 claims extracted locally by Qwen3-32B with the dataset's references (`semdetect_e1`); E2 Qwen3-32B claims with locally generated references (`semdetect_e2`; a text-cleaning truncation was never fixed, so its degradation figure is an upper bound); E3 Sem-Detect on APRDB with its native references (`semdetect_e3`; variant (a) is used in the article's cross-dataset table). Outputs are stored under `$CACHE_DIR/sem_detect_store/`, `sem_detect_e2/`, `sem_detect_e3/`. Notebook outputs may be the only trace of published results: do not re-execute or clear them.

## Requirements

The local LLM parts (claims, reference reviews, cleaning) use a vLLM server: profiles, the Python manager, the watchdog and the on-disk caches are described in `doc/vllm.md`.

Reference-review generation, claim extraction and the textual features need LLM access (cloud providers for the upstream pipeline, a local vLLM server for the adaptation layer) and a Mistral-7B model (`Qwen3-Embedding-0.6B` for claim embeddings, per the audit of the cached models). None of this was run during the documentation work.
