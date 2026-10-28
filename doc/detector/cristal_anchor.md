# Cristal Anchor

Homemade detector of this project (not ported from another repository). Idea: reverse-engineer a prompt from the review under test, ask an LLM to write a new review of the same paper with that prompt, and measure how close the generated review is to the original. A machine-generated review is expected to be reproduced more closely than a human one.

- Code: `src/detector/cristal_anchor/` (`cristal_anchor.py`, `embedding_model.py`, `fetch_academic_paper.py`; also `finetuning_anchor.py` and `script_fetch_paper.py`, see warnings); config `conf/detector/cristal_anchor.yaml`; detector name `cristal_anchor`, type `contextual`.

## How it works

`CristalAnchor.predict(batch)` receives the whole batch (the detector type is `contextual`) and reads `batch['content']` (reviews) and `batch['pdf_path']` (paper PDFs). For each review/paper pair:

1. `extract_text_from_pdf(pdf_path)` (`fetch_academic_paper.py`) extracts the paper text.
2. `review_pipeline(review, paper)`:
   - `reverse_engineer_prompt`: a long "reverse prompt engineering" instruction (`build_prompt` + `REVIEW_OUTPUT_SPEC`) asks the LLM, with thinking enabled, for a reconstructed system prompt and user prompt template (tags `<RECONSTRUCTED_SYSTEM_PROMPT>`, `<RECONSTRUCTED_USER_PROMPT>`, plus justification and falsification), including the target length in words;
   - `generate_review`: the reconstructed prompts, with the paper injected in place of `{paper}`, produce a new review.
3. The embedding model compares the original review with the generated one (cosine similarity) and `predict` returns `embedding_model.compare(texts, results)`: for `all-MiniLM-L6-v2` this is a B×B similarity matrix, not one score per review. It is only correct with `data_loader.batch_size: 1` (the default in `conf/data_loader/default.yaml`); with a larger batch the evaluator would flatten B² scores against B labels. (The unused `predict_` computes one score per pair.) The intended reading is that a high similarity indicates a machine-generated review; the code does not enforce a direction (the evaluator inverts scores automatically when the raw AUROC is below 0.5).

LLM calls go through the OpenAI-compatible API of a local vLLM server at `http://127.0.0.1:8000/v1` (`VLLM_API_BASE`, read from the environment with that default, client created at import with the API key `"EMPTY"`), passing `top_k` and `chat_template_kwargs.enable_thinking` in `extra_body`; when thinking is enabled, `chat()` caps the temperature at 0.6. The served model name is `params.model_name` (e.g. `qwen3-8b`, `qwen3-14b`). `CristalAnchor` does not start the server and does not import `src/utils/vllm_manager.py` (no module imports it). That helper starts tmux sessions named `vllm` (running `.../launch_server.sh <profile>` of the vLLM environment) and `watchdog` (`python watchdog.py`); these scripts, the `profiles/*.env` files and the `vllm-start`/`vllm-stop` commands mentioned in its docstring live in the vLLM environment (`VLLM_ENV_DIR`), outside the repository (audit anomaly A34). Start the server yourself before running this detector; see `doc/vllm.md` for profiles, the manager and the watchdog.

Embedding models (`embedding_model.py`, `embbeding_model_factory`): `all-MiniLM-L6-v2` (local sentence-transformers) or `text-embedding-3-small` (OpenAI API, key from `OPEN_AI_API_KEY`). Any other name raises `NotImplementedError`.

## Parameters (`conf/detector/cristal_anchor.yaml` and presets)

| Key | Value | Notes |
|---|---|---|
| `params.model_name` | `qwen3-8b` (presets use `qwen3-14b`) | served model name |
| `params.embedder_name` | `all-MiniLM-L6-v2` | |
| `params.seed` | `${...seed}` | **not passed to the LLM calls**, so generation is not seeded |
| `params_generate_review` | temperature 0.7, `max_token` 4096, `enable_thinking: False`, `top_p` 0.95, `top_k` 20 | |
| `params_reverse_prompt` | `enable_thinking: True`, temperature 0.6, `max_token` 20000 | |

## Data requirements

The dataset must provide `pdf_path`. APRDB does so with `with_pdf: true` (the presets also set `keep_meta: true`). `SemDetectDataset` does **not** provide `pdf_path` (`with_paper: true` only adds `paper_content`, `paper_appendix` and `forum_id`), so Cristal Anchor cannot be evaluated on Sem-Detect as the code stands. PDFs are fetched with the OpenReview client (`fetch_academic_paper.py`, credentials in `OPENREVIEW_LOGIN`/`OPENREVIEW_MDP`).

## Presets and running

Presets in `conf/experiment/eval/`: `cristal_anchor_1.yaml`, `_2`, `_3` (subset sizes 10, 60 and 600, default model `qwen3-8b`), and the `qwen3-14b` presets `cristal_anchor_1_14b.yaml` (`name: "cristal_anchor_1.1_14b"`, `subset_size: 10`), `_10_14b_oasmall`, `_60_14b`, `_60_14b_oasmall`, `_600_14b`, where the number is the subset size. Their `name` values are ambiguous (`cristal_anchor_1` and `cristal_anchor_10_14b_oasmall` both use `"cristal_anchor_1.1"`; see audit anomaly A2), so identify runs by their config, not by `name`.

```bash
# the vLLM server must already be reachable on 127.0.0.1:8000
python -m src.evaluation.run_eval experiment/eval=cristal_anchor_1_14b
```

## Caveats

- Not deterministic: same config, same hash, different results (audit anomaly A19; AUROC values of 0.0, 0.0 and 0.76 were recorded for three runs of `cristal_anchor_1.1_14b`).
- `predict` has no error handling: one failing PDF or LLM call aborts the evaluation. The alternative `predict_` (with `try/except` returning `NaN` for failures and a `(N, 1)` array) is not the method called by the evaluator.
- Import-time side effects: `logging.basicConfig(filename='example.log', level=logging.error)` (invalid level object; the file may or may not be created depending on import order, audit anomaly A10) and the OpenAI client.
- **Never import** `finetuning_anchor.py` (runs a fine-tuning at import) or `script_fetch_paper.py` (downloads papers and writes `data.pkl` at import).
- The `__main__` block of `cristal_anchor.py` cannot run: its hard-coded config has the key `params_revers_prompt` (misspelt) while `__init__` reads `config['params_reverse_prompt']` (a `KeyError`), lacks `embedder_name`, and uses `max_tokens` instead of `max_token`.
