# Serving LLMs with vLLM

Cristal Anchor and the local parts of Sem-Detect call a language model through an **OpenAI-compatible API served by vLLM** on `127.0.0.1:8000` by default (override with `VLLM_API_BASE`). The server lives in a separate Python environment outside the repository and is controlled either by hand (tmux) or from Python by `src/utils/vllm_manager.py`.

This page documents the vLLM environment the project was run with (scripts, logs, profiles; the profile files `profiles/*.env` contain only vLLM model settings, no credentials) and the code in the repository that talks to it. The environment itself is not in the repository; its location is set by `VLLM_ENV_DIR` (default `~/vllm-env`). Nothing here was started or executed while writing.

### Settings (all optional environment variables, read from the environment or from `.env`)

| Variable | Default | Used by |
|---|---|---|
| `VLLM_API_BASE` | `http://127.0.0.1:8000/v1` | `cristal_anchor.py`, and the default `base_url` of `claims_llm.py` and `llm_jobs.py` |
| `VLLM_ENV_DIR` | `~/vllm-env` | `src/utils/vllm_manager.py` (`launch_server.sh`, `watchdog.py`, `profiles/`, `logs/`) |
| `SAGE_SEGMENTATION_DEVICE` | `cuda:1` with two GPUs, `cuda:0` with one, `cpu` without | SAGE segmentation (relevant to GPU sharing, below) |

The environment variables are read when the module is imported, so set them before importing it.

## 1. Who uses the server

| Component | Client code | Model name used | Notes |
|---|---|---|---|
| Cristal Anchor | `src/detector/cristal_anchor/cristal_anchor.py` (`_client = OpenAI(base_url=VLLM_API_BASE, api_key="EMPTY")`) | `params.model_name` of the detector config (`qwen3-8b` by default, `qwen3-14b` in the 14b presets) | must equal the profile's `SERVED_NAME`; chat completions with `top_k` and `chat_template_kwargs.enable_thinking` in `extra_body` |
| Sem-Detect claims | `src/detector/sem_detect/claims_llm.py` (`LocalClaimExtractor`) | `served_model`, e.g. `qwen3-32b` | JSON-schema constrained output, 16 workers by default, thinking off by default |
| Sem-Detect reference reviews and cleaning | `src/detector/sem_detect/llm_jobs.py` (`LocalReviewGenerator`, `LocalReviewCleaner`) | `mistral-small-3.2-fp8`, `gemma-3-27b`, `glm-4-32b` (generators); `qwen3-32b` (cleaner) | the generator also uses the server's tokenize/detokenize endpoints and `/v1/models` to read `max_model_len` |
| Notebooks `semdetect_e1`, `e2`, `e3`, `semdetect_claims_local` | `VLLMManager(gpu=1, ...)` then `mgr.ensure(<profile>)` | see section 4 | start and switch models automatically |

SAGE does not use vLLM, but its segmentation runs by default on `cuda:1` when two GPUs are present, the same GPU the server uses by default (set `SAGE_SEGMENTATION_DEVICE` to move it): **stop the server before evaluating SAGE** (`VLLMManager.stop()` or `tmux kill-session`).

## 2. The environment (`$VLLM_ENV_DIR`)

A Python virtual environment (vLLM 0.11.0, from `lib/*/site-packages/vllm-0.11.0.dist-info`) with these project scripts at its root:

| File | Role |
|---|---|
| `launch_server.sh [profile]` (default profile `qwen3-8b`) | loads `profiles/<profile>.env`, activates the environment and runs `vllm serve` |
| `activate_vllm.sh` | activates the environment and exports the cache paths (below) |
| `watchdog.py` | stops the server after two hours without a finished request (section 3.3) |
| `test_sanity.py` | minimal in-process vLLM check (`LLM(...)`, no server; the model is downloaded from Hugging Face) with `Qwen/Qwen2.5-0.5B-Instruct` on GPU 1 (`gpu_memory_utilization=0.15`, `enforce_eager=True`) |
| `profiles/*.env` | one file per served model |
| `logs/<profile>_<YYYYMMDD_HHMMSS>.log` | server output, one file per start |

Environment variables set by the scripts: `VLLM_USE_FLASHINFER_SAMPLER=0`, `TRITON_CACHE_DIR=$CACHE_DIR/triton_vllm` and `VLLM_CACHE_ROOT=$CACHE_DIR/vllm` (see `doc/reproducing-results.md`).

### 2.1 Profiles

Ten profiles are present. The `-batch` profiles (high GPU memory share, many concurrent sequences) are the ones used by the Sem-Detect notebooks; the plain ones (`qwen3-8b`, `qwen3-14b`, `qwen3-8b-max_length`) take a small share of the GPU (0.25 to 0.32) so that something else can run on the same card.

| Profile | `MODEL_ID` | `SERVED_NAME` | Quantization | dtype | `MAX_MODEL_LEN` | `GPU_MEM_UTIL` | `MAX_NUM_SEQS` | Reasoning parser |
|---|---|---|---|---|---|---|---|---|
| `qwen3-8b` | `Qwen/Qwen3-8B-AWQ` | `qwen3-8b` | `awq_marlin` | float16 | 32768 | 0.25 | 8 | `qwen3` |
| `qwen3-8b-max_length` | `Qwen/Qwen3-8B-AWQ` | `qwen3-8b` | `awq_marlin` | float16 | 65536 | 0.25 | 8 | `qwen3` |
| `qwen3-8b-batch` | `Qwen/Qwen3-8B-AWQ` | `qwen3-8b` | `awq_marlin` | float16 | 16384 | 0.85 | 32 | `qwen3` |
| `qwen3-14b` | `Qwen/Qwen3-14B-AWQ` | `qwen3-14b` | `awq_marlin` | float16 | 32768 | 0.32 | 8 | `qwen3` |
| `qwen3-14b-batch` | `Qwen/Qwen3-14B-AWQ` | `qwen3-14b` | `awq_marlin` | float16 | 16384 | 0.85 | 32 | `qwen3` |
| `qwen3-30b-a3b-batch` | `Qwen/Qwen3-30B-A3B-GPTQ-Int4` | `qwen3-30b-a3b` | `gptq_marlin` | float16 | 16384 | 0.90 | 32 | `qwen3` |
| `qwen3-32b-batch` | `Qwen/Qwen3-32B-AWQ` | `qwen3-32b` | `awq_marlin` | float16 | 16384 | 0.90 | 32 | `qwen3` |
| `gemma-3-27b-batch` | `ISTA-DASLab/gemma-3-27b-it-GPTQ-4b-128g` | `gemma-3-27b` | none (GPTQ detected by vLLM) | bfloat16 | 32768 | 0.90 | 8 | none |
| `glm-4-32b-batch` | `mratsim/GLM-4-32B-0414.w4a16-gptq` | `glm-4-32b` | none | auto | 32768 | 0.90 | 16 | none |
| `mistral-small-3.2-batch` | `RedHatAI/Mistral-Small-3.2-24B-Instruct-2506-FP8` | `mistral-small-3.2-fp8` | none (FP8 checkpoint) | auto | 32768 | 0.90 | 8 | none |

The Mistral profile also sets `EXTRA_ARGS='--tokenizer-mode mistral --config-format mistral --load-format mistral --limit-mm-per-prompt {"image":0}'`. No profile sets `PORT` (default 8000) or `CUDA_VISIBLE_DEVICES` (default 1). Because `launch_server.sh` sources the profile before applying the default, a profile that did define `CUDA_VISIBLE_DEVICES` would override the `gpu=` given to `VLLMManager`. The profile used for Cristal Anchor depends on the preset's `model_name`: `qwen3-8b` / `qwen3-14b` are served by either the plain or the `-batch` profile of that model.

A profile is a shell file sourced by `launch_server.sh`. The variables the script reads are:

| Variable | Required | Meaning |
|---|---|---|
| `MODEL_ID` | yes | model passed to `vllm serve` (Hugging Face id or path) |
| `SERVED_NAME` | yes | name exposed by the API (`--served-model-name`); this is the string clients pass as `model` |
| `DTYPE` | yes | `--dtype` |
| `MAX_MODEL_LEN` | yes | `--max-model-len` |
| `GPU_MEM_UTIL` | yes | `--gpu-memory-utilization` |
| `MAX_NUM_SEQS` | no (16) | `--max-num-seqs` |
| `PORT` | no (8000) | listening port; if you change it, set `VLLM_API_BASE` accordingly |
| `QUANTIZATION` | no | `--quantization` |
| `REASONING_PARSER` | no | `--reasoning-parser` (needed to receive Qwen3 thinking as `reasoning_content`) |
| `EXTRA_ARGS` | no | extra arguments split on spaces (e.g. `--tokenizer-mode mistral`) |
| `CUDA_VISIBLE_DEVICES` | no (1) | GPU used by the server |

The server always listens on `127.0.0.1` (not reachable from other machines). To add a model, create a new profile with these variables; the file name is the profile name, and `SERVED_NAME` must be what the client config uses.

## 3. Running the server

### 3.1 By hand

```bash
tmux new -d -s vllm "CUDA_VISIBLE_DEVICES=1 $VLLM_ENV_DIR/launch_server.sh qwen3-14b 2>&1 | tee -a $VLLM_ENV_DIR/logs/qwen3-14b_manual.log"
curl -s http://127.0.0.1:8000/health          # 200 when the process is up
curl -s http://127.0.0.1:8000/v1/models        # lists the served name
curl -s http://127.0.0.1:8000/metrics | grep vllm:request_success_total
tmux kill-session -t vllm                      # stop
```

The first start of a model can take several minutes (loading and compilation; the manager waits up to 1800 s). The `vllm-start` and `vllm-stop` commands mentioned in the docstring of `vllm_manager.py` are not in the repository nor in `vllm-env` (whether they exist elsewhere on a `PATH` was not checked); the tmux commands above are what the manager actually runs, plus `tmux new -d -s watchdog "source <env>/bin/activate && python <env>/watchdog.py"` when `restart_watchdog=True`.

### 3.2 From Python: `VLLMManager`

```python
from src.utils.vllm_manager import VLLMManager

mgr = VLLMManager(gpu=1, restart_watchdog=True)   # defaults: 127.0.0.1:8000, env dir $VLLM_ENV_DIR or ~/vllm-env
mgr.ensure("qwen3-32b-batch")                     # reuse if already served, otherwise stop + start
```

Behaviour (`vllm_manager.py`):

- `profile_served_name(profile)` reads the `SERVED_NAME=` line of the profile file (this is the only variable the manager parses).
- `ensure(profile)`: if `/health` is 200, the only served model is the profile's `SERVED_NAME`, and a one-token chat request succeeds, the server is reused; otherwise `stop()` then `start(profile)`.
- `stop()`: kills the tmux sessions `watchdog` then `vllm`, then waits (up to 180 s) until `/health` fails and the GPU memory used (via `nvidia-smi`) is at most 1500 MiB; raises `RuntimeError` otherwise (a stray process still holds the GPU).
- `start(profile)`: refuses to start if a tmux session `vllm` already exists, runs `CUDA_VISIBLE_DEVICES=<gpu> launch_server.sh <profile>` in a detached session `vllm` with the output tee'd to `logs/<profile>_<timestamp>.log`, waits for readiness (health, model listed, real test request), then, if `restart_watchdog=True`, starts `watchdog.py` in a session `watchdog`.
- `tmux` and `nvidia-smi` must be installed (if `nvidia-smi` fails, `subprocess.CalledProcessError` is raised, not `RuntimeError`); a failed start raises with the tail of the log.
- `VLLMManager.port` only changes the control URLs; it is not passed to `launch_server.sh`, so a profile with another `PORT` would break the manager.
- `ensure()` reuses a healthy server without restarting a dead watchdog.

### 3.3 Watchdog

`watchdog.py` polls `/metrics` every 60 s and sums `vllm:request_success_total`. After 7200 s without change it kills the tmux session `vllm` and exits (if the kill fails, it retries at the next poll). A long job that sends no request for two hours (for example a notebook paused between cells) loses its server, and the next `ensure` restarts it. When the server is unreachable the watchdog exits by itself. The file also contains a fully commented-out copy of the same code above the active one.

## 4. Models used by the experiments

From the Sem-Detect notebooks (code read from the notebooks; they were not run):

| Step | Served name → profile |
|---|---|
| Claim extraction (E1, E2, E3) and review cleaning (E2) | `qwen3-32b` → `qwen3-32b-batch` |
| Reference-review generation (E2) | `mistral-small-3.2-fp8` → `mistral-small-3.2-batch`, `gemma-3-27b` → `gemma-3-27b-batch`, `glm-4-32b` → `glm-4-32b-batch` |
| Claim extraction campaign (`semdetect_claims_local`) | `qwen3-8b`, `qwen3-14b`, `qwen3-14b` with thinking (12 000 max tokens), `qwen3-30b-a3b`, `qwen3-32b` → the matching `-batch` profiles |
| Cristal Anchor presets | `qwen3-8b` (`cristal_anchor_1/2/3`) or `qwen3-14b` (`*_14b*`); the profile with that `SERVED_NAME` must be running (`qwen3-8b`, `qwen3-14b`) |

`semdetect_e2` and `semdetect_claims_local` check that the profile's `SERVED_NAME` equals the name they use and raise `ValueError` otherwise; `semdetect_e1` and `semdetect_e3` call `mgr.ensure("qwen3-32b-batch")` without that check. `semdetect_claims_local` uses `restart_watchdog=False`, the other three `True`.

## 5. Caches written by the clients (`CACHE_DIR`)

Sem-Detect jobs are resumable: each record is appended to a JSONL file and records already `ok` are skipped.

| Path under `CACHE_DIR` | Written by | Key |
|---|---|---|
| `claims_llm/vllm_<model>_<hash>.jsonl` and `.params.json` | `LocalClaimExtractor` | provenance tag `vllm:<served>:<hash>` over served model, temperature, max tokens, thinking flag, prompt and schema hashes |
| `llm_jobs/review_generation/`, `llm_jobs/review_cleaning/` | `LocalReviewGenerator`, `LocalReviewCleaner` | task + served model + decoding parameters + `extra_body` + prompt fingerprint (`review_generation/` also holds an orphan `.params.json` from an earlier model name, `mistral-small-3.2`) |
| `sem_detect_store/emb/`, `sem_detect_store/txt/` | `SemDetectFeatureStore` | claim provenance + embedding model; text provenance + model + max length (these use local GPU models, not vLLM) |
| `sem_detect_e2/local_refs.parquet`, `sem_detect_e3/pilot_universe.parquet` | notebooks | experiment outputs |

Changing the model, the decoding parameters or the prompt produces a new cache file instead of overwriting the old one. Cristal Anchor has no cache: every evaluated review costs two LLM calls (prompt reconstruction with thinking, then generation).

## 6. Practical notes and pitfalls

- Memory: the `-batch` profiles claim 85 to 90 % of GPU 1, so run one profile at a time and keep SAGE segmentation off that GPU while they run. The plain `qwen3-8b` (0.25) and `qwen3-14b` (0.32) profiles leave room for other processes, but whether SAGE's two RoBERTa models plus its activations fit next to them was not tested.
- Request failures in Cristal Anchor abort the evaluation (`predict`, the method the evaluator calls, has its `try/except` commented out; only the unused `predict_` handles errors), so verify `/v1/models` before a long run.
- The log folder contains several logs of the same profile started within minutes (for example four `mistral-small-3.2-batch` logs); they were not read and may be failed starts.
- The base URL comes from `VLLM_API_BASE` (default `http://127.0.0.1:8000/v1`) for Cristal Anchor and as the default `base_url` of `LocalClaimExtractor`, `LocalReviewGenerator` and `LocalReviewCleaner`; `VLLMManager` takes `host` and `port` arguments (default 8000), so keep them consistent when you change the port.
- Generation is not seeded (`seed` is not sent), so reruns differ (audit anomaly A19).
- Qwen3 thinking mode: pass `enable_thinking` through `chat_template_kwargs`; thinking tokens count in `max_tokens`, which is why the Sem-Detect thinking variant uses 12 000 and Cristal Anchor's prompt reconstruction 20 000.
- Environment recreation: the repository does not contain the vLLM environment, its profiles or its scripts. To rebuild, create a virtual environment with the vLLM version of the audit, copy the scripts described above and write one profile per model with the variables of section 2.1 (audit anomaly A34).
