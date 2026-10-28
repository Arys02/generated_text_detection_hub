# Known issues

Limits of the code base as found during the audit (a read-only review; the project is frozen, so none of these was fixed). The identifiers (A1, A2, ...) come from the audit's anomaly register, which is kept in the maintainers' private repository. "Static" means established by reading the code, not by running it.

## Results and reproducibility

| ID | Issue | Where |
|---|---|---|
| A11 | The reference SAGE-APRDB run (hash `b5ad7993…`) can no longer be recomposed from the presets, and its artifact was overwritten by identical retrainings (the current file is the last, 2026-09-16) | `conf/experiment/train/train_sage_gru_2_cat.yaml`, `results/b5ad7993…` |
| A14 | The APRDB dataset seed changed from `${seed}` (42) to the literal `0` on 2026-10-15: old rows of `results.jsonl` (1-36) are not recomposable by default; `dataset.seed=42` should restore them (not verified) | `conf/dataset/aprdb.yaml` |
| A38 | In `results.jsonl` records of SAGE evaluations the `name` key holds the train config's name (`gru`), not the eval preset name; only the hash and `train_preset` identify the run | `src/evaluation/run_eval.py` |
| A39 | `evaluation_batch/evaluation_segmented_txt.jsonl` grows at every SAGE evaluation (7.9 GB on 2026-10-20); each 10 000-review row adds about 50 MB to `results.jsonl` | `sage.py`, `$CACHE_DIR/results/` |
| A15 | `results.jsonl` line 42 contains two records because concurrent runs append without locking; both records are recoverable | `src/evaluation/eval_utils.py` (`append_to_jsonl`) |
| A16 | SAGE needs two RoBERTa models with no documented source or training code (available through `scripts/fetch_sage_models.py`); the segmentation device and model directory used to be hard-coded (`cuda:1`, a machine-specific path) and are now settable (`SAGE_SEGMENTATION_DEVICE`, `SAGE_MODELS_DIR`); CPU and single-GPU runs are untested | `training/linguistic_segmentation.py` |
| A17 | `.env.template` points `CACHE_DIR` to a non-existing location and omits several variables | `.env.template` |
| A18 | Exports of `semdetect_sage_recap` are missing on disk (tables, score parquets, one figure; one PNG is empty); published cross-dataset numbers survive only in notebook outputs | `$CACHE_DIR/recap/` |
| A19 | Same config, same hash, different results for `dummy`, some early SAGE rows and `cristal_anchor_1.1_14b` (non-deterministic) | `random_base_detector.py`, `cristal_anchor.py` |
| A8, A20 | `requirements_local.txt` is incomplete and version-divergent | `requirements_local.txt` |
| A21 | No deterministic GPU flags: training is not bit-for-bit reproducible | `gru_train.py`, `transformer_train.py` |
| A22 | APRDB revision not pinned; network needed on a cache miss; one dataset cache is at an old location | `dataset_aprdb.py` |
| A1 | A refactor-equivalence test gave 163 and 124 reviews against 120 in the 2026-09-16 reference (not investigated; test removed) | former `tests/golden/` |

## Code defects (static)

| ID | Issue | Where |
|---|---|---|
| A35 | `training/gru_features.py` reads `spec.include_valid` (absent from `FeatureSpec`) and calls `get_spec("v1")`/`get_spec("v2")` (not in `SPECS`): the legacy `SageGRU` path, `transformer_train.py` and one `xai` helper cannot work | `training/gru_features.py` |
| A36 | The Transformer trainer picks the best epoch on the validation AUC: optimistic validation AUC | `training/transformer_train.py` |
| A37 | `SinusoidalPE` is a fixed table of `max_len + 1` positions (its comment claims it extrapolates): a longer sequence fails at inference | `model/SubjTransformer.py` |
| A4 | Transformer artifacts (a real experimental variant of SAGE, four artifacts, not in the article for lack of space) cannot be evaluated: `Sage.build_model` only knows `mlp` and `gru` | `sage.py` |
| A23 | `segmentation_df_v2` concatenates a string path with `/`: `TypeError` if reached | `linguistic_segmentation.py` |
| A7 | The preset `train_sage_gru_2_cat_merged` refers to a feature spec that does not exist | `conf/experiment/train/` |
| A13 | Seven of the fourteen XAI experiments declared in `experiments.py` are never fed by `results_xai` | `xai/experiments.py` |
| A6 | `dataset_openturingbench.py` depends on `DATA_HWT_PATH`, which `config.py` no longer defines; `run_eval.py` still imports it | `src/data/`, `src/config.py` |
| A5 | The `bspline` and `bspline_theory` branches of Ada-DetectGPT are unreachable with the current presets (no `train_dataset`) | `ada_detect_gpt/` |
| A12 | `training.split` and `training.optimizer` are never read by the code, although `training.split` is part of the training hash | `conf/training/` |
| A32 | `torch.set_grad_enabled(False)` at import of `binocular.py`, which the detector factory imports | `binocular.py:20` |
| A10 | `logging.basicConfig(..., level=logging.error)` passes a function as level; `example.log` may be created at import | `cristal_anchor.py:20` |
| A9 | Sem-Detect E2: a text-cleaning truncation was never fixed; its measured degradation is an upper bound | `semdetect_e2.ipynb` |

Further observations from the detector pages: `n_perturbation` of the DetectGPT YAML is never read (10 perturbations are used), Fast-DetectGPT loads an unused sampling model and always takes the analytic branch, `BinocularDetector.predict` drops empty texts (misaligning scores and labels), Cristal Anchor requires `batch_size: 1` and `pdf_path`, and the MLP SAGE variant returns hard labels, not probabilities.

## Configuration and naming

| ID | Issue |
|---|---|
| A2 | Copy-pasted `name` values in presets (`fast_detect_gpt_simple` carries `detect_gpt_gpu1`, `sage_gru_2_cat` carries `sage_gru_2`, several `cristal_anchor_*` share names); `name` is part of the eval hash, and rows of `results.jsonl` carry misleading names |
| A3 | `sage_gru_3_semdetect.yaml` has a stale header; `ada_detect_gpt_simple_gemma_identity` differs from `ada_detect_gpt_simple` only by an empty `params` block and its `name` |
| A34 | `vllm_manager.py` relies on scripts in the vLLM environment (`VLLM_ENV_DIR`) (`launch_server.sh`, `watchdog.py`, the `vllm-start`/`vllm-stop` commands) that are not in the repository, and nothing imports it |

## Documentation and licences

| ID | Issue |
|---|---|
| A28 | `LICENCE` is empty; the upstream Sem-Detect repository has no licence file; the Ada-DetectGPT repository the baselines are ported from showed no licence file either (2026-10-20); the ported files carry no provenance header |
| A29 | APRDB is under an Intel research-use licence |
| A30 | The 0/1 meaning of the APRDB `label` is not documented upstream |
| A31 | The project article has no public reference yet |
| A24-A27 | Documentation items corrected or rewritten during the documentation work (`doc/detector/binocular.md`, `doc/detector/ada_detect_gpt.md`, agent definition) |
