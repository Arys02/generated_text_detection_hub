# Architecture

How one run goes from a Hydra config to a line in `results.jsonl`, and which files implement each step.
All paths are relative to the repository root; commands are run from the root as `python -m <module>` with the `.venv` interpreter.
Facts below were read from the code during the audit and checked by an independent review; where a claim could not be verified without running code, it is said so. Artifacts are overwritten if a training is re-run with an identical config (same hash): keep a copy before retraining (audit anomaly A11).

## 1. Big picture

```
conf/ (Hydra)  ──►  entry point  ──►  dataset_factory ─► Dataset ─► DataLoader ┐
                    run_eval.py        detector_factory ─► Detector            ├► Evaluator ─► metrics ─► RESULT_DIR/results.jsonl
                    gru_train.py ...                                           ┘
                    (training)  ─────────────────────────────────────────────────► RESULT_DIR/<config_hash>/artifact.joblib
```

Two kinds of entry points, both Hydra applications whose root config lives in `conf/`:

| Entry point | `config_name` | Output |
|---|---|---|
| `src/evaluation/run_eval.py` (`main`) | `eval` | one JSON record appended to `RESULT_DIR/results.jsonl` |
| `src/detector/subjectivity_aware_generation_estimator/training/{gru,mlp,transformer}_train.py` | `train` | `RESULT_DIR/<config_hash>/artifact.joblib` (GRU also writes `gru_result.npz`) |

Only SAGE has a training entry point; the other detectors are evaluated directly (Ada-DetectGPT fits a weight function internally when configured with a B-spline `w_func`, see section 6).
Sem-Detect is not registered in `detector_factory`: it is not evaluated by `run_eval` but used from notebooks (`src/detector/sem_detect/notebook/`, `notebook/results/`).

## 2. Configuration (Hydra)

### 2.1 Entry configs

`conf/eval.yaml` composes the groups `hardware`, `dataset`, `data_loader`, `detector`, and an optional `experiment/eval`.
`conf/train.yaml` composes `hardware`, two copies of the `dataset` group under `dataset_train` and `dataset_val`
(package redirection `dataset@dataset_train`), `data_loader`, `training/split`, `training/model` and an optional `experiment/train`.
Both set `seed: 42` at the root.

### 2.2 Presets

A preset is a file `conf/experiment/{eval,train}/<name>.yaml` starting with `# @package _global_`. It selects group choices with
`override /<group>: <choice>` and overrides values. Select it from the command line with `experiment/eval=<name>` or `experiment/train=<name>`:

```bash
python -m src.evaluation.run_eval experiment/eval=sage_gru_3_semdetect
python -m src.detector.subjectivity_aware_generation_estimator.training.gru_train experiment/train=train_sage_gru_v3_semdetect
```

Any value can still be overridden on the command line (`hardware=cpu`, `dataset.subset_size=10000`, ...).
`python -m <module> <overrides> --cfg job` prints the composed config without running anything.

### 2.3 Name-based resolution

Nothing is resolved with `_target_`, `importlib` or `getattr`. A detector or dataset is selected by its `name` field in an `if/elif`
chain: `src/detector/detector_factory.py` and `src/data/dataset_factory.py`. A new component therefore needs **both** a YAML in
`conf/detector/` or `conf/dataset/` **and** a branch in the factory (see `doc/extending.md`).

Registered detectors (`name` → class): `dummy` → `RandomBaseDetector`, `binocular` → `BinocularDetector`, `detect_gpt` → `DetectGPT`,
`fast_detect_gpt` → `FastDetectGPT`, `ada_detect_gpt` → `AdaDetectGPT`, `cristal_anchor` → `CristalAnchor`, `sage` → `Sage`.
Registered datasets: `aprdb` → `AprdbDataset`, `sem_detect` → `SemDetectDataset`. Anything else raises `ValueError`.

## 3. Environment and import-time side effects

`src/config.py` is imported by almost every module and runs at import time:

- loads `.env` (`python-dotenv`); `DATA_DIR`, `CACHE_DIR` are wrapped in `Path(os.getenv(...))`, so a missing variable raises a `TypeError`; `HF_HOME` is copied into `os.environ`;
- defines `RESULT_DIR = CACHE_DIR / "results"` (where artifacts and `results.jsonl` live, **outside the repository**);
- logs `PROJ_ROOT` through `loguru`.

Other import-time effects worth knowing before importing modules in a notebook or script:

- `src/detector/binocular.py` calls `torch.set_grad_enabled(False)` at module level, and `detector_factory` imports it, so importing the factory (hence `run_eval`) disables autograd globally.
- `cristal_anchor/cristal_anchor.py` calls `logging.basicConfig(filename='example.log', ..., level=logging.error)` at import (line 20) and creates an `OpenAI` client for the local vLLM URL (line 25, no request is sent). `level=logging.error` passes a function instead of a level, and `basicConfig` does nothing if the root logger already has handlers (`src/data/dataloader.py` calls `logging.basicConfig` at import). Whether `example.log` appears therefore depends on import order; this was not tested by running the code (audit anomaly A10).
- `src/data/dataloader.py` calls `logging.basicConfig(level=INFO)` at import.
- `cristal_anchor/finetuning_anchor.py` runs a fine-tuning at import and `cristal_anchor/script_fetch_paper.py` downloads papers and writes `data.pkl` at import: never import them.

## 4. Data layer (`src/data/`)

A dataset is a `torch.utils.data.Dataset` returning dict-like items. The evaluator relies on these keys:

| Key | Meaning |
|---|---|
| `content` | review text (string) |
| `label` | integer label; the metrics treat `1` as the positive (machine-generated) class |
| `review-id` | identifier carried to the output record (`ids`) |

`AprdbDataset` (`dataset_aprdb.py`) builds its data from the Hugging Face dataset `IntelLabs/AI-Peer-Review-Detection-Benchmark`
(CSV files per generator/conference/year). With `keep_meta: true` it also keeps `subset`, `generator`, `conference`, `year`, `file`, `PaperId`.
When `subset_size` is set, it takes half of the positives and half of the negatives (capped by what exists), shuffled with the dataset `seed`
(`conf/dataset/aprdb.yaml`), then shuffles the union. The built dataset is cached under `CACHE_DIR` (`dataset_cache/`), keyed by a hash of the config
(`get_hash()` returns `cache_id`). `SemDetectDataset` (`dataset_semdetect.py`) loads `Sem-Detect/ML_Conferences-Peer-Reviews`.
`src/data/dataloader.py` wraps a dataset in a plain `torch.utils.data.DataLoader` using `batch_size` and `num_workers` from `conf/data_loader/`; `pin_memory` is set from `torch.cuda.is_available()` and the `pin_memory` key of the YAML is ignored.

`dataset_openturingbench.py` is not in the factory but is imported directly by `run_eval.py`.

## 5. Evaluation flow (`src/evaluation/`)

`run_eval.main` does, in order:

1. Resolve the config and compute `experiment_hash = generate_config_hash(resolved)` (MD5 of the JSON-serialised resolved config, keys sorted; `eval_utils.py`).
2. **Train preset linking.** If the eval config has a `train_preset` key, `compose(config_name="train", overrides=[f"experiment/train={preset}"])` rebuilds the training config, hashes it
   with `generate_config_hash(train_cfg, exclude=["hardware"])` and injects the hash into `cfg.detector.params.model_hash` and the resolved training config into `cfg.detector.params.model_meta`.
   The training hash is appended to `experiment_hash`. This is how an evaluation finds the artifact written by training.
3. `dataset_factory(cfg.dataset)` → dataset; `data_loader(cfg.data_loader, dataset)` → loader.
4. `detector_factory(cfg.detector)` → detector.
5. `Evaluator(model, dataloader, device, type).run()` (`eval_engine.py`): for each batch it reads `batch['content']`, `batch['label']`, `batch['review-id']` and calls
   `model.predict(texts)`, or `model.predict(batch)` when the detector `type` is `"contextual"` (Cristal Anchor needs the whole batch). Scores, labels, ids and texts are concatenated.
   The decision threshold is `model.config.threshold` if present, else `0.5`.
6. `compute_metrics` (`metrics.py`) returns AUROC, TPR at 5 % and 1 % FPR, precision, recall, F1, accuracy, plus the raw `y_true` and `y_scores`.
   **Orientation.** If the raw AUROC is below 0.5 the score is treated as inverted: AUROC and TPR are computed on `-y_scores`, and the threshold comparison is flipped. Reported
   metrics therefore never show an AUROC under 0.5, and the raw scores in `results.jsonl` keep their original sign.
7. The final record is `{hash, flattened eval config, metrics, ids, content, flattened train config}` (later keys win: with a `train_preset`, the eval `name` is overwritten by the train config's `name`, e.g. `"gru"`, audit anomaly A38) and is appended as one line to `RESULT_DIR/results.jsonl` by `append_to_jsonl`.

Consequences to keep in mind:

- `compute_metrics` returns only `{"auroc": 0.0, "tpr@5%fpr": 0.0}` (different keys) when sklearn raises a `ValueError`, e.g. a single class in the data; `get_standards_metrics` also prints to stdout.
- The record contains the flattened eval config recomputed after the training config was injected, so `detector.params.model_hash` and `detector.params.model_meta.*` appear next to the `train_cfg_out` keys (redundant); the stored `hash` is computed before the injection.
- `train_preset` is meaningful only for `sage`: it injects `model_hash` into `detector.params` without checking the detector.
- Lines can be very large because they embed `y_scores`, `ids` and every review text (the recorded `results.jsonl` line 42 alone is about 30 MB).
- Writes are plain appends with no lock: concurrent runs can interleave and corrupt a line.
- The hash is a function of the resolved config only, not of the data or of the code. Runs with non-deterministic detectors (`dummy`, Cristal Anchor without seed) share a hash and differ in result.
- The `eval` hash includes the `hardware` block (only the train hash excludes it).
- MLflow is not used: it was removed from the project (results go only to `results.jsonl`).

## 6. Detector contract (`src/detector/`)

`BaseDetector` (`base.py`) takes a `config` mapping and requires one method, `predict(texts)`, returning one score per text. The base class does not fix the score direction (for Binoculars, for instance, low scores mean machine-generated);
the evaluator's automatic inversion (section 5) handles it. `get_meta()` is an optional hook returning `None`. Each detector YAML (`conf/detector/<name>.yaml`) has `name`, `type`
(`statistical`, `linguistic` or `contextual`), optional `trainable`, and a `params` block read by the class through `config['params']`.

| Detector | Package / file | Type | Notes |
|---|---|---|---|
| `dummy` | `random_base_detector.py` | statistical | random scores; used for smoke runs |
| `binocular` | `binocular.py` | statistical | observer `tiiuae/falcon-7b` and performer `tiiuae/falcon-7b-instruct` (`conf/detector/binocular.yaml`) |
| `detect_gpt`, `fast_detect_gpt` | `detect_gpt.py`, `fast_detect_gpt.py` | statistical | perturbation / sampling-discrepancy detectors |
| `ada_detect_gpt` | `ada_detect_gpt/` | statistical | optional learned weight function: for `w_func` = `bspline` or `bspline_theory` a cache `CACHE_DIR/ada_detect_gpt_wfunc_weight_<hash>.pth` is read or written and a `train_dataset` is required; the default `identity` has no cache |
| `cristal_anchor` | `cristal_anchor/` | contextual | sentence-transformer embeddings + LLM through the local vLLM server (`127.0.0.1:8000`); receives the full batch and reads `batch['content']` and `batch['pdf_path']`, so the dataset must provide PDF paths |
| `sage` | `subjectivity_aware_generation_estimator/` | linguistic | trained model loaded from an artifact |

Per-detector pages will be added under `doc/detector/` (the folder that already holds `binocular.md` and `ada_detect_gpt.md`).

## 7. SAGE: training, artifacts, evaluation

Package `src/detector/subjectivity_aware_generation_estimator/`:

- `training/linguistic_segmentation.py` segments reviews into sentences and classifies each one (speech-act category and subjectivity score) with local RoBERTa models.
  The model directory (`SAGE_MODELS_DIR`) and the segmentation GPU (`SAGE_SEGMENTATION_DEVICE`) come from environment variables with the defaults described in `doc/detector/sage.md`.
- `features/feature_spec.py` defines named, hashable feature specifications (`SPECS`, `FeatureSpec`) that fix the column layout; `features/tensor_builder.py` and `pairing.py` build sequences and
  pair/balance the data. `n_features` in a training preset must equal the spec's `input_dim`, otherwise `train_gru` raises.
- `model/SimpleGRU.py`, `model/SubjTransformer.py`; MLP is a scikit-learn pipeline (`training/mlp_train.py`).
- Training (`gru_train.train_gru`): hash = `generate_config_hash(cfg, exclude=["hardware"])`, target directory `RESULT_DIR/<hash>/`, seed set from `training.model.seed`, data prepared, GRU trained, validation metrics printed, then `artifact.joblib`
  (`state_dict` on CPU, `model_class: "SimpleGRU"`, `model_kwargs`, `feature_spec`, `ctx` with normalisation statistics, the resolved training `cfg`) and `gru_result.npz` written.
- Evaluation (`sage.py`): `Sage.build_model` picks `SageMLP` or `SageGRU` from `model_meta.name`, which is the root `name:` of the train preset (`mlp` or `gru`). Any other value, including `transformer`, raises `ValueError`, so Transformer artifacts cannot be evaluated. `SageGRU` supports three artifact generations
  (legacy `model` pickle; intermediate and current `state_dict` formats) and, for the spec formats, re-validates the saved `feature_spec` and `feature_version` against the preset, and the model `n_features` against the spec's `input_dim`, before loading.
  `SageGRU.evaluate` segments the texts, builds features with the training-time normalisation (`ctx`) and returns one probability P(machine) per review, realigned to the input order (it raises instead of misaligning). `SageMLP.evaluate` returns `self.pipe.predict(...)` of the scikit-learn pipeline; check the final estimator before assuming it yields probabilities.
  Side effect: with the default `save_segmented=True`, `SageGRU` appends the segmented texts to `RESULT_DIR/evaluation_batch/evaluation_segmented_txt.jsonl` on every evaluation.

Artifact compatibility constraints: GRU artifacts rebuild the model from the string `"SimpleGRU"` (`model/artifact_io.py`); Transformer artifacts pickle the whole `SubjectivityTransformer` object; MLP artifacts pickle a scikit-learn `Pipeline`
and `SageMLP` asserts the scikit-learn version matches. Keep these class names and import paths unchanged, or existing artifacts stop loading.

## 8. Directory map

```
conf/                  Hydra configs: eval.yaml, train.yaml, groups (hardware, dataset, data_loader, detector, training/*), presets (experiment/{eval,train})
src/config.py          paths, .env (import-time side effects)
src/data/              datasets, factory, dataloader
src/detector/          base class, factory, baselines, and packages ada_detect_gpt/, cristal_anchor/, sem_detect/, subjectivity_aware_generation_estimator/
src/evaluation/        run_eval.py (entry), eval_engine.py (Evaluator), eval_utils.py (hash, flatten, jsonl), metrics.py
src/utils/             vllm_manager.py (tmux-based vLLM lifecycle)
notebook/, data/       exploratory and result notebooks, dataset-preparation notebooks (do not execute the result notebooks)
```
