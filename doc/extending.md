# Extending the codebase

Recipes for adding a detector, a dataset, a SAGE feature or a preset, with the constraints that are easy to miss.
Read `doc/architecture.md` first (configuration, evaluation flow, artifacts). Commands are run from the repository root with the `.venv` interpreter.
This page was derived from the code; none of the recipes below was executed while writing it, so test each one with `--cfg job` before a real run.

## 1. Check what you changed without running anything

`python -m <entry module> <overrides> --cfg job` prints the composed Hydra config and exits. For example:

```bash
python -m src.evaluation.run_eval experiment/eval=sage_gru_3_semdetect --cfg job
python -m scripts.check_structure --compare scripts/check_structure_baseline.json   # imports, presets, --cfg job launches; no GPU, no training
```

`scripts/check_structure.py` exits with code 1 on a regression compared to the recorded baseline (26 expected failures are stored in `scripts/check_structure_baseline.json`).

## 2. Add a detector

A detector needs three things: a class, a YAML and a branch in the factory.

### 2.1 The class

Subclass `BaseDetector` (`src/detector/base.py`). The constructor receives the `detector` section of the config (an OmegaConf mapping with `name`, `type`, `params`).
`predict(texts)` receives a list of strings (a list of one batch) and must return one score per text, as a list or array that `np.concatenate(...).ravel()` can flatten into shape `(N,)`.
Look at `src/detector/random_base_detector.py` for the smallest example (it returns `sigmoid(randn)` values as a list of one-element lists). Never filter out texts and never return `None`: `BinocularDetector.predict` skips empty texts and can return `None`, which breaks the alignment with the labels.

Notes:

- The score direction is free. If the raw AUROC is below 0.5, the evaluator evaluates `-score` and flips the threshold comparison. Do not invert scores yourself to "help".
- `Evaluator.run` reads `self.model.config.threshold` (the `config` attribute of your detector object) instead of the default `0.5` for precision/recall/F1/accuracy. Only the Binoculars presets define one, under `detector.params.threshold`, and it works because `BinocularDetector.__init__` stores `self.config = config['params']`. A detector that keeps the root config in `self.config` (the `BaseDetector` default, `Sage`) silently ignores `params.threshold`.
- If the detector needs more than the review text (like Cristal Anchor, which reads `batch['pdf_path']`), set `type: "contextual"` in the YAML: the evaluator then calls `predict(batch)` with the whole batch dictionary instead of `predict(texts)`. `pdf_path` only exists when the dataset is built with `with_pdf: true` (APRDB) or `with_paper: true` (Sem-Detect).
- The evaluator wraps inference in `torch.no_grad()`, but `RandomBaseDetector` also moves itself to `config.get("device")` inside `forward`; if your detector uses a GPU, take the device from `params` (see how `conf/detector/sage.yaml` passes `device: ${hardware.device}`).
- Avoid import-time work in the module: `detector_factory` imports every detector module, so anything executed at import runs for every evaluation (see `doc/architecture.md`, section 3).

### 2.2 The YAML

Create `conf/detector/<name>.yaml`. Copy the shape of an existing one, e.g. `conf/detector/binocular.yaml`:

```yaml
name: "my_detector"        # the string tested in detector_factory
type: "statistical"        # statistical | linguistic | contextual (only "contextual" changes the evaluator behaviour)
trainable: false
params:
  some_option: 1
  device: ${hardware.device}
  seed: ${seed}
```

Values placed under `params` are read by your class through `config['params']`. Use interpolations (`${hardware.device}`, `${seed}`) rather than copying values.

### 2.3 The factory branch

Add an import and an `elif name == 'my_detector':` branch to `src/detector/detector_factory.py`. There is no automatic registration: without the branch the factory raises `ValueError('Unknown detector ...')`.

### 2.4 A preset to run it

Create `conf/experiment/eval/<preset>.yaml`:

```yaml
# @package _global_
defaults:
  - override /detector: my_detector
  - override /hardware: single_gpu
  - override /dataset: aprdb
  - override /data_loader: default

hardware:
  device: "cuda:0"
data_loader:
  batch_size: 8
dataset:
  subset_size: 100
name: "my_detector_small"
```

Run it with `python -m src.evaluation.run_eval experiment/eval=<preset>`. One line is appended to `RESULT_DIR/results.jsonl`.

Hash implications (they matter for reproducibility): the stored `hash` is the MD5 of the resolved eval config, **including `hardware` and `name`**; when the preset has a `train_preset`, the stored value is that hash followed by the 32-character training hash (64 characters). Changing the preset `name`, the device or any value changes the hash;
copy-pasted `name` values in existing presets are an audit finding (A2), so give each preset a unique and accurate `name`.

## 3. Add a dataset

1. Write a `torch.utils.data.Dataset` in `src/data/`. Each item must provide at least `content` (text), `label` (int, `1` is the positive class) and `review-id` (an identifier; `Evaluator` concatenates it with NumPy, so keep it a string).
   Add any metadata columns you want in the output only if the code that consumes them expects them (SAGE training uses `generator`, `conference`, `year`; Cristal Anchor uses `pdf_path`). `AprdbDataset` only keeps `review-id` and the metadata when `keep_meta: true`; without it the evaluator raises a `KeyError`. `label` must survive the `DataLoader` collate, because the evaluator calls `batch['label'].to(device)`.
   Provide `get_hash()`, `get_config()` and `to_pandas()` like `AprdbDataset` and `SemDetectDataset`: `run_eval` logs `get_hash()`, and the SAGE segmentation (`training/linguistic_segmentation.py`) calls all three, so any dataset used for SAGE training needs them.
2. The constructor signature used by the factory is `Dataset(config=config, **kwargs)`, where `config` is the Hydra node.
3. Create `conf/dataset/<name>.yaml` with a `name:` key.
4. Add a branch in `src/data/dataset_factory.py`.
5. Caching pattern: both existing datasets hash their config (`generate_config_hash`) and cache the built dataset under `CACHE_DIR` (`dataset_cache/` for APRDB). If your dataset reads the network, pin the revision in the YAML as
   `sem_detect.yaml` does (`revision: ...`); the APRDB loader does not pin one (audit anomaly A22).

Remember that `train.yaml` instantiates the dataset group twice (`dataset_train`, `dataset_val`): a new dataset must also work under that package redirection. On the command line this is `dataset@dataset_train=my_dataset`; inside a preset it is `override /dataset@dataset_train: my_dataset`. The same `conf/dataset/<name>.yaml` is used by `eval.yaml`, and for training it must provide the keys `gru_train` reads (`split`, `seed`, `subset_size`, `with_pdf`, ...).

Existing label semantics to reuse or imitate: `SemDetectDataset` has `label_mode` (`human_vs_ai`, `ai_vs_rest`, `human_vs_llm`, `three_class`) and `balance` (`uniform`, `none`, `paired`); `AprdbDataset` is binary with `1` = machine-generated (`doc/dataset/ai-peer-review-detection-benchmark.md` states 0 = human and 1 = machine-generated, but the upstream dataset card does not document the column).

## 4. Train and evaluate a SAGE model

### 4.1 Training preset

Create `conf/experiment/train/<preset>.yaml`. Use `train_sage_gru_v3.yaml` as the template. The elements that are checked:

- `override /training/model: sage_gru_1` (or `sage_mlp`, `sage_gru_2`, `sage_transformer_2`) selects the model family defaults; the root `name:` of the preset (`"gru"`, `"mlp"`, `"transformer"`) is what `Sage.build_model` reads at evaluation time.
- `training.model.feature_spec` names an entry of `SPECS` in `features/feature_spec.py`; `training.model.feature_version.{segmentation, subj_category}` must match that spec. This is only checked at evaluation (`SageGRU` raises if they disagree); `gru_train` takes the segmentation from the spec and does not read `feature_version`, but it is part of the hash.
- `training.model.params_model.n_features` is optional but, if given, must equal the spec's `input_dim`, otherwise `train_gru` raises (it is then replaced by `input_dim`).
- `dataset_train` and `dataset_val` select the data (`split: calibration` / `test` for APRDB). The weight-initialisation seed is `training.model.seed` (default 0), but pairing and balancing also depend on `dataset_train.seed` and `dataset_val.seed`: changing them changes the data and the hash (audit anomaly A11). These rules on `feature_spec` and seeds apply to the GRU (and partly Transformer) trainers; `mlp_train.py` uses neither a seed nor `feature_spec`.

Train with the module that matches the model family:

```bash
python -m src.detector.subjectivity_aware_generation_estimator.training.gru_train experiment/train=<preset>
```

(`mlp_train` and `transformer_train` are siblings; both use the same `train.yaml`.) The artifact goes to `RESULT_DIR/<hash>/artifact.joblib`, where
`<hash> = generate_config_hash(cfg, exclude=["hardware"])`. **Re-running with an identical config overwrites the previous artifact** (audit anomaly A11): copy the directory first if you need to keep it.
Training requires the segmentation models and GPUs described in `doc/architecture.md`, section 7.

### 4.2 Evaluation preset

Create `conf/experiment/eval/<preset>.yaml` as in section 2.4 with `override /detector: sage` and:

```yaml
train_preset: "<name of the train preset>"
```

`run_eval` recomposes `train.yaml` with that preset, hashes it, and passes the hash to `Sage`, which loads `RESULT_DIR/<hash>/artifact.joblib`. Only the exact training config reproduces the hash: editing the train preset after training changes the hash
and the artifact is no longer found. Only GRU and MLP models can be evaluated (`Sage.build_model`).

### 4.3 Add a feature (columns) to SAGE

Features are described in `features/feature_spec.py`:

1. A **builder** `(block, ctx, group)` maps one review block (a DataFrame with one row per sentence) to an array of shape `(L, group.width)`. Register it with `@builder("name")`. It must use only the block, the `ctx` dictionary and the group
   (normalisation statistics fitted on the training set by `tensor_builder.fit_context`); recomputing statistics inside a builder leaks information between train and test.
2. A **`FeatureGroup`** (`name`, `builder`, `columns`, `scope`, `requires`, `ablation_value`, and encoding parameters `merge` / `merge_into`, which only the `speech_act_merged` builder accepts) declares the columns and the DataFrame columns they require (`requires`).
   Encoding parameters must live in the group, not in `ctx`, otherwise two experimental conditions share the same `spec.hash()`.
3. A **`FeatureSpec`** assembles groups, fixes the segmentation version (`1` or `2`; the label vocabulary `label_cats` is then constrained to `LABEL_CATS_1` or `LABEL_CATS_2`), and is registered in `SPECS`. `input_dim` is derived from the groups.
4. Create a new spec rather than editing an existing one: `spec.hash()` is stored in artifacts, and `SageGRU._resolve_spec` checks it when loading (by name if the hash matches, otherwise by structure).
   Editing a spec in place breaks loading by name; `_resolve_spec` then falls back to a structural match, which only succeeds if exactly one registered spec matches.
5. Set `training.model.feature_spec` and `params_model.n_features` in a new train preset, and retrain.

Segmentation (sentence splitting and speech-act/subjectivity classification) is expensive and cached on disk; the cache key combines the dataset hash (`subset_size`, `split`, `seed`, ...) with a hash of `spec.segmentation_key()` (segmentation version and `subj_category`), so changing only the column encoding reuses the cache. The segmentation hard-codes `cuda:1` (`linguistic_segmentation.py:221, 588`), which requires two GPUs whatever `hardware` says.

### 4.4 Add a model architecture

- GRU artifacts are rebuilt from `model_class` through the registry `_MODEL_CLASSES` in `model/artifact_io.py`. A new model class must be registered there and saved with its `model_kwargs` and `state_dict`.
- Do not rename the class `SimpleGRU` / `SubjectivityTransformer` or move the modules `model/SimpleGRU.py` / `model/SubjTransformer.py`: existing artifacts refer to them (current GRU by the string `"SimpleGRU"`; Transformer and legacy GRU by pickling the object).
- To make a new family evaluable, add a branch to `Sage.build_model` and an evaluation class (see `SageGRU`, `SageMLP`).

## 5. Pitfalls checklist

- Importing `src.*` requires a valid `.env` defining `DATA_DIR`, `CACHE_DIR` and `HF_HOME`; `src/config.py` wraps them in `Path(os.getenv(...))`, so a missing one fails at import. `scripts.check_structure` also needs a valid `.env`.
- Never import `cristal_anchor/finetuning_anchor.py` or `cristal_anchor/script_fetch_paper.py` (they run work at import).
- `xai/direction.py` and the folder `xai/direction/` share a name; Python resolves the module. The folder uses relative imports and several broken absolute imports, and is not reachable from any notebook or `src/` module. Do not rename it.
- Hydra presets that share a `name:` produce records with misleading names in `results.jsonl`; set a unique `name:`.
- `results.jsonl` is appended without locking: do not run several evaluations writing to the same `RESULT_DIR` at the same time (audit anomaly A15 documents a corrupted line caused by concurrent writes).
- Evaluation of non-deterministic detectors (`dummy`, Cristal Anchor calls to vLLM without a seed) gives different results for the same hash.
- The code base has no automated tests; use `--cfg job` and `scripts.check_structure` after every change.
