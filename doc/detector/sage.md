# SAGE (Subjectivity-Aware Generation Estimator)

SAGE is the project's own detector. It does not look at token probabilities: it describes each review as a sequence of sentences, each sentence being reduced to a few linguistic features (speech-act category and, for some categories, a subjectivity score), and a small sequence model classifies the sequence as human-written or machine-generated.

Code: `src/detector/subjectivity_aware_generation_estimator/`. Detector name in the factory: `sage` (`conf/detector/sage.yaml`, `type: "linguistic"`).
Read `doc/architecture.md` (evaluation flow, hashes, artifacts) and `doc/extending.md` (adding presets and features) alongside this page.

## 1. Pipeline

```
review text
  └─ sentence splitting                          pysbd, English        (linguistic_utils.segment_text)
      └─ speech-act label per sentence           rules (segmentation 1) or RoBERTa classifier (segmentation 2)
          └─ subjectivity score 1-5              RoBERTa regressor, only for sentences whose label is in `subj_category`
              └─ feature columns per sentence    FeatureSpec groups (one-hot speech act, z-scored subjectivity, ...)
                  └─ sequence model              GRU (main), Transformer or MLP variants
                      └─ one probability per review  (P(machine-generated))
```

### 1.1 Segmentation (`training/linguistic_segmentation.py`)

- `explode_into_sentences` splits `content` with pysbd and drops empty sentences; one row per sentence, original columns duplicated.
- Speech acts are stored in the column `rule_based_SA-label` (the name is kept for both versions):
  - **segmentation 1**: rule-based `classify()` with categories `Assertion`, `Neither`, `Question` (`LABEL_CATS_1`).
  - **segmentation 2**: a local sequence classifier (`roberta_speech_act_classifier`) with categories `Assertion`, `Expressive`, `Neither`, `Question`, `Suggestion` (`LABEL_CATS_2`).
- Subjectivity: a local regression model (`roberta_subjectivity_regressor`) scores the sentences whose speech-act label is in `subj_category`. The raw output is mapped back to the 1-5 scale (`raw * 4 + 1`) and written to `raw_prediction_1_5`, with `rounded_prediction` and `predicted_label` (`WORLD`, `WORLD+VALUES`, `VALUES`, `TASTE+VALUES`, `TASTE`). Other sentences keep a missing value.
- `segment_database_spec(dataset, spec)` segments a whole dataset and caches the result as `RESULT_DIR/implicite_segmentation/<dataset_hash>_<key>/sentences_degrees_final.csv`, where `<key>` is a short hash of `spec.segmentation_key()`. It also appends a line to `implicite_segmentation/mapping_hash_config.json`.
- `segment_texts(texts, segmentation, subj_category)` is the inference path: it numbers the texts (`review-id` = position) and runs the same pipeline.

Hard constraints of this module (important for anyone trying to run SAGE elsewhere):

- The model directory is `MODELS_ROOT`: the environment variable `SAGE_MODELS_DIR`, else `<repo>/models/` (installed by `python -m scripts.fetch_sage_models`). Models are loaded with `local_files_only=True`. The training code of these models is not in the repository and their source is not documented (audit anomaly A16).
- The segmentation device is the module constant `SEGMENTATION_DEVICE`, independent of `hardware.device`: the variable `SAGE_SEGMENTATION_DEVICE`, else `cuda:1` with two GPUs, `cuda:0` with one, `cpu` without GPU. Running on CPU or one GPU is therefore possible but untested.
- Several generations of functions coexist. `segment_dataframe_` (called by the spec-based path, with `segmentation_df_v1_optimized` for version 1) and `segment_database_` are live: `gru_train` and `transformer_train` import them. `segment_database` (used by `mlp_train`), `segment_dataframe` and `segmentation_df_v2` are older; `segmentation_df_v2` concatenates a `str` path with `/` and would raise a `TypeError` if reached (anomaly A23). `segment_texts_spec` has no caller (`sage.py` imports `segment_texts`).
- `SEGMENTATION_DEVICE` is passed by `segment_database_`, `segment_database`, `segment_texts`, `segment_database_spec` and `segment_texts_spec`.
- `mapping_hash_config.json` (JSON lines despite the extension) is only appended to when a segmentation is computed, not when it is loaded from the cache.

## 2. Features (`features/`)

- `feature_spec.py` defines `FeatureGroup` (a block of columns produced by one registered builder), `FeatureSpec` (segmentation version, scored categories, label vocabulary, ordered groups) and the registry `SPECS`. `spec.input_dim` is the sum of group widths and must equal `params_model.n_features` in a training preset. `spec.hash()` identifies the encoding and is stored in artifacts.
- Builders registered in the file: `speech_act_onehot`, `subjectivity_zscore`, `subjectivity_valid`, `speech_act_merged`.

Registered specs:

| Spec | Segmentation | Scored categories | Groups | `input_dim` |
|---|---|---|---|---|
| `v1_base` | 1 | Assertion | one-hot (3) + subjectivity z (1) + valid (1) | 5 |
| `v2_assertion` | 2 | Assertion | one-hot (5) + z + valid | 7 |
| `v2_multi` | 2 | Assertion, Expressive, Question, Suggestion | one-hot (5) + z + valid | 7 |
| `v3_multi_novalid` | 2 | same four | one-hot (5) + z | 6 |
| `v3_merge_<cat>` (`expressive`, `neither`, `question`, `suggestion`) | 2 | same four | merged one-hot (width kept at 5: `<cat>` is encoded as `Assertion`, its own column stays constantly zero) + z | 6 |
| `v3_cat_only` | 2 | same four | one-hot only | 5 |
| `v3_subj_only` | 2 | same four | z only | 1 |

(`input_dim` values follow from the group widths in `feature_spec.py`.) The `v3_merge_*`, `v3_cat_only` and `v3_subj_only` specs exist for the ablation experiments. Which spec each published model used will be listed in `doc/reproducing-results.md` (to be written).

- `tensor_builder.py`: `fit_context(df_train, spec)` estimates `subj_mean` and `subj_std` on the training set only; `block_to_matrix` builds the `(L, input_dim)` matrix of one review; `build_dataset` returns padded tensors, lengths, labels and ids `(X, L, Y, ids)` (blocks sorted by label), while `build_sequences` (inference) returns `(X, L, ids)` without labels. Blocks are grouped by `[review-id, generator]` when a `generator` column exists.
- `pairing.py`: `pair_and_balance(df, seed)` keeps, for each `review-id` present on both sides, one human block and one machine block, the generator being chosen deterministically (MD5 of seed and id). This gives exact class balance and avoids mixing sentences of different generators in one sequence. It assumes a `review-id` shared by the human and machine versions, which is specific to APRDB; datasets already paired upstream (`balance: paired`, Sem-Detect) skip it.

## 2.5 Transformer models used by SAGE

SAGE involves four distinct kinds of neural or learned components. Only the first three are Transformers.

| Component | Role | Architecture (from `config.json` / code) | Where it lives |
|---|---|---|---|
| Speech-act classifier (`roberta_speech_act_classifier`) | segmentation version 2: one label per sentence | `RobertaForSequenceClassification`, 12 layers, hidden size 768, vocabulary 50 265, single-label classification with 5 classes `Assertion`, `Expressive`, `Neither`, `Question`, `Suggestion`; saved with `transformers` 5.12.1 | `<SAGE_MODELS_DIR>/roberta_speech_act_classifier/` (`model.safetensors`, `tokenizer.json`, `training_args.bin`) |
| Subjectivity regressor (`roberta_subjectivity_regressor`) | scores a sentence from 1 (WORLD) to 5 (TASTE) | `RobertaForSequenceClassification` with `problem_type: regression` and one output, same size as above; saved with `transformers` 5.0.0; the training target was normalised as `(y - 1) / 4` and `denormalize_to_1_5` inverts it | `<SAGE_MODELS_DIR>/roberta_subjectivity_regressor/` |
| `SubjectivityTransformer` (`model/SubjTransformer.py`) | one of the three sequence classifiers over the sentence features | pre-LN Transformer encoder (see below) | trained by `training/transformer_train.py` |
| Rules-based classifier (`classify`) and `SimpleGRU` | segmentation version 1 / default sequence model | not Transformers | `linguistic_segmentation.py`, `model/SimpleGRU.py` |

**Directory names.** The two models were originally stored as `final_model_definitivo` (the subjectivity regressor) and `final_model_definitivov2` (the speech-act classifier); the `v2` was misleading, since they are two different models and not two versions of one. On 2026-10-20 the directories were renamed to `roberta_subjectivity_regressor` and `roberta_speech_act_classifier`, with symbolic links at the old names (needed by two obsolete notebooks under `notebook/linguistic/` that still use the old path), and `linguistic_segmentation.py` was updated (`SUBJECTIVITY_MODEL_DIR`, `SPEECH_ACT_MODEL_DIR` and the strings of the older functions). The roles are set in that file (lines 31-32) and confirmed by the models' own `config.json` (regression with one output; single-label classification with the five speech-act labels). "Segmentation version 2" is unrelated to the old `v2` suffix: segmentation 1 uses the rules plus the subjectivity regressor; segmentation 2 uses the speech-act classifier plus the subjectivity regressor.

About the two RoBERTa models:

- Both are fine-tuned RoBERTa-base-sized encoders (hidden size 768, 12 layers, 514 positions). Their `config.json` has no `_name_or_path`, so the exact checkpoint they were fine-tuned from, the training data and the training code are **not recorded** in the repository (audit anomaly A16). "RoBERTa-base" is an inference from the dimensions.
- They are loaded with `AutoTokenizer` / `AutoModelForSequenceClassification` and `local_files_only=True`, so no network access is involved but the directories must exist. They run on `SEGMENTATION_DEVICE`.
- Speech acts are predicted in batches of 1024 sentences (tokenizer `max_length=512`, `truncation=True`, argmax of the logits). Subjectivity is predicted in batches of 512 (`score_subjectivity_optimized`, tokenizer with `truncation=True` and no explicit `max_length`), and only for sentences whose speech-act label is in `subj_category`.
- Sentence splitting uses pysbd (English), not a Transformer.
- The two models are reloaded each time a segmentation function is called (`load_sequence_classifier` is called inside `label_speech_acts_model` and `score_subjectivity*`), so every `segment_texts` call (every evaluation batch) pays the loading cost.

`SubjectivityTransformer` (the sequence model of the Transformer variant, `model/SubjTransformer.py`):

| Element | Behaviour |
|---|---|
| Input | `(B, L, input_dim)` sentence features plus the real lengths; `input_dim` is the number of speech-act categories plus 2 (subjectivity z-score and validity flag): 7 with segmentation 2 (five categories, the class default), 5 with segmentation 1 (three categories), which is what the preset `sage_transformer_2` selects (`feature_version.segmentation: 1`) |
| Projection | `Linear(input_dim, d_model)` + `LayerNorm`, then input dropout |
| Positional encoding | `pos_encoding` = `sinusoidal` (default; a fixed table of `max_len + 1` positions, so a longer sequence fails despite what the code comment suggests; the trainer sets `max_len` to the longest train/validation sequence), `learned` (raises if `L` exceeds `max_len + 1`) or `none` (set encoder, permutation-invariant) |
| Encoder | `nn.TransformerEncoder`, `num_layers` pre-LN layers (`norm_first=True`), `nhead` heads, feed-forward width `ff_mult * d_model`, GELU, final `LayerNorm`; `d_model` must be divisible by `nhead` |
| Pooling | `attention` (additive pooling whose weights can be returned with `return_attention=True` as an explanation trace), `mean` (masked mean) or `cls` (learned token) |
| Output | one logit per review (`BCEWithLogitsLoss`) |
| Defaults / preset | class defaults `d_model=64, nhead=4, num_layers=2`; `conf/training/model/sage_transformer_2.yaml` uses `d_model=16, nhead=2, num_layers=1, ff_mult=4, dropout=0.1, pooling=attention, pos_encoding=sinusoidal`, trained for 30 epochs with `lr=1e-3`, `weight_decay=0.01`, `warmup_ratio=0.1`, `grad_clip=1.0`, `patience: null`; plus `batch_size: 64`, `seed: 0`, `dev_frac: 0.0`, `input_dropout: 0.0` |

Remember that Transformer artifacts cannot be evaluated through `Sage` (section 4) and that `transformer_train.py` depends on the statically broken `training/gru_features.py` (section 7).

## 3. Models and training

All trainers are Hydra entry points with `config_name="train"` and write to `RESULT_DIR/<hash>/` where `hash = generate_config_hash(cfg, exclude=["hardware"])`.

**Status of the Transformer variant.** `SubjectivityTransformer` was used to experiment with a different sequence model **in place of the GRU** on the same sentence features. It does not appear in the article for lack of space (statement of the project author), but it is a real part of SAGE and not dead code: three presets (`train_sage_transformer_2_cat`, `_2`, `_3`) and four trained artifacts exist. The artifacts, all written on 2026-09-10 with segmentation 2 (`input_dim` 7), recorded these validation AUCs on the APRDB test subset (`transformer_result_1.npz`): d_model 16, attention pooling: 0.9558 and 0.9463; d_model 64, attention pooling: 0.9607; d_model 16, mean pooling: 0.9599. For comparison, the GRU of `train_sage_gru_v3` (6 columns, a different spec) reached 0.9549. Caveats: two of the four runs selected the best epoch on the validation set itself (audit anomaly A36), the specs differ between the GRU and Transformer runs, and the artifact hashes (`5b017aae…`, `836ecf3d…`, `9ae6c370…`, `ec202c37…`) no longer match what the current presets compose (`c8bf3412…`, `a666f98b…`, `9cc9490b…`), so these runs cannot be reproduced by hash. Differences between the presets: epochs 30 (`_cat`) or 100 (`_cat_2`, `_cat_3`), attention pooling (`_cat`, `_cat_2`) or mean pooling (`_cat_3`); all with `d_model: 16, nhead: 2, num_layers: 1`. The Transformer artifacts cannot be evaluated through `run_eval` (`Sage.build_model` has no branch for them), so their only recorded results are these validation numbers.

| Variant | Trainer | Model | Artifact content | Evaluable by `Sage`? |
|---|---|---|---|---|
| GRU (main) | `training/gru_train.py` | `model/SimpleGRU.py`: unidirectional GRU (`n_layers`, default 1), last hidden state of the last layer, linear head; packed sequences | `state_dict`, `model_class: "SimpleGRU"`, `model_kwargs`, `feature_spec`, `ctx`, `cfg` (resolved YAML string), `hash`, `seed`, `n_outputs`, `auc`, `metrics`, `torch_version`; plus `gru_result.npz` | yes (`SageGRU`) |
| Transformer | `training/transformer_train.py` | `model/SubjTransformer.py` (`SubjectivityTransformer`): encoder with sinusoidal, learned or no positional encoding and attention, mean or cls pooling | `hash`, `cfg`, pickled `model`, `model_class: "SubjectivityTransformer"`, `model_kwargs`, `state_dict`, `subj_mean`, `subj_std`, `seed`, `history`, `torch_version`, and a `feature_spec` dict with other keys than the GRU one (`segmentation_version`, `label_cats`, `columns`, `input_dim`, `max_len_train`; no `name` or `spec_hash`); also writes `transformer_result_1.npz` and `roc_transformer.png` | no (`Sage.build_model` raises `ValueError` for `transformer`) |
| MLP | `training/mlp_train.py` | scikit-learn `Pipeline(StandardScaler, MLPClassifier)` on 8 aggregate features per review (proportions of the 3 speech acts and of the 5 subjectivity levels) | `pipeline`, `sklearn_version`, `cfg`, `hash`; plus `mlp_result.npz` | yes (`SageMLP`) |

Training details of the GRU trainer (`train_gru`):

1. `prepare_data`: for `dataset_train` and `dataset_val`, load the dataset through `dataset_factory`, segment it (`segment_database_spec`), then `pair_and_balance` (unless `balance == "paired"`). `_check_val_split` raises if, for two datasets with the same `name`, `val_fraction` or `val_seed` differ, or if `val_fraction` is set and both use the same `split`.
2. `fit_context` on the training blocks; `build_dataset` for both sets with the training context.
3. Model hyper-parameters come from `training.model.params_model` (`hidden`, `n_layers`, optional `n_outputs`); `n_features` is optional in the preset; if declared and different from `spec.input_dim`, `train_gru` raises, and it is then set to `spec.input_dim`. `n_outputs = 1` (binary, `BCEWithLogitsLoss`) or `K > 1` (cross-entropy, e.g. the three-class Sem-Detect encoding).
4. Optimiser: Adam with `lr` and `epochs` from `params_training`; no class weighting (the data is balanced).
5. Validation: for a binary model prints the classification report, AUC, per-conference/year AUC and per-generator AUC; for `n_outputs > 1` it prints the macro-F1, the OvR macro AUC and the confusion matrix. Then it saves the artifact and `gru_result.npz` (`y_true`, `y_proba`, conference, year, generator). No preset in `conf/` sets `n_outputs`, and `SageGRU._forward` always applies a sigmoid, so multi-class artifacts are not evaluable through `Sage`.

The Transformer and MLP trainers are older code paths. The Transformer trainer passes the validation loader as `dev_dl` and restores the epoch with the best validation AUC, so its reported validation AUC is optimistic (audit anomaly A36). It also relies on `training/gru_features.py`, which is statically broken (anomaly A35).
The other details: the MLP trainer uses `AprdbDataset` and `segment_database` directly (rule-based segmentation, no `feature_spec`, no seed argument other than the sklearn `random_state` in its YAML), and the Transformer trainer builds its tensors with `build_split_tensors` and a fixed layout (label categories plus `subj_z` and `subj_valid`).

Reproducibility notes: `set_seed` seeds Python, NumPy and PyTorch, but there is no `cudnn.deterministic` or `use_deterministic_algorithms`, so GPU training is not guaranteed bit-for-bit identical (anomaly A21). Retraining with an identical config overwrites the artifact (anomaly A11).

## 4. Evaluation (`sage.py`)

`Sage(config)` reads `config['params']`: `model_hash` and `model_meta` are injected by `run_eval` from the `train_preset` of the eval preset. `build_model` returns a `SageMLP` or `SageGRU` according to `model_meta.name`.

- `SageGRU` loads `RESULT_DIR/<hash>/artifact.joblib` and has code for three artifact generations (legacy pickled model with `subj_mean`/`subj_std`; intermediate and current `state_dict` formats). The legacy path goes through `training/gru_features.build_features_mono`, which is statically broken (see section 7), so only the two `state_dict` generations should be considered evaluable. For the intermediate generation (no `ctx` saved) `_rebuild_ctx` re-segments the original training set and re-pairs it, which needs the data, the local models and a GPU. For the current format it validates: the declared `feature_spec` and `feature_version` of the preset against the artifact, `ctx.label_cats` against the spec, and the model `n_features` against `spec.input_dim`. It resolves the spec by name if the saved `spec_hash` matches, otherwise by structure.
- `evaluate(texts)`: segmentation, then `build_sequences` with the training `ctx`, a forward pass in batches of 64 and `sigmoid` to give P(machine-generated) per review. In the spec path scores are realigned to the input order by `review-id` (`_align`); it raises instead of returning misaligned scores if a text has no sentence, or ids are duplicated or out of range. The legacy path and `SageMLP` have no such protection.
- Side effect: with the default `save_segmented=True`, every call appends the segmented sentences to `RESULT_DIR/evaluation_batch/evaluation_segmented_txt.jsonl`.
- `SageMLP.evaluate` returns `pipe.predict(...)`, which for an `MLPClassifier` is the class label (0 or 1), not a probability. Evaluating an MLP artifact through `run_eval` therefore feeds hard labels to the metrics (the AUROC then reduces to a single operating point).

Presets to look at: `conf/experiment/train/train_sage_gru_v3.yaml`, `train_sage_gru_v3_semdetect.yaml`, and the eval preset `conf/experiment/eval/sage_gru_3_semdetect.yaml` (its `train_preset` is `train_sage_gru_v3`).

## 5. Explainability (`xai/`)

Modules used by `tests/results_xai.ipynb`, which produces the article's XAI results (`sage_results.ipynb` does not import `xai/`; `export_accuracy.py` is a standalone `main()` script imported by nothing):

| Module | Role (from its content) |
|---|---|
| `experiments.py` | declares the experiments (`Axis`, `Method`, `Experiment`) by axis: confounders, feature blocks, temporality, categories |
| `category_permutation.py` | marginal distributions per category and resampling of categories (permutation-style importance) |
| `order_perturbation.py` | perturbs the sentence order (`add_order`, `ORDER_COL`) to test order sensitivity |
| `lofo_resample.py` | noise transforms for leave-one-feature-out conditions (`make_noise_transform`, `noise_conditions`) |
| `direction.py` | helpers `subjectivity_pools`, `convert_toward` (see the warning below) |
| `metrics.py` | AUROC, accuracy and thresholds at fixed FPR (1 % and 0.1 %) |
| `results_store.py` | persistent store of experiment results |
| `export_common.py`, `export_accuracy.py`, `export_latex.py`, `export_figures.py`, `sources.py` | export of tables (LaTeX), figures and bibliography references for the article |
| `reporting.py`, `notebook_display.py` | text tables and `category_report`; display of LaTeX and figures inside a notebook |

Warnings:

- `xai/direction.py` coexists with a folder `xai/direction/`. Python resolves `...xai.direction` to the module `direction.py`; the folder has no `__init__.py`, uses relative imports and several imports that do not resolve (e.g. `xai.text_normalization`, `xai.lofo_conditions`), and nothing reaches it. Do not rename or "fix" it without an explicit decision.
- `xai/results_store.py` and `xai/direction/result_store.py` are near-duplicates.
- Seven of the fourteen experiments declared in `experiments.py` are never fed by `results_xai` (anomaly A13).

## 6. Running it

```bash
# train (the preset puts the model on cuda:0; the segmentation uses cuda:1; local models required)
python -m src.detector.subjectivity_aware_generation_estimator.training.gru_train experiment/train=train_sage_gru_v3
# evaluate on Sem-Detect with that model
python -m src.evaluation.run_eval experiment/eval=sage_gru_3_semdetect
```

The eval preset `sage_gru_3_semdetect` has `train_preset: "train_sage_gru_v3"` (a model trained on APRDB) and sets `hardware.device: "cuda:1"`. To evaluate the model trained on Sem-Detect (`train_sage_gru_v3_semdetect`) you need an eval preset whose `train_preset` points to it.

Neither command was executed while writing this documentation; their composition was checked with `--cfg job` by `scripts/check_structure.py`. See `doc/reproducing-results.md` for what each published result needs.

## 7. Known defects (static findings)

- `training/gru_features.py` is broken as written: `block_to_sequence` reads `spec.include_valid`, an attribute `FeatureSpec` does not have, and `_as_spec` calls `get_spec("v1")` / `get_spec("v2")`, names that are absent from `SPECS` (`v1_base`, `v2_assertion`, ...). This affects the legacy `SageGRU` path (`build_features_mono`), `transformer_train.py` (`build_features`) and `xai/category_permutation.build_eval_tensors`. This was established by reading the code, not by running it (anomaly A35).
- `transformer_train.py`: validation leakage through best-epoch selection on the validation AUC (anomaly A36).
- Other SAGE anomalies: A11 (artifact overwritten by identical-hash retraining), A13 (unfed xai experiments), A16 (hard-coded models and GPU), A21 (no deterministic GPU flags), A23 (`segmentation_df_v2`).

Not covered by this page: `model/SimpleGRUWithGlobal.py` and `sage.old.py` (no importer found in the audit), `model/artifact_io.py` (described in `doc/architecture.md`, section 7).
