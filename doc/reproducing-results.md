# Reproducing results

What can be recomputed from this repository, what needs the original machine, and which notebook produces which result of the article.
Sources: the reproducibility audit and its review. Nothing in this page was executed while writing it; the smoke run is a proposal to be run by you.

## 1. What a run needs

| Need | Detail |
|---|---|
| Python | 3.10, a virtual environment at the repository root (`.venv`); run `python -m <module>` from the root |
| Dependencies | `requirements_local.txt` is **incomplete and partly out of date** (audit A8, A20): torch, numpy, pandas, scipy, matplotlib, tqdm, pyarrow, lightgbm, sentence-transformers, shap, pysbd, PyMuPDF, PyYAML, requests, boto3 and google-genai are missing; `transformers` 5.8.0 and `huggingface-hub` 1.14.0 were installed while the file pins 4.57.5 and 0.36.0. Pin the real environment before any attempt to rebuild |
| `.env` | defines `DATA_DIR`, `CACHE_DIR`, `HF_HOME` (template: `.env.template`, whose `CACHE_DIR` value does not match the real data location, anomaly A17). Optional: `HF_TOKEN`, `OPENREVIEW_LOGIN`, `OPENREVIEW_MDP`, `OPEN_AI_API_KEY`, `MISTRAL_API_KEY`, `SEMDETECT_CONFIG` |
| Hugging Face caches | APRDB, Sem-Detect and the models below; `HF_HUB_OFFLINE=1` works only if the needed files are cached |
| Models | `tiiuae/falcon-7b` and `-instruct` (Binoculars, Ada-DetectGPT), `gpt2-xl` (Fast-DetectGPT, DetectGPT) and a T5 mask filler (DetectGPT), `all-MiniLM-L6-v2` (Cristal Anchor), Qwen3 models through vLLM (Cristal Anchor, Sem-Detect adaptation), `Mistral-7B-Instruct-v0.3` and `Qwen3-Embedding-0.6B` (Sem-Detect) |
| SAGE segmentation models | `roberta_subjectivity_regressor` (subjectivity score, regression) and `roberta_speech_act_classifier` (formerly `final_model_definitivo` and `final_model_definitivov2`). Install with `python -m scripts.fetch_sage_models` (Hugging Face Hub, checked against `scripts/sage_models.sha256`, written to `models/`); `linguistic_segmentation.py` reads `SAGE_MODELS_DIR`, else `<repo>/models/`. No training code or documented origin (anomaly A16, model cards in `doc/model_cards/`) |
| GPUs | SAGE segmentation uses `SAGE_SEGMENTATION_DEVICE` (default `cuda:1` with two GPUs, `cuda:0` with one, `cpu` otherwise; anomaly A16), SAGE training puts the model on `hardware.device` (`cuda:0` in the presets; override with `hardware.device=...`). A vLLM server (separate environment `VLLM_ENV_DIR`, port 8000, GPU 1 by default) is needed for Cristal Anchor and the LLM parts of Sem-Detect and shares GPU 1 with SAGE segmentation by default: stop vLLM before evaluating SAGE or move one of them |

## 2. Smoke run (no GPU, no network, no model)

The cheapest path through `run_eval` to a metric is the `dummy` detector on a cached dataset:

```bash
CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 .venv/bin/python -m src.evaluation.run_eval \
  hardware=cpu detector=dummy dataset.subset_size=10000 data_loader.batch_size=500
```

Notes (from the audit, checked by reading the code):

- A valid `.env` (the four variables of section 1) is required even for this run, and the cached dataset below exists only on the original machine.

- `hardware=cpu` is required, otherwise the evaluator moves labels to `cuda:0`; `CUDA_VISIBLE_DEVICES=""` is needed because the dummy's `randn` goes to CUDA whenever a GPU is visible.
- `dataset.subset_size=10000` with the current dataset seed 0 hits the cached dataset `dataset_cache/dataset_13af569f….pt` on the original machine, so no network is needed there. The default `subset_size: 100` has no cache and triggers a network listing: do not use it offline. On another machine the cache does not exist and APRDB is downloaded.
- It appends one line to `RESULT_DIR/results.jsonl` (no locking; line 42 of the original file is already corrupted by concurrent writes). Copy the file before running and check afterwards that exactly one line was added.
- Importing the factory creates a vLLM client object (no request) and may create `example.log` in the current directory (anomaly A10).
- Expected AUROC is close to 0.5 (earlier dummy runs: 0.51 to 0.64), because scores are random and unseeded.

## 3. Training and evaluating SAGE

**Validated on 2026-10-20** (after renaming the segmentation model directories), on the original machine: the `dummy` smoke run (AUROC 0.5022, one line appended), `gru_train experiment/train=train_sage_gru_v3` (about 2 minutes; artifact hash `40267e07…`) and `run_eval experiment/eval=sage_gru_3_aprdb` (10 000 reviews, AUROC 0.9548, accuracy 0.8817). The retraining reproduced the previous artifact exactly: identical weights, validation probabilities and AUC (0.9549), and the new evaluation row has the same hash (`a41f67e8…`) and AUROC as the earlier row 37 of `results.jsonl`. The retraining overwrote `artifact.joblib` in place (the new one has two extra keys, `n_outputs` and `metrics`); a copy of the previous one was kept outside the repository.

```bash
python -m src.detector.subjectivity_aware_generation_estimator.training.gru_train experiment/train=train_sage_gru_v3
python -m src.evaluation.run_eval experiment/eval=sage_gru_3_aprdb
```

These need the SAGE segmentation models (section 1) and two GPUs; their composition was checked with `--cfg job` (`scripts/check_structure.py`), they were not run. Known limits:

- The training hash of the article's model `b5ad7993…` (preset `train_sage_gru_2_cat`) cannot be recomposed from the current presets: four keys changed since (`dataset_train.seed` and `dataset_val.seed` 42 → 0, `training.model.feature_spec` and `training.model.seed` added), and the artifact on disk was overwritten by the third identical training on 2026-09-16. Retraining to the identical artifact is lost (anomaly A11).
- Old evaluations used `dataset.seed=42`; the dataset seed is now 0, so recomposing old presets with `subset_size: 100` gives another sample and another hash. Try `dataset.seed=42` to approach the old ones (not verified; anomalies A14, A11).
- GPU training is not deterministic bit for bit (no `cudnn.deterministic`, anomaly A21).

## 4. Which notebook produces which result

No script produces a table or figure of the article: the scripts are `gru_train`, `mlp_train`, `transformer_train` and `run_eval` (rows in `results.jsonl`). Result notebooks read those rows, caches and artifacts. Do not execute or clear them: their saved outputs may be the only trace of published numbers.

| Result | Notebook | Inputs | Recomputable by script? |
|---|---|---|---|
| SAGE tables (AUROC, accuracy, ROC per hash) | `src/detector/subjectivity_aware_generation_estimator/sage_results.ipynb` | `results.jsonl` (row with hash `580ae5ec…`), APRDB | read only; regenerating the row is not possible exactly: the training config was recovered but the original artifact was overwritten (A11) |
| XAI: feature-block importance, speech-act noise, sentence order (CSV, LaTeX, figures) | `…/tests/results_xai.ipynb` | train preset `train_sage_gru_v3`, APRDB calibration/test, cached segmentation, `xai/*`, five seeds, cached CSV under `RESULT_DIR/xai/v3` | no (many retrainings; GPU). With the cached CSV it can be re-read on CPU |
| SAGE vs Sem-Detect, tables A and B, ROC, AUC per generator (E0 to E3) | `notebook/results/semdetect_sage_recap.ipynb` | artifacts `40267e07…` (APRDB) and `69ec0e68…` (Sem-Detect), Sem-Detect caches, `lightgbm_best_model.pkl` | no. Its last saved run is reported (project notes) to contain an error output, and several exports are missing on disk (A18) |
| SVM fusion of SAGE and Sem-Detect | `notebook/results/hybrid_svm.ipynb` | same inputs plus `recap/sage_scores_*` | no |
| `tab:detector_comparison` | `src/results_exploration/detector_comparison.ipynb` | rows 4, 5, 7, 13, 14, 26 of `results.jsonl`; `recap/scores_partie_B.parquet` (missing); hard-coded `tab:cross_dataset` values | read only; rows 4, 5 and 7 are recomposed by `dataset.seed=42` (A14), rows 13 and 14 are not recomposable because their presets changed since |
| E0 Sem-Detect reproduction | `src/detector/sem_detect/notebook/semdetect_repro.ipynb` | dataset revision `18b74b55…`, Mistral-7B and Qwen3-Embedding caches, `lightgbm_best_model.pkl` | no |
| E1 Qwen3-32B claims, dataset references | `…/semdetect_e1.ipynb` | plus `claims_llm` cache | no (vLLM) |
| E2 Qwen3-32B claims and local references | `…/semdetect_e2.ipynb` | `sem_detect_e2/local_refs.parquet` | no (vLLM); a cleaning truncation was never fixed, so its degradation is an upper bound (A9) |
| E3 Sem-Detect on APRDB (variant a used in the article) | `…/semdetect_e3.ipynb` | `sem_detect_e3/pilot_universe.parquet` | no (vLLM) |

The identification of notebooks with E0 to E3 comes from project notes and is not re-verified.

## 5. Artifacts outside the repository

Everything below lives under `CACHE_DIR` (a directory of your choice) and is not versioned:

- `results/<hash>/artifact.joblib` (+ `gru_result.npz`): trained models; `results/results.jsonl`: evaluation rows, one JSON object per line (the file was 118 MB with 42 lines; line 42 holds two records merged by a concurrent write, anomaly A15);
- `results/implicite_segmentation/<dataset_hash>_<spec_hash>/sentences_degrees_final.csv`: cached SAGE segmentation;
- `results/evaluation_batch/evaluation_segmented_txt.jsonl`: segmented texts appended at every SAGE evaluation;
- `results/xai/`: notebook outputs; `recap/` and `hybrid_svm/` (directly under `CACHE_DIR`, not under `results/`): notebook outputs;
- `dataset_cache/`: built datasets (the cache of `subset_size=100, seed=42` sits at an old location and is not found by the current loader, anomaly A22);
- `sem_detect_store/`, `sem_detect_e2/`, `sem_detect_e3/`, `claims_llm/`: Sem-Detect experiment caches.

Top-level entries of the `CACHE_DIR` of the original machine (listing only; sizes: `results/` about 23 GB, `hf/` about 171 GB):

| Entry | Content |
|---|---|
| `results/` | artifacts, `results.jsonl`, segmentation cache, XAI store (above) |
| `dataset_cache/` and loose `dataset_<hash>.pt` | built dataset caches (the loose ones are an older location) |
| `claims_llm/`, `llm_jobs/`, `sem_detect_store/`, `sem_detect_e2/`, `sem_detect_e3/`, `contamination/` | Sem-Detect experiment caches (see `doc/vllm.md`, section 5) |
| `recap/`, `hybrid_svm/` | notebook outputs |
| `hf/` and `models--*` folders | Hugging Face caches (a second one existed next to it) |
| `papers_pdf/`, `review_paper_mapping.json`, `paper_sources.parquet` | paper PDFs and mappings for Cristal Anchor |
| `ada_detect_gpt_wfunc_weight_<hash>.pth` | cached Ada-DetectGPT witness function (so a B-spline fit was run at least once) |
| `vllm/`, `triton_vllm`, `torch/`, `triton/`, `pip/`, `xdg/` | library and compiler caches (vLLM, Triton, PyTorch) |
| `outputs/{eval,train}` | Hydra run directories |

A fresh machine therefore needs: the environment (section 1), the datasets from Hugging Face, the models, and the two local segmentation models; everything else is regenerated or lost, as detailed in `doc/known-issues.md`.
