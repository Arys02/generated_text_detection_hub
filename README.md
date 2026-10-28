# Generated Text Detection Hub

A research code base for detecting machine-generated text, mainly **peer reviews** (human-written vs. machine-generated, "MGT"). It contains:

- **SAGE** (Subjectivity-Aware Generation Estimator), the project's main detector: a small sequence model (GRU by default) over per-sentence linguistic features (speech-act category and subjectivity score).
- **Cristal Anchor**, a second homemade detector that compares a review with an LLM-regenerated version of it.
- Re-implementations of zero-shot baselines — Binoculars, DetectGPT, Fast-DetectGPT, Ada-DetectGPT — ported from the [Ada-DetectGPT repository](https://github.com/Mamba413/AdaDetectGPT) (Zhou et al., NeurIPS 2025, [arXiv 2510.01268](https://arxiv.org/abs/2510.01268)).
- **Sem-Detect**, a third-party detector, embedded together with an adaptation layer (used from notebooks).
- A Hydra-driven evaluation framework with two datasets: the AI Peer Review Detection Benchmark (APRDB) and the Sem-Detect ML-conference reviews.

> **Status: frozen.** Development has stopped and the repository was audited, cleaned and documented. There are no automated tests. Read [Known issues](doc/known-issues.md) before relying on any result, and [Reproducing results](doc/reproducing-results.md) to see what can be recomputed. The article this code supports has no public reference yet.

## Documentation

| Page | Content |
|---|---|
| [doc/architecture.md](doc/architecture.md) | How a run flows from Hydra config to `results.jsonl`; hashes, artifacts, import side effects, directory map |
| [doc/extending.md](doc/extending.md) | Adding a detector, a dataset, a preset, a SAGE feature or model |
| [doc/detector/](doc/detector/README.md) | One page per detector (SAGE, Cristal Anchor, Binoculars, DetectGPT, Fast-DetectGPT, Ada-DetectGPT, Sem-Detect) |
| [doc/datasets.md](doc/datasets.md) | APRDB and Sem-Detect loaders and configuration keys |
| [doc/vllm.md](doc/vllm.md) | Serving local LLMs with vLLM (profiles, manager, watchdog) for Cristal Anchor and Sem-Detect |
| [doc/reproducing-results.md](doc/reproducing-results.md) | Requirements, smoke run, notebook ↔ article result map, artifacts outside the repository |
| [doc/known-issues.md](doc/known-issues.md) | Limits and defects found by the audit |

## Repository layout

```
conf/        Hydra configs: eval.yaml, train.yaml, config groups, presets (experiment/{eval,train})
src/         all Python code, imported as src.* (config, data, detector, evaluation, utils, ...)
notebook/    exploratory and result notebooks (notebook/results/ holds the article's result notebooks)
data/        dataset-preparation notebooks and a few data files
scripts/     check_structure.py (structural check) and a commented-out run script
doc/         documentation (this page's table above)
```

## Installation

- Python 3.10 in a virtual environment at the repository root (`.venv` was used). Dependencies are listed in `requirements_local.txt`, which is **incomplete**: torch and several libraries (numpy, pandas, scipy, matplotlib, lightgbm, sentence-transformers, shap, pysbd, PyMuPDF, ...) must be installed separately, and some pinned versions differ from the ones actually used. See [doc/reproducing-results.md](doc/reproducing-results.md).
- Copy `.env.template` to `.env` and set `HF_HOME`, `DATA_DIR` and `CACHE_DIR`. `src/config.py` reads them at import time and fails if one is missing, so every `src.*` import needs a valid `.env`. Trained models, caches and `results.jsonl` are written under `CACHE_DIR`, outside the repository.
- Sem-Detect's upstream code is not redistributed (the upstream repository has no licence). Fetch it once with `python -m scripts.fetch_semdetect` (clones the pinned upstream commit, installs `src/detector/sem_detect/semdetect/` and `models/lightgbm_best_model.pkl`, checks them against `semdetect.sha256`); `--check` verifies an existing installation offline. See `src/detector/sem_detect/VENDORED.md`.
- Optional variables used by some detectors: `HF_TOKEN`, `OPENREVIEW_LOGIN`, `OPENREVIEW_MDP`, `OPEN_AI_API_KEY`, `MISTRAL_API_KEY`.
- Some components need a local vLLM server (OpenAI-compatible API on `127.0.0.1:8000`) running in a separate environment, and SAGE needs two RoBERTa segmentation models (install them with `python -m scripts.fetch_sage_models`, which downloads them from the Hugging Face Hub into `models/` and checks their SHA-256) and two GPUs (see [SAGE](doc/detector/sage.md)). These resources are specific to the original machine.

## Quick start

Run everything from the repository root with `python -m <module>`.

Smoke run of the evaluation chain (random detector, CPU only; expected AUROC around 0.5; it appends one line to `results.jsonl`, see the notes in [doc/reproducing-results.md](doc/reproducing-results.md)):

```bash
CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 python -m src.evaluation.run_eval \
  hardware=cpu detector=dummy dataset.subset_size=10000 data_loader.batch_size=500
```

Evaluate a baseline or a trained detector with a preset:

```bash
python -m src.evaluation.run_eval experiment/eval=binocular_simple
python -m src.evaluation.run_eval experiment/eval=sage_gru_3_aprdb
```

Train SAGE (GRU), then evaluate it (the eval preset names the train preset it needs through `train_preset`):

```bash
python -m src.detector.subjectivity_aware_generation_estimator.training.gru_train experiment/train=train_sage_gru_v3
```

Print a composed configuration without running anything: add `--cfg job` to any of these commands. Check the project structure without GPU or training: `python -m scripts.check_structure --compare scripts/check_structure_baseline.json`.

These training and evaluation commands were checked for configuration composition only; they were not executed during the audit.

## Notebooks

Result notebooks (`notebook/results/`, `src/detector/sem_detect/notebook/`, `src/detector/subjectivity_aware_generation_estimator/sage_results.ipynb` and `tests/results_xai.ipynb`) hold published results in their saved outputs. Do not execute them or clear their outputs. Which notebook produces which result is listed in [doc/reproducing-results.md](doc/reproducing-results.md).

## Licence

The code of this repository is released under the [MIT licence](LICENCE). It does not cover third-party material:

- **Sem-Detect upstream code** (`src/detector/sem_detect/semdetect/`, `models/lightgbm_best_model.pkl`, from `github.com/avduarte333/Sem-Detect` at commit `d97175a5…`): the upstream repository has no licence file, so it is not distributed here; fetch it with `python -m scripts.fetch_semdetect`. No right over it is granted by this repository.
- **Ported baselines** (Binoculars, DetectGPT, Fast-DetectGPT, Ada-DetectGPT): ported from the Ada-DetectGPT repository (https://github.com/Mamba413/AdaDetectGPT), which has no licence file (confirmed by the project author, 2026-10-21). The MIT licence applies to this repository's own code only; the upstream authors' rights remain theirs.
- **APRDB** is distributed under an Intel research-use licence (see [doc/datasets.md](doc/datasets.md)); datasets are not part of this repository.

## Citation

The project article has no public reference yet; a citation entry will be added when it is available.

### Related work cited by this repository

```bibtex
@inproceedings{la-cava-tagarelli-2025-openturingbench,
    title = "{O}pen{T}uring{B}ench: An Open-Model-based Benchmark and Framework for Machine-Generated Text Detection and Attribution",
    author = "La Cava, Lucio  and
      Tagarelli, Andrea",
    editor = "Christodoulopoulos, Christos  and
      Chakraborty, Tanmoy  and
      Rose, Carolyn  and
      Peng, Violet",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.emnlp-main.1354/",
    doi = "10.18653/v1/2025.emnlp-main.1354",
    pages = "26666--26682",
    ISBN = "979-8-89176-332-6"
}

@misc{OTB,
      title={OpenTuringBench: An Open-Model-based Benchmark and Framework for Machine-Generated Text Detection and Attribution},
      author={Lucio La Cava and Andrea Tagarelli},
      year={2025},
      eprint={2504.11369},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2504.11369},
}

@inproceedings{LaCava2024ECAI,
  author       = {Lucio La Cava and
                  Davide Costa and
                  Andrea Tagarelli},
  editor       = {Ulle Endriss and
                  Francisco S. Melo and
                  Kerstin Bach and
                  Alberto Jos{\'{e}} Bugar{\'{\i}}n Diz and
                  Jose Maria Alonso{-}Moral and
                  Sen{\'{e}}n Barro and
                  Fredrik Heintz},
  title        = {Is Contrasting All You Need? Contrastive Learning for the Detection
                  and Attribution of AI-generated Text},
  booktitle    = {{ECAI} 2024 - 27th European Conference on Artificial Intelligence,
                  19-24 October 2024, Santiago de Compostela, Spain - Including 13th
                  Conference on Prestigious Applications of Intelligent Systems {(PAIS}
                  2024)},
  series       = {Frontiers in Artificial Intelligence and Applications},
  volume       = {392},
  pages        = {3179--3186},
  publisher    = {{IOS} Press},
  year         = {2024},
  url          = {https://doi.org/10.3233/FAIA240862},
  doi          = {10.3233/FAIA240862},
}
```

Ada-DetectGPT, the repository the baselines are ported from:

```bibtex
@inproceedings{zhou2025adadetect,
  title={AdaDetectGPT: Adaptive Detection of LLM-Generated Text with Statistical Guarantees},
  author={Hongyi Zhou and Jin Zhu and Pingfan Su and Kai Ye and Ying Yang and Shakeel A O B Gavioli-Akilagun and Chengchun Shi},
  booktitle={The Thirty-Ninth Annual Conference on Neural Information Processing Systems},
  year={2025}
}
```

Other works whose methods are implemented or embedded: Binoculars, DetectGPT, Fast-DetectGPT, Sem-Detect (Duarte et al., arXiv 2605.21713) and the AI Peer Review Detection Benchmark (arXiv 2502.19614). Their BibTeX entries are in [doc/references.bib](doc/references.bib) (`hans2024spotting`, `mitchell2023detectgpt`, `bao2024fast`, `duarte2026semdetect`, `yu2026your`).
