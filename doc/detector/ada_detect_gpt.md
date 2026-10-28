# Ada-DetectGPT

Fast-DetectGPT-style statistic with an adaptive "witness" function `w` applied to the log-probabilities, learned from data with B-splines.

- Code: `src/detector/ada_detect_gpt/` (`ada_detect_gpt.py`, `BSpline.py`, `nuisance_func.py`, `nuisance_func_human.py`, `helper.py`); config `conf/detector/ada_detect_gpt.yaml`; detector name `ada_detect_gpt`, type `statistical`.
- Provenance: this package comes from the Ada-DetectGPT repository (https://github.com/Mamba413/AdaDetectGPT), whose other detectors the project also ported (Binoculars, DetectGPT, Fast-DetectGPT). Paper: Zhou, Zhu, Su, Ye, Yang, Gavioli-Akilagun and Shi, *AdaDetectGPT: Adaptive Detection of LLM-Generated Text with Statistical Guarantees*, NeurIPS 2025, arXiv 2510.01268. The repository showed no licence file when consulted on 2026-10-20 (to be re-checked; audit anomaly A28). Comments in this package are in English; `BSpline.py` contains a `# in R` comment, a hint that part of the code was ported from R.

## How it works

`get_classification_stat(logits_ref, logits_score, labels, w_func, shift_value)` computes, in float32, the Fast-DetectGPT statistic with `w_func` applied to the scoring log-probabilities, a numerically stable variance (`E[(X - E[X])^2]`) and an optional shift. `AdaDetectGPT.predict` returns, for each text, `1 - norm.cdf(stat)`.

Note the direction: the score is a one-sided p-value, `1 - Φ(stat)`, so **a large statistic gives a small score**; the evaluator's automatic inversion handles it (raw AUROC below 0.5, see `doc/architecture.md`, section 5). The same applies to rows of `results.jsonl`: the stored `y_scores` have the raw direction.

Scoring and sampling models may differ (`sampling_model_name`); if they are the same model it is loaded once. When they differ, the tokenizations must produce identical token ids (assertion "Tokenizer is mismatch").

## Witness function (`w_func`)

| Value | Behaviour |
|---|---|
| `identity` (default) | `nn.Identity()`: no learning, no cache. This is the Fast-DetectGPT statistic with a shift of 0 |
| `pretrained` | `BSplineTwoSample` with hard-coded coefficients (`beta_hat`) and B-spline config `{start: -32, end: 0, n_bases: 7, spline_order: 2, intercept: 1}` |
| `bspline` | fits `BSplineTwoSample` on human texts (`original`) and machine texts (`sampled`) |
| `bspline_theory` | fits `BSplineTheory` on human texts only |

For `bspline` and `bspline_theory`, `params.train_dataset` is required (not present in the default YAML): a name or list of names; each is read as `DATA_DIR/<name>.json` containing the keys `original` and `sampled` (`load_data`, `load_training_data`). The fitted function is cached as `CACHE_DIR/ada_detect_gpt_wfunc_weight_<hash>.pth`, where `<hash>` combines `train_dataset`, `seed` and `w_func`; a cached file is loaded instead of refitting (`WFUNC_REGISTRY` maps the class names).

## Parameters (`conf/detector/ada_detect_gpt.yaml`)

| Key | Value | Notes |
|---|---|---|
| `scoring_model_name`, `sampling_model_name` | `tiiuae/falcon-7b` | the constructor defaults (`tiuuae/...`, a typo) are overridden by the config |
| `w_func` | `identity` | see above |
| `device` | `${hardware.device}` | |
| `seed` | `${seed}` | seeds Python, NumPy and torch at the end of `__init__` |
| `burn_in`, `checkpoint_path` | `0.0`, `null` | not read: `grep -rn "burn_in\|checkpoint_path" src` finds no use in the Python code; they appear only in `conf/` (this YAML and four `conf/experiment/experiment_old/` presets); notebooks were not searched |
| `discrepancy_analytic` | absent | read by `predict` but not used (always the same criterion) |

Models are loaded from `CACHE_DIR/local.<model>` if present, otherwise from the hub with the `HF_TOKEN` environment variable.

## Presets and running

`conf/experiment/eval/ada_detect_gpt_simple.yaml` (`name: "ada_gpt_basic"`) and `ada_detect_gpt_simple_gemma_identity.yaml`, which despite its name has the same settings as the former except for an empty `detector: params:` block and `name: "ada_gpt_gpu_basic"` (audit anomaly A3). Whether the empty `params:` block overrides the detector `params` after Hydra composition was not verified (it would need running the composition).

```bash
python -m src.evaluation.run_eval experiment/eval=ada_detect_gpt_simple
```

Requires `tiiuae/falcon-7b` and a GPU. `trainable: true` in the YAML refers to the optional witness-function fitting performed inside the detector's constructor; there is no separate training entry point.
