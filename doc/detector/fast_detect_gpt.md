# Fast-DetectGPT

Zero-shot detector that replaces DetectGPT's perturbations by a closed-form "conditional probability curvature": the log-likelihood of the text under the scoring model is compared to its expectation and variance when tokens are drawn from a reference model.

- Code: `src/detector/fast_detect_gpt.py` (class `FastDetectGPT`); config `conf/detector/fast_detect_gpt.yaml`; detector name `fast_detect_gpt`, type `statistical`.
- Provenance: according to the project author, ported from the Ada-DetectGPT repository (https://github.com/Mamba413/AdaDetectGPT; Zhou et al., NeurIPS 2025, arXiv 2510.01268). That repository states that its baseline detectors are modified from their official versions or from the Fast-DetectGPT repository, and showed no licence file when consulted on 2026-10-20 (to be re-checked; audit anomaly A28).

## How it works (`predict`)

For each text: tokenize (truncation on), run the scoring model, and compute a discrepancy with `logits_ref = logits_score` (self-detection):

- analytic version (`get_sampling_discrepancy_analytic`): `(sum log p(x) - sum mean_ref) / sqrt(sum var_ref)`, averaged;
- sampling version (`get_sampling_discrepancy`): 10 000 samples from the reference distribution.

The returned score is `sigmoid(discrepancy)`; higher means more likely machine-generated. `NaN` discrepancies become `-1e4`.

## Parameters (`conf/detector/fast_detect_gpt.yaml`)

| Key | Value | Notes |
|---|---|---|
| `scoring_model_name` | `gpt2-xl` | scoring LM |
| `sampling_model_name` | `gpt2-xl` | **loaded but not used for scoring**: `logits_ref = logits_score` (a comment in the code says to substitute a separate sampling model); loading it still costs memory |
| `discrepancy_analytic` | `"True"` (a string) | `predict` does `self.config.get('discrepancy_analytic', True)` and tests its truthiness: the string `"False"` is also truthy, so the analytic branch is always taken |
| `device` | `${hardware.device}` | |

The constructor reads `config['discrepancy_analytic']` without a default (removing the key from the YAML would raise a `KeyError` at load time), while `predict` uses `.get(..., True)`. Loading: local directory `CACHE_DIR/local.<model>` if present, otherwise the hub with `cache_dir=CACHE_DIR`. `predict` prints the `use_analytic` value on every call.

## Preset and running

`conf/experiment/eval/fast_detect_gpt_simple.yaml` (APRDB, `cuda:0`) and `fast_detect_gpt_full.yaml` (`cuda:1`, `name: "fast_detect_gpt"`). The `name` of `fast_detect_gpt_simple` is `detect_gpt_gpu1` (copy-paste, audit anomaly A2): rows in `results.jsonl` produced by this preset carry that name.

```bash
python -m src.evaluation.run_eval experiment/eval=fast_detect_gpt_simple
```
