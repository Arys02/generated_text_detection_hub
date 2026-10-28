# Binoculars

Zero-shot detector based on the ratio between the perplexity of a "performer" model and the cross-perplexity between an "observer" and the performer.

- Code: `src/detector/binocular.py` (classes `Binoculars` and `BinocularDetector`); config `conf/detector/binocular.yaml`; detector name `binocular`, type `statistical`.
- Provenance: according to the project author, ported from the Ada-DetectGPT repository (https://github.com/Mamba413/AdaDetectGPT; Zhou et al., NeurIPS 2025, arXiv 2510.01268). That repository states that its baseline detectors are modified from their official versions or from the Fast-DetectGPT repository, and showed no licence file when consulted on 2026-10-20 (to be re-checked; audit anomaly A28).

## How it works

1. `Binoculars.__init__` loads two causal LMs with the same tokenizer vocabulary (`assert_tokenizer_consistency`): the observer `tiiuae/falcon-7b` on `device1` and the performer `tiiuae/falcon-7b-instruct` on `device2`, in bfloat16 by default, using the Hugging Face token from `huggingface_config`.
2. `compute_score(texts)` tokenizes (padding to the longest text, truncation to `max_token_observed` tokens), runs both models, computes the performer perplexity `ppl` and the cross-entropy `x_ppl` between observer and performer distributions, and returns `ppl / x_ppl` per text.
3. **Direction:** a low score means machine-generated. `Binoculars.predict` (not used by the evaluator) thresholds with `score < threshold` → "Most likely AI-generated". The evaluator uses `BinocularDetector.predict`, which returns the raw scores; the raw AUROC is then below 0.5 and `Evaluator`/`compute_metrics` automatically evaluates `-score` and flips the threshold comparison (see `doc/architecture.md`, section 5).
4. Thresholds from the original work, for Falcon-7B/Falcon-7B-Instruct in bfloat16: `BINOCULARS_ACCURACY_THRESHOLD = 0.9015…` (F1-optimised) and `BINOCULARS_FPR_THRESHOLD = 0.8536…` (low-FPR). `mode: "low-fpr"` or `"accuracy"` selects one in `change_mode`.

`BinocularDetector` (the class registered in the factory) wraps `Binoculars`:

- it keeps `self.config = config['params']`, which is why `params.threshold` is visible to the evaluator (`Evaluator` reads `model.config.threshold`);
- `predict(texts)` drops empty or non-string texts **before** scoring and returns `None` if nothing remains. Dropping texts shortens the output and misaligns scores and labels; the evaluator expects one score per text.

## Parameters (`conf/detector/binocular.yaml`)

| Key | Default in YAML | Meaning |
|---|---|---|
| `observer_name`, `performer_name` | `tiiuae/falcon-7b`, `tiiuae/falcon-7b-instruct` | the two models |
| `max_token_observed` | 512 (the class default is 1024) | truncation length |
| `mode` | `low-fpr` | picks the threshold constant; valid values are `low-fpr` and `accuracy` (the YAML comment mentioning `high-fpr` would raise `ValueError`) |
| `device1`, `device2` | `${hardware.device1}`, `${hardware.device2}` | devices of the two models (`conf/hardware/dual_gpu.yaml`) |
| `threshold` | not set in the YAML; set in the presets | `Binoculars.__init__` receives it but `change_mode(mode)` immediately overwrites it with the constant of `mode`; the value that matters is the one the evaluator reads from `model.config.threshold` (precision/recall/F1/accuracy), falling back to the low-FPR constant |
| `seed` | `${seed}` | not used by the scoring code |

If `device1`/`device2` are absent the module defaults are `cuda:0` and, if two GPUs exist, `cuda:1`; they are computed at import time from `torch.cuda`.

## Presets and running

Presets in `conf/experiment/eval/`: `binocular_simple.yaml` (APRDB, `hardware: dual_gpu`, both models on `cuda:1`, `max_token_observed: 512`, threshold 0.8536…), `binocular_simple_small.yaml` (smaller Falcon3-1B models according to the audit notes, while keeping the Falcon-7B threshold 0.8536…, which was calibrated for Falcon-7B and does not apply to other models; not re-verified here) and `binocular_simple_full.yaml` (`split: test`, `subset_size: 10000`, `keep_meta: true`).

```bash
python -m src.evaluation.run_eval experiment/eval=binocular_simple
```

Requires the two Falcon models (cached by Hugging Face or downloaded) and a GPU with enough memory for two 7B models in bfloat16.

## Caveats

- `torch.set_grad_enabled(False)` runs at import of this module, and `detector_factory` imports it: autograd is disabled for every run that imports the factory (audit anomaly A32).
- `compute_score` itself does not guard against empty texts (the filter is in `BinocularDetector.predict`); without CUDA `DEVICE_1` falls back to `cpu`.
- `trust_remote_code=False`; the token comes from the `HF_TOKEN` environment variable.
- Several `print` calls in `BinocularDetector.__init__` (tokenizer and model vocabulary sizes) are debugging output.
