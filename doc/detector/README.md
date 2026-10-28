# Detectors

One page per detector. The common contract (`BaseDetector`, YAML, factory, evaluator) is in `doc/architecture.md`, and how to add a detector is in `doc/extending.md`.

| Name in the factory | Type | Page | Provenance |
|---|---|---|---|
| `sage` | linguistic | [sage.md](sage.md) | this project |
| `cristal_anchor` | contextual | [cristal_anchor.md](cristal_anchor.md) | this project |
| `binocular` | statistical | [binocular.md](binocular.md) | ported from the Ada-DetectGPT repository |
| `detect_gpt` | statistical | [detect_gpt.md](detect_gpt.md) | ported from the Ada-DetectGPT repository |
| `fast_detect_gpt` | statistical | [fast_detect_gpt.md](fast_detect_gpt.md) | ported from the Ada-DetectGPT repository |
| `ada_detect_gpt` | statistical | [ada_detect_gpt.md](ada_detect_gpt.md) | Ada-DetectGPT repository |
| (not in the factory) | n/a | [sem_detect.md](sem_detect.md) | third-party code plus an adaptation layer |
| `dummy` | statistical | this page | this project |

`dummy` (`src/detector/random_base_detector.py`, `conf/detector/dummy.yaml`) returns `sigmoid(randn)` scores. It needs no model and is used for smoke runs of the evaluation chain (for example with `hardware=cpu`). Its `randn` call moves to CUDA whenever a GPU is visible, regardless of `hardware`; hide the GPUs (`CUDA_VISIBLE_DEVICES=""`) for a pure CPU run. The `seed` of its config is not used, so two runs with the same config have the same hash and different results (audit anomaly A19).

The ported detectors come from the Ada-DetectGPT repository (https://github.com/Mamba413/AdaDetectGPT, paper arXiv 2510.01268), as stated by the project author. That repository lists many other detectors (BiScope, DetectLLM, DNA-GPT, GLTR, ImBD, RADAR, ...) that were not ported here. No licence file was visible there when consulted on 2026-10-20 (to be re-checked; audit anomaly A28).
