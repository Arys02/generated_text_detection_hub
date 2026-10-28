---
library_name: transformers
pipeline_tag: text-classification
tags: [roberta, speech-acts, peer-review, sage]
language: en
---

# roberta-speech-act-classifier

Speech-act classifier used by SAGE (Subjectivity-Aware Generation Estimator) of
[generated_text_detection_hub](https://github.com/Arys02/generated_text_detection_hub) to label each sentence of a review.

- Architecture: `RobertaForSequenceClassification` (RoBERTa-base size: 12 layers, hidden size 768, 514 positions), 5 classes.
- Labels: `0 Assertion`, `1 Expressive`, `2 Neither`, `3 Question`, `4 Suggestion`.
- Input: one sentence (the hub splits reviews with pysbd). Max length 512 tokens.

## Provenance and limits

The checkpoint this model was fine-tuned from, the training data, the training procedure and the evaluation metrics are **not
documented**: `config.json` has no `_name_or_path` and the training code is not in the repository. No licence is stated for the
weights. Use it as a component of the hub, not as a validated standalone classifier.

## Use

```bash
python -m scripts.fetch_sage_models   # from the hub repository: downloads and checks both SAGE models into models/
```
