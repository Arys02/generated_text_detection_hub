---
library_name: transformers
pipeline_tag: text-classification
tags: [roberta, regression, subjectivity, peer-review, sage]
language: en
---

# roberta-subjectivity-regressor

Subjectivity regressor used by SAGE (Subjectivity-Aware Generation Estimator) of
[generated_text_detection_hub](https://github.com/Arys02/generated_text_detection_hub): it scores the assertions of a review on a 1 to 5 scale.

- Architecture: `RobertaForSequenceClassification` with a single regression output (`problem_type: regression`), RoBERTa-base size.
- Output: one real score per sentence (the hub reads it as subjectivity from 1 to 5). Max length 512 tokens.

## Provenance and limits

The checkpoint this model was fine-tuned from, the training data, the training procedure and the evaluation metrics are **not
documented**: `config.json` has no `_name_or_path` and the training code is not in the repository. No licence is stated for the
weights. Use it as a component of the hub, not as a validated standalone regressor.

## Use

```bash
python -m scripts.fetch_sage_models   # from the hub repository: downloads and checks both SAGE models into models/
```
