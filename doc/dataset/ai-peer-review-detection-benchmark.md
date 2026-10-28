# AI Peer Review Detection Benchmark (APRDB)

- Hugging Face: `IntelLabs/AI-Peer-Review-Detection-Benchmark` (viewer: https://huggingface.co/datasets/IntelLabs/AI-Peer-Review-Detection-Benchmark)
- Methodology paper: https://arxiv.org/abs/2502.19614 (v2)
- Size reported on the dataset card: 788,984 AI-generated reviews.
- Loader in this repository: `src/data/dataset_aprdb.py` (see `doc/datasets.md`).
- Licence: `license: other` (Intel research and development licence, research use only on the dataset card). Reviews of human origin come from OpenReview (CC BY 4.0) and ASAP-Review (Apache 2.0) according to the card.

## The `label` column

The dataset has a `label` column that is documented neither on the Hugging Face card nor in the paper. In this project `label = 0` is read as human-written and `label = 1` as machine-generated (the code only splits on `label == 1` / `label == 0`, `dataset_aprdb.py`); the mapping is the project's interpretation and could not be checked against upstream documentation (audit anomaly A30).

## Fields by conference (from the dataset card as recorded by the project)

| Conference | Fields |
|---|---|
| ICLR 2017-2019, 2021 | review, rating, confidence |
| ICLR 2020 | review, rating, confidence, experience assessment, checking correctness of derivations and theory, checking correctness of experiments, thoroughness in paper reading |
| ICLR 2022 | summary of the paper, main review, summary of the review, correctness, technical novelty and significance, empirical novelty and significance, flag for ethics review, recommendation, confidence |
| ICLR 2023 | summary of the paper, strength and weaknesses, clarity quality novelty and reproducibility, summary of the review, rating, confidence |
| ICLR 2024 | summary, strengths, weaknesses, questions, soundness, presentation, contribution, flag for ethics review, rating, confidence |
| NeurIPS 2016-2017 | review, rating, confidence |
| NeurIPS 2018 | review, overall score, confidence score |
| NeurIPS 2019 | review, overall score, confidence score, contribution |
| NeurIPS 2021 | summary, main review, limitations and societal impact, rating, confidence, needs ethics review, ethics review area |
| NeurIPS 2022 | summary, strengths and weaknesses, questions, limitations, ethics flag, ethics review area, rating, confidence, soundness, presentation, contribution |
| NeurIPS 2023-2024 | summary, strengths, weaknesses, questions, limitations, ethics flag, ethics review area, rating, confidence, soundness, presentation, contribution |
