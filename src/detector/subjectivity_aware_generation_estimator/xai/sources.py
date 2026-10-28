# file: src/detector/subjectivity_aware_generation_estimator/xai/sources.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Source:
    key: str
    citation: str
    url: str
    bibtex: str


SOURCES: dict[str, Source] = {
    "molnar_iml": Source(
        key="molnar_iml",
        citation="Molnar, C. Interpretable Machine Learning (édition en ligne).",
        url="https://christophm.github.io/interpretable-ml-book/",
        bibtex="""@book{molnar_iml,
  author = {Molnar, Christoph},
  title  = {Interpretable Machine Learning},
  note   = {Édition en ligne, consultée en 2026},
  url    = {https://christophm.github.io/interpretable-ml-book/}
}""",
    ),
    "molnar_pfi": Source(
        key="molnar_pfi",
        citation="Molnar, C. Interpretable Machine Learning, ch. Permutation Feature Importance.",
        url="https://christophm.github.io/interpretable-ml-book/feature-importance.html",
        bibtex="""@incollection{molnar_pfi,
  author    = {Molnar, Christoph},
  title     = {Permutation Feature Importance},
  booktitle = {Interpretable Machine Learning},
  url       = {https://christophm.github.io/interpretable-ml-book/feature-importance.html}
}""",
    ),
    "molnar_lofo": Source(
        key="molnar_lofo",
        citation="Molnar, C. Interpretable Machine Learning, ch. Leave One Feature Out Importance.",
        url="https://christophm.github.io/interpretable-ml-book/lofo.html",
        bibtex="""@incollection{molnar_lofo,
  author    = {Molnar, Christoph},
  title     = {Leave One Feature Out (LOFO) Importance},
  booktitle = {Interpretable Machine Learning},
  url       = {https://christophm.github.io/interpretable-ml-book/lofo.html}
}""",
    ),
    "breiman2001": Source(
        key="breiman2001",
        citation="Breiman, L. (2001). Random Forests. Machine Learning 45(1), 5-32.",
        url="https://doi.org/10.1023/A:1010933404324",
        bibtex="""@article{breiman2001random,
  author  = {Breiman, Leo},
  title   = {Random Forests},
  journal = {Machine Learning},
  volume  = {45}, number = {1}, pages = {5--32}, year = {2001},
  doi     = {10.1023/A:1010933404324}
}""",
    ),
    "fisher2019": Source(
        key="fisher2019",
        citation=("Fisher, A., Rudin, C., Dominici, F. (2019). All Models Are Wrong, but Many "
                  "Are Useful. JMLR 20:177."),
        url="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8323609/",
        bibtex="""@article{fisher2019all,
  author  = {Fisher, Aaron and Rudin, Cynthia and Dominici, Francesca},
  title   = {All Models Are Wrong, but Many Are Useful: Learning a Variable's Importance
             by Studying an Entire Class of Prediction Models Simultaneously},
  journal = {Journal of Machine Learning Research},
  volume  = {20}, number = {177}, year = {2019}
}""",
    ),
    "lei2018": Source(
        key="lei2018",
        citation=("Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J., Wasserman, L. (2018). "
                  "Distribution-Free Predictive Inference for Regression. JASA 113(523)."),
        url="https://doi.org/10.1080/01621459.2017.1307116",
        bibtex="""@article{lei2018distributionfree,
  author  = {Lei, Jing and G'Sell, Max and Rinaldo, Alessandro and
             Tibshirani, Ryan J. and Wasserman, Larry},
  title   = {Distribution-Free Predictive Inference for Regression},
  journal = {Journal of the American Statistical Association},
  volume  = {113}, number = {523}, pages = {1094--1111}, year = {2018},
  doi     = {10.1080/01621459.2017.1307116}
}""",
    ),
    "molnar2023subgroups": Source(
        key="molnar2023subgroups",
        citation=("Molnar, C., König, G., Bischl, B., Casalicchio, G. (2023). Model-Agnostic "
                  "Feature Importance and Effects with Dependent Features. DMKD."),
        url="https://doi.org/10.1007/s10618-022-00901-9",
        bibtex="""@article{molnar2023modelagnostic,
  author  = {Molnar, Christoph and K{\\"o}nig, Gunnar and Bischl, Bernd and
             Casalicchio, Giuseppe},
  title   = {Model-Agnostic Feature Importance and Effects with Dependent Features --
             A Conditional Subgroup Approach},
  journal = {Data Mining and Knowledge Discovery}, year = {2023},
  doi     = {10.1007/s10618-022-00901-9}
}""",
    ),
    "dugan2024raid": Source(
        key="dugan2024raid",
        citation=("Dugan, L. et al. (2024). RAID: A Shared Benchmark for Robust Evaluation of "
                  "Machine-Generated Text Detectors. ACL 2024, 12463-12492."),
        url="https://aclanthology.org/2024.acl-long.674/",
        bibtex="""@inproceedings{dugan-etal-2024-raid,
  author    = {Dugan, Liam and Hwang, Alyssa and Trhl{\\'i}k, Filip and Zhu, Andrew and
               Ludan, Josh Magnus and Xu, Hainiu and Ippolito, Daphne and
               Callison-Burch, Chris},
  title     = {{RAID}: A Shared Benchmark for Robust Evaluation of Machine-Generated
               Text Detectors},
  booktitle = {Proceedings of the 62nd Annual Meeting of the ACL (Volume 1: Long Papers)},
  pages     = {12463--12492}, year = {2024},
  url       = {https://aclanthology.org/2024.acl-long.674/}
}""",
    ),
    "bento2021timeshap": Source(
        key="bento2021timeshap",
        citation=("Bento, J., Saleiro, P., Cruz, A. F., Figueiredo, M. A. T., Bizarro, P. (2021). "
                  "TimeSHAP: Explaining Recurrent Models through Sequence Perturbations. KDD."),
        url="https://arxiv.org/abs/2012.00073",
        bibtex="""@inproceedings{bento2021timeshap,
  author    = {Bento, Jo{\\~a}o and Saleiro, Pedro and Cruz, Andr{\\'e} F. and
               Figueiredo, M{\\'a}rio A. T. and Bizarro, Pedro},
  title     = {{TimeSHAP}: Explaining Recurrent Models through Sequence Perturbations},
  booktitle = {Proceedings of the 27th ACM SIGKDD Conference on KDD},
  pages     = {2565--2573}, year = {2021},
  doi       = {10.1145/3447548.3467166}
}""",
    ),
}


def bibliography(keys: list[str] | None = None) -> str:
    keys = sorted(SOURCES) if keys is None else keys
    return "\n\n".join(SOURCES[k].bibtex for k in keys)