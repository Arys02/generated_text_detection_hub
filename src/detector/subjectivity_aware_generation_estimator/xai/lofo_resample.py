# file: src/detector/subjectivity_aware_generation_estimator/xai/lofo_resample.py
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from src.detector.subjectivity_aware_generation_estimator.features.feature_spec import FeatureSpec
from src.detector.subjectivity_aware_generation_estimator.xai.category_permutation import (
    ALL, fit_marginals, resample_category,
)

TransformFn = Callable[[pd.DataFrame, np.random.Generator], pd.DataFrame]


def make_noise_transform(target: str, spec: FeatureSpec) -> TransformFn:
    """Rend la catégorie `target` non informative, marginales conservées.

    Chaque phrase devient `target` avec probabilité p_target indépendamment de sa
    catégorie d'origine ; celles qui la perdent reçoivent une catégorie tirée dans
    P(C | C != target). L'indicateur 1[C = target] garde donc sa prévalence mais
    perd tout lien avec l'étiquette.

    Les marginales sont ajustées sur le DataFrame transformé lui-même, ce qui
    reproduit exactement une permutation de l'indicateur au sein de ce jeu : les
    ajuster sur le train introduirait un décalage de distribution au test.
    """
    label_cats = list(spec.label_cats)
    scored = list(spec.subj_category)

    def transform(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
        marginals = fit_marginals(df, label_cats, scored)
        return resample_category(df, target, marginals, rng)

    return transform


def noise_conditions(spec: FeatureSpec, spec_name: str,
                     include_all: bool = True) -> dict[str, tuple]:
    """Conditions prêtes pour run_grid : {nom -> (spec, transform_train, transform_test)}.

    La même fabrique est utilisée pour le train et le test, mais avec le même `rng` :
    les deux tirages sont donc indépendants, ce qui est correct — ce sont deux jeux.
    """
    conditions: dict[str, tuple] = {"reference": (spec_name, None, None)}
    targets = list(spec.label_cats) + ([ALL] if include_all else [])
    for target in targets:
        fn = make_noise_transform(target, spec)
        name = "bruit_TOUTES" if target == ALL else f"bruit_{target}"
        conditions[name] = (spec_name, fn, fn)
    return conditions