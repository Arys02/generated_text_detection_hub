# file: src/detector/subjectivity_aware_generation_estimator/features/feature_spec.py
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Callable

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Constructeurs de colonnes
#
# Contrat : (block, ctx) -> np.ndarray de forme (L, width), dtype float64.
# `block` est le DataFrame d'UNE séquence (une review, un label), lignes déjà
# ordonnées. `ctx` porte les statistiques ajustées sur le train (moyennes,
# écarts-types) : rien de dépendant des données ne doit être recalculé ici,
# sous peine de fuite entre train et test.
# --------------------------------------------------------------------------- #


BuilderFn = Callable[[pd.DataFrame, dict, "FeatureGroup"], np.ndarray]
_BUILDERS: dict[str, BuilderFn] = {}


def builder(name: str):
    def deco(fn: BuilderFn) -> BuilderFn:
        if name in _BUILDERS:
            raise KeyError(f"builder déjà enregistré : {name}")
        _BUILDERS[name] = fn
        return fn

    return deco


@builder("speech_act_onehot")
def _speech_act_onehot(block: pd.DataFrame, ctx: dict, group: "FeatureGroup") -> np.ndarray:
    cats: tuple[str, ...] = tuple(ctx["label_cats"])
    idx = block["rule_based_SA-label"].map({c: i for i, c in enumerate(cats)})
    if idx.isna().any():
        unknown = sorted(set(block.loc[idx.isna(), "rule_based_SA-label"].astype(str)))
        raise ValueError(
            f"acte(s) de langage hors vocabulaire {unknown} ; attendu parmi {list(cats)}"
        )
    return np.eye(len(cats))[idx.to_numpy(dtype=int)]


@builder("subjectivity_zscore")
def _subjectivity_zscore(block: pd.DataFrame, ctx: dict, group: "FeatureGroup") -> np.ndarray:
    raw = block["raw_prediction_1_5"].to_numpy(dtype=float)
    z = np.where(np.isnan(raw), 0.0, (raw - ctx["subj_mean"]) / ctx["subj_std"])
    return z.reshape(-1, 1)


@builder("subjectivity_valid")
def _subjectivity_valid(block: pd.DataFrame, ctx: dict, group: "FeatureGroup") -> np.ndarray:
    raw = block["raw_prediction_1_5"].to_numpy(dtype=float)
    return (~np.isnan(raw)).astype(float).reshape(-1, 1)


@builder("speech_act_merged")
def _speech_act_merged(block: pd.DataFrame, ctx: dict, group: "FeatureGroup") -> np.ndarray:
    """Retirer une catégorie = la rendre indiscernable d'une AUTRE catégorie.
    La reléguer seule dans une colonne résiduelle ne détruit rien.

    Les positions sont conservées : la catégorie fusionnée devient une colonne
    constamment nulle, les autres gardent leur index. Largeur fixe — capacité
    constante entre conditions — et column_names reste véridique.
    """
    cats: tuple[str, ...] = tuple(ctx["label_cats"])
    merge = tuple(group.merge)
    target = group.merge_into

    if target in merge:
        raise ValueError(f"impossible de fusionner {target!r} dans elle-même")
    if target not in cats:
        raise ValueError(f"cible de fusion {target!r} hors vocabulaire {list(cats)}")
    unknown_merge = sorted(set(merge) - set(cats))
    if unknown_merge:
        raise ValueError(f"catégorie(s) à fusionner hors vocabulaire : {unknown_merge}")

    pos = {c: i for i, c in enumerate(cats)}
    remap = {c: (target if c in merge else c) for c in cats}

    labels = block["rule_based_SA-label"]
    unknown = sorted(set(labels.astype(str)) - set(cats))
    if unknown:
        raise ValueError(f"acte(s) hors vocabulaire {unknown}")

    idx = labels.map(lambda c: pos[remap[c]]).to_numpy(dtype=int)
    out = np.zeros((len(block), len(cats)))
    out[np.arange(len(block)), idx] = 1.0
    return out


# --------------------------------------------------------------------------- #
# Spécification
# --------------------------------------------------------------------------- #


class Scope(str, Enum):
    SENTENCE = "sentence"  # varie au sein de la séquence ; sensible à l'ordre
    REVIEW = "review"  # constante sur les L positions ; invariante par permutation


@dataclass(frozen=True)
class FeatureGroup:
    """Un joueur du jeu coopératif d'ablation, et un bloc de colonnes du tenseur."""
    name: str
    builder: str
    columns: tuple[str, ...]
    scope: Scope = Scope.SENTENCE
    ablation_value: float = 0.0
    requires: tuple[str, ...] = ()
    # paramètres d'encodage propres au groupe : ils DOIVENT vivre ici et non dans ctx,
    # sinon deux conditions expérimentales partagent le même spec.hash()
    merge: tuple[str, ...] = ()
    merge_into: str = "Assertion"

    @property
    def width(self) -> int:
        return len(self.columns)


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    segmentation: int
    subj_category: tuple[str, ...]
    groups: tuple[FeatureGroup, ...]
    label_cats: tuple[str, ...]

    def __post_init__(self) -> None:
        names = [g.name for g in self.groups]
        if len(set(names)) != len(names):
            raise ValueError(f"noms de groupes dupliqués : {names}")
        for g in self.groups:
            if g.builder not in _BUILDERS:
                raise KeyError(f"builder inconnu : {g.builder}")
            if g.merge and g.builder != "speech_act_merged":
                raise ValueError(f"groupe {g.name!r} : merge non supporté par {g.builder!r}")

        expected = LABEL_CATS_1 if self.segmentation == 1 else LABEL_CATS_2
        if self.segmentation not in (1, 2):
            raise ValueError(f"segmentation {self.segmentation} non supportée")
        if tuple(self.label_cats) != expected:
            raise ValueError(
                f"label_cats {list(self.label_cats)} incompatible avec "
                f"segmentation={self.segmentation} (attendu {list(expected)})"
            )
        # un scoring restreint à des catégories hors vocabulaire produirait des NaN muets
        unknown = sorted(set(self.subj_category) - set(self.label_cats))
        if unknown:
            raise ValueError(f"subj_category hors vocabulaire : {unknown}")

    def make_ctx(self, subj_mean: float, subj_std: float) -> dict:
        """Contexte de construction. label_cats et les paramètres de fusion sont
        dérivés de la spec : une seule source de vérité, pas de dérive possible."""
        if not np.isfinite(subj_std) or subj_std == 0:
            raise ValueError(f"subj_std dégénéré : {subj_std}")
        ctx = {"label_cats": self.label_cats,
               "subj_mean": float(subj_mean), "subj_std": float(subj_std)}
        for g in self.groups:
            if g.builder == "speech_act_merged":
                ctx["merge"] = g.merge
                ctx["merge_into"] = g.merge_into
        return ctx

    @property
    def input_dim(self) -> int:
        return sum(g.width for g in self.groups)

    @property
    def players(self) -> tuple[str, ...]:
        return tuple(g.name for g in self.groups)

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(c for g in self.groups for c in g.columns)

    @property
    def layout(self) -> dict[str, slice]:
        """Position de chaque groupe dans le tenseur. Dérivé, jamais écrit à la main."""
        out, off = {}, 0
        for g in self.groups:
            out[g.name] = slice(off, off + g.width)
            off += g.width
        return out

    def segmentation_key(self) -> dict:
        """Partie de la spec qui affecte la SEGMENTATION, donc le cache disque.
        Les groupes n'en font pas partie : ils se calculent depuis le CSV segmenté."""
        return {"segmentation": self.segmentation,
                "subj_category": sorted(self.subj_category)}

    def hash(self, n: int = 12) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, default=str).encode()
        return hashlib.md5(payload).hexdigest()[:n]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["input_dim"] = self.input_dim
        d["column_names"] = list(self.column_names)
        d["spec_hash"] = self.hash()
        return d

    def validate(self, df: pd.DataFrame) -> None:
        """Valide AVANT toute construction : un KeyError au milieu de la boucle
        ne dit pas quel groupe l'a causé."""
        needed = {c for g in self.groups for c in g.requires}
        missing = needed - set(df.columns)
        if missing:
            culprits = {g.name: sorted(set(g.requires) & missing)
                        for g in self.groups if set(g.requires) & missing}
            raise KeyError(f"colonnes absentes du DataFrame : {culprits}")

    @property
    def sentence_groups(self) -> tuple[str, ...]:
        return tuple(g.name for g in self.groups if g.scope is Scope.SENTENCE)

    @property
    def review_groups(self) -> tuple[str, ...]:
        return tuple(g.name for g in self.groups if g.scope is Scope.REVIEW)


# --------------------------------------------------------------------------- #
# Registre
# --------------------------------------------------------------------------- #

LABEL_CATS_1 = ("Assertion", "Neither", "Question")
LABEL_CATS_2 = ("Assertion", "Expressive", "Neither", "Question", "Suggestion")

SA_REQ = ("rule_based_SA-label",)
SUBJ_REQ = ("raw_prediction_1_5",)

_SUBJ_Z = FeatureGroup("subj", "subjectivity_zscore", ("subj_z",), requires=SUBJ_REQ)
_SUBJ_VALID = FeatureGroup("valid", "subjectivity_valid", ("subj_valid",), requires=SUBJ_REQ)

_MULTI = ("Assertion", "Expressive", "Question", "Suggestion")


def _cat_group(cats: tuple[str, ...], merge: tuple[str, ...] = ()) -> FeatureGroup:
    if not merge:
        return FeatureGroup("cat", "speech_act_onehot", cats, requires=SA_REQ)
    return FeatureGroup("cat", "speech_act_merged", cats, requires=SA_REQ,
                        merge=merge, merge_into="Assertion")


SPECS: dict[str, FeatureSpec] = {
    "v1_base": FeatureSpec(
        name="v1_base", segmentation=1, subj_category=("Assertion",),
        label_cats=LABEL_CATS_1,
        groups=(_cat_group(LABEL_CATS_1), _SUBJ_Z, _SUBJ_VALID),
    ),
    "v2_assertion": FeatureSpec(
        name="v2_assertion", segmentation=2, subj_category=("Assertion",),
        label_cats=LABEL_CATS_2,
        groups=(_cat_group(LABEL_CATS_2), _SUBJ_Z, _SUBJ_VALID),
    ),
    "v2_multi": FeatureSpec(
        name="v2_multi", segmentation=2, subj_category=_MULTI,
        label_cats=LABEL_CATS_2,
        groups=(_cat_group(LABEL_CATS_2), _SUBJ_Z, _SUBJ_VALID),
    ),
    # v3 : sans valid. En segmentation=2, valid = 1 - onehot[Neither] : colonne
    # dupliquée, dont la présence dilue l'importance de Neither entre deux canaux.
    "v3_multi_novalid": FeatureSpec(
        name="v3_multi_novalid", segmentation=2, subj_category=_MULTI,
        label_cats=LABEL_CATS_2,
        groups=(_cat_group(LABEL_CATS_2), _SUBJ_Z),
    ),
}



# conditions de fusion : une spec par catégorie, donc un hash par condition
for _c in LABEL_CATS_2:
    if _c != "Assertion":
        _n = f"v3_merge_{_c.lower()}"
        SPECS[_n] = FeatureSpec(
            name=_n, segmentation=2, subj_category=_MULTI, label_cats=LABEL_CATS_2,
            groups=(_cat_group(LABEL_CATS_2, merge=(_c,)), _SUBJ_Z),
        )
        
SPECS["v3_cat_only"] = FeatureSpec(
    name="v3_cat_only", segmentation=2, subj_category=_MULTI, label_cats=LABEL_CATS_2,
    groups=(_cat_group(LABEL_CATS_2),),
)
SPECS["v3_subj_only"] = FeatureSpec(
    name="v3_subj_only", segmentation=2, subj_category=_MULTI, label_cats=LABEL_CATS_2,
    groups=(_SUBJ_Z,),
)

def get_spec(name: str) -> FeatureSpec:
    if name not in SPECS:
        raise ValueError(f"spec {name!r} inconnue ; disponibles : {sorted(SPECS)}")
    return SPECS[name]
