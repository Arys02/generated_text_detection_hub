# file: src/detector/subjectivity_aware_generation_estimator/xai/experiments.py
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Axis(str, Enum):
    CONFOUNDERS = "confondants"  # ce que le détecteur ne mesure PAS
    FEATURES = "blocs_features"  # axe 1
    TEMPORALITY = "temporalite"  # axe 2
    CATEGORIES = "categories"  # axe 3


class Method(str, Enum):
    DESCRIPTIVE = "descriptif"  # niveau données, aucun modèle
    BASELINE = "baseline_supervisee"  # modèle simple sur features de surface
    PFI = "pfi"  # perturbation, modèle figé
    LOFO = "lofo"  # réentraînement sans l'information
    INTERVENTION = "intervention_dirigee"  # conversion vers une valeur, donne le signe
    ABLATION_INFERENCE = "ablation_inference"  # diagnostic, hors distribution
    PERMUTE_RETRAIN = "permutation_reapprentissage"


@dataclass(frozen=True)
class Experiment:
    key: str
    axis: Axis
    method: Method
    question: str
    procedure: str
    sources: tuple[str, ...]
    limits: tuple[str, ...] = field(default_factory=tuple)
    replicate_kind: str = "seed"  # seed | permutation | none


EXPERIMENTS: dict[str, Experiment] = {
    # ---------------- confondants ----------------
    "longueur_univariee": Experiment(
        key="longueur_univariee", axis=Axis.CONFOUNDERS, method=Method.DESCRIPTIVE,
        question="Le nombre de phrases suffit-il à séparer les classes ?",
        procedure="AUC du nombre de phrases par relecture, sans modèle.",
        sources=("molnar_iml",), replicate_kind="none",
    ),
    "baseline_formatage": Experiment(
        key="baseline_formatage", axis=Axis.CONFOUNDERS, method=Method.BASELINE,
        question="Le balisage typographique suffit-il ?",
        procedure=("Régression logistique en validation croisée stratifiée sur des "
                   "indicateurs Markdown agrégés par relecture."),
        sources=("dugan2024raid",), replicate_kind="none",
    ),
    "baseline_structurelle": Experiment(
        key="baseline_structurelle", axis=Axis.CONFOUNDERS, method=Method.BASELINE,
        question="Le plan stéréotypé de relecture suffit-il ?",
        procedure=("Régression logistique sur titres de section, lignes d'étiquette et "
                   "fragments Neither, sur corpus normalisé, avec et sans la longueur."),
        sources=("dugan2024raid",),
        limits=("Seuil de décision fixé avant mesure : <0.85 confondant mineur.",),
        replicate_kind="none",
    ),
    "normalisation_markdown": Experiment(
        key="normalisation_markdown", axis=Axis.CONFOUNDERS, method=Method.DESCRIPTIVE,
        question="Le balisage contamine-t-il les annotateurs en amont ?",
        procedure=("Suppression symétrique des marques dans le texte brut, resegmentation "
                   "complète, comparaison des profils de catégories avant/après."),
        sources=("molnar_iml",), replicate_kind="none",
    ),

    # ---------------- axe 1 : blocs de features ----------------
    "pfi_blocs": Experiment(
        key="pfi_blocs", axis=Axis.FEATURES, method=Method.PFI,
        question="Sur quels blocs ce GRU s'appuie-t-il ?",
        procedure=("Rééchantillonnage global à marginales conservées, modèle figé. "
                   "Le lien feature-étiquette est rompu sans modifier les distributions "
                   "marginales, ce qui évite les instances irréalistes."),
        sources=("molnar_pfi", "breiman2001", "fisher2019"),
        limits=("Interprétation marginale : une feature redondante obtient une importance "
                "faible même si elle est informative.",),
        replicate_kind="permutation",
    ),
    "lofo_blocs": Experiment(
        key="lofo_blocs", axis=Axis.FEATURES, method=Method.LOFO,
        question="Que vaut chaque bloc d'information en autonomie ?",
        procedure=("Réentraînement sans le bloc, hyperparamètres figés. Conditions : "
                   "référence, catégories seules, subjectivité seule, degré seul, validité seule."),
        sources=("molnar_lofo", "lei2018"),
        limits=("L'interprétation porte sur l'apprenant, pas seulement sur le modèle inspecté.",
                "Hyperparamètres figés : une condition pourrait exiger plus de capacité."),
    ),

    # ---------------- axe 2 : temporalité ----------------
    "ordre_inference": Experiment(
        key="ordre_inference", axis=Axis.TEMPORALITY, method=Method.PFI,
        question="Ce GRU dépend-il de l'ordre des phrases ?",
        procedure=("Mélange intra-relecture et inversion, modèle figé. La composition est "
                   "strictement conservée : seule l'information d'ordre est détruite."),
        sources=("molnar_pfi", "bento2021timeshap"),
        limits=("Mélange le biais de récence du modèle et l'information d'ordre réelle.",),
        replicate_kind="permutation",
    ),
    "ordre_troncature": Experiment(
        key="ordre_troncature", axis=Axis.TEMPORALITY, method=Method.ABLATION_INFERENCE,
        question="Quelle part du signal se trouve au début, à la fin ?",
        procedure="Conservation des k premières ou k dernières phrases, modèle figé.",
        sources=("bento2021timeshap",),
        limits=("Séquences hors distribution : diagnostic, pas mesure.",
                "La longueur varie avec k, sauf sous filtre de longueur constante."),
        replicate_kind="none",
    ),
    "ordre_lofo": Experiment(
        key="ordre_lofo", axis=Axis.TEMPORALITY, method=Method.LOFO,
        question="L'ordre porte-t-il de l'information exploitable ?",
        procedure=("Grille 2×2 : mélange appliqué au train, au test, aux deux, à aucun. "
                   "Seule la condition train+test mesure l'information d'ordre."),
        sources=("molnar_lofo", "lei2018"),
        limits=("Le mélange détruit aussi le groupement local de catégories.",),
    ),

    # ---------------- axe 3 : catégories ----------------
    "profil_categories": Experiment(
        key="profil_categories", axis=Axis.CATEGORIES, method=Method.DESCRIPTIVE,
        question="Que contiennent les données, indépendamment de tout modèle ?",
        procedure=("Prévalence par classe, AUC univariée du taux, probabilité MGT "
                   "conditionnelle à la présence."),
        sources=("molnar_iml",),
        limits=("Taux compositionnels : ils somment à un, les effets sont couplés.",
                "Catégories prédites, non annotées."),
        replicate_kind="none",
    ),
    "pfi_categories": Experiment(
        key="pfi_categories", axis=Axis.CATEGORIES, method=Method.PFI,
        question="Quelle catégorie ce GRU exploite-t-il le plus ?",
        procedure=("Rééchantillonnage bidirectionnel d'une catégorie à prévalence conservée, "
                   "score de subjectivité réconcilié. ΔAUC rapporté à la fraction de phrases "
                   "effectivement modifiée, 2p(1-p), pour rendre les catégories comparables."),
        sources=("molnar_pfi", "fisher2019"),
        limits=("Les importances ne s'additionnent pas : les interactions sont comptées "
                "dans chaque feature concernée, la somme dépasse la perte totale.",
                "La normalisation par la dose est une approximation de premier ordre."),
        replicate_kind="permutation",
    ),
    "lofo_fusion_categories": Experiment(
        key="lofo_fusion_categories", axis=Axis.CATEGORIES, method=Method.LOFO,
        question="Quelle distinction de catégorie apporte de l'information propre ?",
        procedure=("Fusion de la catégorie dans Assertion, à l'entraînement et au test, "
                   "puis réentraînement. Les positions du one-hot sont conservées : la "
                   "catégorie fusionnée devient une colonne nulle, la capacité reste constante. "
                   "Fusion et non suppression : supprimer une colonne du one-hot ne retire "
                   "aucune information, le vecteur nul restant identifiable."),
        sources=("molnar_lofo", "lei2018"),
        limits=("Des catégories redondantes obtiennent toutes deux une importance faible.",),
    ),
    "lofo_bruit_categories": Experiment(
        key="lofo_bruit_categories", axis=Axis.CATEGORIES, method=Method.PERMUTE_RETRAIN,
        question="Que perd-on si l'annotation d'une catégorie devient du bruit ?",
        procedure=("Rééchantillonnage bidirectionnel de l'indicateur de catégorie à "
                   "prévalence conservée, appliqué au train ET au test, puis "
                   "réentraînement. Variante du LOFO signalée par Molnar : on ajoute "
                   "une version permutée de la variable plutôt que de la retirer, ce "
                   "qui conserve le nombre de features et donc la capacité du modèle."),
        sources=("molnar_lofo", "molnar_pfi", "lei2018"),
        limits=("La permutation est elle-même aléatoire : elle ajoute une source "
                "d'incertitude par rapport au LOFO par ablation.",
                "Le score de subjectivité reste attaché à la phrase : si la "
                "subjectivité covarie avec la catégorie d'origine, une part de "
                "l'information fuit par ce canal et le delta sous-estime l'effet."),
    ),
    "direction_categories": Experiment(
        key="direction_categories", axis=Axis.CATEGORIES, method=Method.INTERVENTION,
        question="Dans quel sens chaque catégorie pousse-t-elle la décision ?",
        procedure=("Conversion d'une fraction de phrases vers la catégorie cible, mesure du "
                   "déplacement de la probabilité MGT moyenne. Plusieurs doses pour vérifier "
                   "la monotonie de la réponse."),
        sources=("molnar_pfi",),
        limits=("Comble une limite explicite de la PFI, qui donne l'ampleur et non la "
                "direction de l'effet.",
                "Les entrées converties peuvent s'éloigner de la distribution d'entraînement."),
        replicate_kind="permutation",
    ),
}


def by_axis(axis: Axis) -> list[Experiment]:
    return [e for e in EXPERIMENTS.values() if e.axis is axis]
