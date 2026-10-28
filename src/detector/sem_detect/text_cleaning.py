# src/detector/sem_detect/text_cleaning.py
"""Nettoyages déterministes, appliqués à l'identique à toutes les reviews (cibles et références),
sans recours aux labels. Aucun appel LLM : pas de risque de troncature silencieuse."""
import re

from src.detector.sem_detect.semdetect.review_cleaner import strip_markdown

# marqueurs de liste en début de ligne : puces (-, *, •) et numéros (1. ou 1))
_LIST_MARKER = re.compile(r"(?m)^[ \t]*(?:[-*•]|\d+[.)])[ \t]+")


def clean_a(text: str) -> str:
    """(a) strip_markdown du code d'origine seul (titres, gras, liens, backticks)."""
    return strip_markdown(text or "")


def clean_b(text: str) -> str:
    """(b) (a) + retrait des marqueurs de liste : rapproche le format des reviews IA de celui
    du train de Sem-Detect (nettoyage LLM), au prix d'un écart pour les reviews humaines."""
    return _LIST_MARKER.sub("", strip_markdown(text or ""))


CLEANERS = {"a": clean_a, "b": clean_b}