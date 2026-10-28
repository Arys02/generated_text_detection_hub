# file: src/detector/subjectivity_aware_generation_estimator/xai/text_normalization.py
from __future__ import annotations

import re

import pandas as pd

_SEPARATOR_LINE = re.compile(r"^[ \t]*(?:[=\-_*#~]{4,})[ \t]*$", re.MULTILINE)
_ATX_HEADER = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]*", re.MULTILINE)
_BULLET = re.compile(r"^[ \t]*(?:[-*+•]|\d{1,2}[.)])[ \t]+", re.MULTILINE)
_EMPHASIS = re.compile(r"(\*{1,3}|_{2,3}|`+)")
_BLOCKQUOTE = re.compile(r"^[ \t]{0,3}>[ \t]?", re.MULTILINE)
_MULTI_BLANK = re.compile(r"\n{3,}")
_TRAILING_WS = re.compile(r"[ \t]+$", re.MULTILINE)


def normalize_markdown(text: str) -> str:
    """Retire les marques de mise en forme, symétriquement pour les deux classes.

    Supprime : lignes de séparation (==== côté humain, #### côté généré), titres ATX,
    puces et numérotation en début de ligne, emphase (**, *, __, `), citations.
    Le contenu lexical est préservé : seules les marques sont retirées, pas les mots.
    """
    if not isinstance(text, str):
        return ""
    t = _SEPARATOR_LINE.sub("", text)
    t = _ATX_HEADER.sub("", t)
    t = _BLOCKQUOTE.sub("", t)
    t = _BULLET.sub("", t)
    t = _EMPHASIS.sub("", t)
    t = _TRAILING_WS.sub("", t)
    t = _MULTI_BLANK.sub("\n\n", t)
    return t.strip()


def normalize_dataframe(df: pd.DataFrame, column: str = "content") -> pd.DataFrame:
    out = df.copy()
    out[column] = out[column].map(normalize_markdown)
    return out