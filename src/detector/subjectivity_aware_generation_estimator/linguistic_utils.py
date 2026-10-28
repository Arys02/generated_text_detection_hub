## linguistic_utils.py
import pysbd
import re


_seg = pysbd.Segmenter(language="en", clean=False)  # instancier une seule fois, hors boucle

def segment_text(review_text: str) -> list[str]:
    return _seg.segment(review_text)

