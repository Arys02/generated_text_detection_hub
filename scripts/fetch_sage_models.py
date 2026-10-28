# file: scripts/fetch_sage_models.py
"""Télécharge les deux modèles RoBERTa de segmentation de SAGE depuis le Hugging Face Hub et les installe dans `models/`.

- roberta_speech_act_classifier : acte de langage de chaque phrase (5 classes : Assertion, Expressive, Neither, Question, Suggestion)
- roberta_subjectivity_regressor : score de subjectivité de chaque assertion (régression, échelle 1 à 5)

Chaque fichier est vérifié contre `scripts/sage_models.sha256`. `linguistic_segmentation.py` lit `<repo>/models/` dès que
`models/roberta_speech_act_classifier` existe (ou le dossier désigné par la variable SAGE_MODELS_DIR).

Usage, depuis la racine du dépôt :
    python -m scripts.fetch_sage_models           # télécharge et vérifie
    python -m scripts.fetch_sage_models --check   # vérifie l'installation existante, sans réseau
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "scripts" / "sage_models.sha256"
DEST = Path(os.getenv("SAGE_MODELS_DIR") or ROOT / "models")
HF_NAMESPACE = "Arys02"  # compte Hugging Face qui héberge les modèles
# modèle -> commit Hugging Face épinglé (poids + fiche de modèle)
MODELS = {
    "roberta_speech_act_classifier": "9fb6d2429a35aa302701522edf542822f5acca5b",
    "roberta_subjectivity_regressor": "0b6ff5709c6802998996ef94f2ec47d3b459b11e",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check() -> int:
    bad = 0
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        digest, rel = line.split(maxsplit=1)
        path = DEST / rel
        if not path.is_file():
            print(f"manquant : {rel}")
            bad += 1
        elif _sha256(path) != digest:
            print(f"différent : {rel}")
            bad += 1
    print("Installation conforme." if not bad else f"{bad} fichier(s) manquant(s) ou différent(s).")
    return 1 if bad else 0


def fetch() -> int:
    from huggingface_hub import snapshot_download

    DEST.mkdir(parents=True, exist_ok=True)
    for name, revision in MODELS.items():
        snapshot_download(
            repo_id=f"{HF_NAMESPACE}/{name.replace('_', '-')}", revision=revision, local_dir=DEST / name,
            allow_patterns=["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"],
        )
    return check()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="vérifie sans télécharger")
    sys.exit(check() if ap.parse_args().check else fetch())
