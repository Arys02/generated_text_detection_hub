# file: scripts/fetch_semdetect.py
"""Télécharge le code de Sem-Detect (Duarte et al., ICML 2026) au commit épinglé et l'installe dans le dépôt.

Le dépôt amont n'a pas de licence : son code n'est pas redistribué ici. Ce script le récupère depuis GitHub,
vérifie chaque fichier contre le manifeste SHA-256 versionné, puis le copie aux emplacements attendus par le code :
- <amont>/semdetect/                      -> src/detector/sem_detect/semdetect/
- <amont>/models/lightgbm_best_model.pkl  -> src/detector/sem_detect/models/lightgbm_best_model.pkl

Usage, depuis la racine du dépôt :
    python -m scripts.fetch_semdetect                   # installe (refuse d'écraser un fichier local différent)
    python -m scripts.fetch_semdetect --check           # vérifie l'installation existante, sans réseau
    python -m scripts.fetch_semdetect --write-manifest  # (mainteneur) régénère le manifeste depuis l'amont
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

UPSTREAM_URL = "https://github.com/avduarte333/Sem-Detect.git"
UPSTREAM_COMMIT = "d97175a513fab1aa97b31cc9a06874bad79a21c0"  # 40 caractères, voir src/detector/sem_detect/VENDORED.md
ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "src" / "detector" / "sem_detect" / "semdetect.sha256"
# (chemin dans le dépôt amont, chemin de destination dans ce dépôt)
TARGETS: tuple[tuple[str, str], ...] = (
    ("semdetect", "src/detector/sem_detect/semdetect"),
    ("models/lightgbm_best_model.pkl", "src/detector/sem_detect/models/lightgbm_best_model.pkl"),
)
IGNORED_PARTS = {"__pycache__"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def list_files(base: Path) -> list[Path]:
    if base.is_file():
        return [base]
    return sorted(
        p for p in base.rglob("*")
        if p.is_file() and not IGNORED_PARTS.intersection(p.parts) and p.suffix != ".pyc"
    )


def upstream_files(upstream: Path) -> dict[str, Path]:
    """Associe chaque chemin de destination (relatif à ROOT) au fichier amont correspondant."""
    files: dict[str, Path] = {}
    for src_rel, dest_rel in TARGETS:
        src = upstream / src_rel
        if not src.exists():
            raise SystemExit(f"Absent du dépôt amont : {src_rel}")
        for path in list_files(src):
            rel = path.relative_to(src) if src.is_dir() else Path()
            files[(Path(dest_rel) / rel).as_posix()] = path
    return files


def clone(workdir: Path) -> Path:
    if len(UPSTREAM_COMMIT) != 40:
        raise SystemExit("UPSTREAM_COMMIT doit être le hash complet (40 caractères).")
    target = workdir / "Sem-Detect"
    subprocess.run(["git", "clone", "--quiet", UPSTREAM_URL, str(target)], check=True)
    subprocess.run(["git", "-C", str(target), "checkout", "--quiet", UPSTREAM_COMMIT], check=True)
    head = subprocess.run(
        ["git", "-C", str(target), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    if head != UPSTREAM_COMMIT:
        raise SystemExit(f"Commit inattendu : {head}")
    return target


def read_manifest() -> dict[str, str]:
    if not MANIFEST.exists():
        raise SystemExit(f"Manifeste absent : {MANIFEST.relative_to(ROOT)}")
    entries: dict[str, str] = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, rel = line.split(maxsplit=1)
            entries[rel] = digest
    return entries


def write_manifest(files: dict[str, Path]) -> None:
    # Format compatible avec `sha256sum -c` (deux espaces entre empreinte et chemin).
    lines = [f"{sha256(path)}  {rel}" for rel, path in sorted(files.items())]
    MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Manifeste écrit : {len(lines)} fichiers -> {MANIFEST.relative_to(ROOT)}")


def check(manifest: dict[str, str]) -> int:
    problems = 0
    for rel, digest in sorted(manifest.items()):
        dest = ROOT / rel
        if not dest.exists():
            print(f"ABSENT     {rel}")
            problems += 1
        elif sha256(dest) != digest:
            print(f"DIFFÉRENT  {rel}")
            problems += 1
    for _, dest_rel in TARGETS:
        base = ROOT / dest_rel
        if base.is_dir():
            for path in list_files(base):
                rel = path.relative_to(ROOT).as_posix()
                if rel not in manifest:
                    print(f"EN TROP    {rel}")
                    problems += 1
    print("Installation conforme." if problems == 0 else f"{problems} écart(s).")
    return 0 if problems == 0 else 1


def install(files: dict[str, Path], manifest: dict[str, str], force: bool) -> int:
    if set(files) != set(manifest):
        for rel in sorted(set(files) ^ set(manifest)):
            print(f"Hors manifeste : {rel}")
        raise SystemExit("Le dépôt amont ne correspond pas au manifeste.")
    for rel, path in files.items():
        if sha256(path) != manifest[rel]:
            raise SystemExit(f"Empreinte amont inattendue : {rel}")
    conflicts = [rel for rel in files if (ROOT / rel).exists() and sha256(ROOT / rel) != manifest[rel]]
    if conflicts and not force:
        for rel in conflicts:
            print(f"Différent en local : {rel}")
        raise SystemExit("Fichiers locaux modifiés : relancer avec --force pour les remplacer.")
    copied = 0
    for rel, path in sorted(files.items()):
        dest = ROOT / rel
        if dest.exists() and sha256(dest) == manifest[rel]:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        copied += 1
    print(f"{copied} fichier(s) copié(s), {len(files) - copied} déjà conforme(s).")
    return check(manifest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="vérifie l'installation sans rien télécharger")
    mode.add_argument("--write-manifest", action="store_true", help="(mainteneur) régénère le manifeste")
    parser.add_argument("--force", action="store_true", help="remplace les fichiers locaux différents")
    args = parser.parse_args()
    if args.check:
        return check(read_manifest())
    with tempfile.TemporaryDirectory() as tmp:  # le clone vit le temps de la copie
        files = upstream_files(clone(Path(tmp)))
        if args.write_manifest:
            write_manifest(files)
            return 0
        return install(files, read_manifest(), args.force)


if __name__ == "__main__":
    sys.exit(main())