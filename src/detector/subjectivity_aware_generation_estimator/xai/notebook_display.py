# file: src/detector/subjectivity_aware_generation_estimator/xai/notebook_display.py
from __future__ import annotations

from pathlib import Path

from IPython.display import Code, Image, Markdown, display


def show_latex(path: Path | str, title: str | None = None) -> None:
    """Imprime le source LaTeX dans la sortie du notebook, coloré si Pygments est là."""
    path = Path(path)
    src = path.read_text(encoding="utf-8")
    display(Markdown(f"#### {title or path.name}"))
    try:
        display(Code(src, language="latex"))
    except Exception:
        # Code() échoue si Pygments manque : on retombe sur du texte brut
        print(src)


def show_figure(path: Path | str, title: str | None = None, width: int = 720) -> None:
    """Affiche la version PNG d'une figure exportée (le PDF n'est pas rendu en notebook)."""
    png = Path(path).with_suffix(".png")
    if not png.exists():
        print(f"[manquant] {png}")
        return
    if title:
        display(Markdown(f"#### {title}"))
    display(Image(filename=str(png), width=width))


def show_all(paths: dict[str, Path] | list[Path], kind: str, **kwargs) -> None:
    items = paths.items() if isinstance(paths, dict) else [(p.stem, p) for p in paths]
    for name, p in items:
        (show_latex if kind == "latex" else show_figure)(p, title=name, **kwargs)