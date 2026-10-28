# file: scripts/check_structure.py
"""Contrôle structurel du repo : vérifie que le code s'importe et que les configs se composent, sans rien entraîner ni évaluer.

Contrôles :
1. imports   : chaque module de src/ s'importe (sauf NEVER_IMPORT, modules à effets de bord dangereux) ;
2. presets   : chaque preset de conf/experiment/{train,eval}/ se compose avec Hydra, ses interpolations se résolvent,
               son train_preset éventuel aussi, et les noms de détecteur/dataset existent dans les factories ;
3. commandes : chaque combinaison (script, paramètres) des configurations PyCharm et des runs Hydra
               (extracts d'audit run_configs.json et hydra_runs.json, si présents) se compose ;
4. notebooks : chaque `from src... import nom` des extraits de notebooks (extracts d'audit) pointe
               vers un module et un nom existants ;
5. cli       : un lancement réel `python -m <script> <preset> --cfg job` par script Hydra
               (affiche la config composée sans exécuter la tâche).

Usage, depuis la racine du repo, avec .venv actif et un .env valide :
    python -m scripts.check_structure --save scripts/check_structure_baseline.json
    python -m scripts.check_structure --compare scripts/check_structure_baseline.json
Avec --compare, le code de sortie vaut 1 si un contrôle réussi dans la référence échoue maintenant,
ou si un nouveau contrôle échoue.

Effet de bord connu : importer cristal_anchor crée example.log dans le répertoire courant (ignoré par git).
"""
from __future__ import annotations

import argparse
import ast
import importlib
import json
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path.cwd()
SRC = ROOT / "src"
CONF = ROOT / "conf"
EXTRACTS = ROOT / "audit" / "extracts"
SAGE = "src.detector.subjectivity_aware_generation_estimator"
NEVER_IMPORT = {
    "src.detector.cristal_anchor.finetuning_anchor",  # lance un fine-tuning à l'import
    "src.detector.cristal_anchor.script_fetch_paper",  # télécharge des articles et écrit data.pkl à l'import
}
HYDRA_SCRIPTS = {  # module -> (config_name par défaut, groupe de presets, preset témoin du contrôle cli)
    "src.evaluation.run_eval": ("eval", "experiment/eval", "sage_gru_1"),
    f"{SAGE}.training.gru_train": ("train", "experiment/train", "train_sage_gru_v3"),
    f"{SAGE}.training.mlp_train": ("train", "experiment/train", "train_sage_mlp_1"),
    f"{SAGE}.training.transformer_train": ("train", "experiment/train", "train_sage_transformer_2_cat"),
}
JOB_TO_MODULE = {module.rsplit(".", 1)[1]: module for module in HYDRA_SCRIPTS}
FACTORY_FILES = {"detector": SRC / "detector" / "detector_factory.py", "dataset": SRC / "data" / "dataset_factory.py"}
CFG_NAME_KEYS = (("detector", "detector"), ("dataset", "dataset"), ("dataset_train", "dataset"), ("dataset_val", "dataset"))
CELL_MARKER = re.compile(r"^# %% \[cellule .*$", re.MULTILINE)

Result = dict[str, object]


def short(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}".strip()
    return (text.splitlines() or [text])[0][:300]


def ok(detail: str = "") -> Result:
    return {"ok": True, "detail": detail}


def fail(detail: str) -> Result:
    return {"ok": False, "detail": detail}


def src_modules() -> list[tuple[str, bool]]:
    modules = []
    for path in sorted(SRC.rglob("*.py")):
        parts = list(path.relative_to(ROOT).parts)
        if any(p in {"outputs", "__pycache__"} for p in parts):
            continue
        parts[-1] = parts[-1][: -len(".py")]
        if parts[-1] == "__init__":
            parts = parts[:-1]
        modules.append((".".join(parts), all(p.isidentifier() for p in parts)))
    return modules


def try_import(module: str) -> Result:
    if module in NEVER_IMPORT:
        return ok("non importé : effets de bord dangereux (NEVER_IMPORT)")
    try:
        importlib.import_module(module)
        return ok()
    except (Exception, SystemExit) as exc:
        return fail(short(exc))


def check_imports() -> dict[str, Result]:
    out = {}
    for module, importable in src_modules():
        out[module] = try_import(module) if importable else fail("nom de fichier non importable comme module")
    return out


def factory_names() -> dict[str, set[str]]:
    pattern = re.compile(r"==\s*['\"]([\w-]+)['\"]")
    return {kind: set(pattern.findall(path.read_text(encoding="utf-8"))) for kind, path in FACTORY_FILES.items()}


def compose_cfg(config_name: str, overrides: list[str]):
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF), version_base=None):
        return compose(config_name=config_name, overrides=overrides)


def cfg_problems(cfg, names: dict[str, set[str]]) -> list[str]:
    from omegaconf import OmegaConf

    try:
        container = OmegaConf.to_container(cfg, resolve=True)
    except Exception as exc:  # les interpolations ${hydra:...} ne se résolvent pas hors application Hydra
        if "hydra" not in str(exc).lower():
            raise
        container = OmegaConf.to_container(cfg, resolve=False)
    problems = []
    for key, kind in CFG_NAME_KEYS:
        node = container.get(key) if isinstance(container, dict) else None
        if isinstance(node, dict) and "name" in node and node["name"] not in names[kind]:
            problems.append(f"{key}.name={node['name']} absent de {FACTORY_FILES[kind].name}")
    return problems


def check_composition(config_name: str, overrides: list[str], names: dict[str, set[str]]) -> Result:
    from omegaconf import OmegaConf

    try:
        cfg = compose_cfg(config_name, overrides)
        problems = cfg_problems(cfg, names)
        train_preset = OmegaConf.select(cfg, "train_preset")
        if train_preset:
            tcfg = compose_cfg("train", [f"experiment/train={train_preset}"])
            problems += [f"train_preset {train_preset} : {p}" for p in cfg_problems(tcfg, names)]
        return fail("; ".join(problems)) if problems else ok()
    except (Exception, SystemExit) as exc:
        return fail(short(exc))


def check_presets(names: dict[str, set[str]]) -> dict[str, Result]:
    out = {}
    for config_name, group in (("train", "experiment/train"), ("eval", "experiment/eval")):
        for path in sorted((CONF / group).glob("*.yaml")):
            out[f"{group}={path.stem}"] = check_composition(config_name, [f"{group}={path.stem}"], names)
    return out


def module_from_script(script: str) -> str:
    rel = script.replace("$PROJECT_DIR$/", "")
    root = ROOT.as_posix() + "/"
    rel = rel[len(root):] if rel.startswith(root) else rel
    return rel[: -len(".py")].replace("/", ".") if rel.endswith(".py") else rel


def clean_overrides(params: list[str]) -> list[str]:
    return [p for p in params if p and not p.startswith(("hydra.", "hydra/", "--"))]


def collected_commands() -> dict[tuple[str, str, tuple[str, ...]], str]:
    commands: dict[tuple[str, str, tuple[str, ...]], str] = {}
    configs = EXTRACTS / "run_configs.json"
    if configs.exists():
        for rc in json.loads(configs.read_text(encoding="utf-8")):
            module = rc["target"] if rc["module_mode"] else module_from_script(rc["target"])
            config_name = HYDRA_SCRIPTS.get(module, ("", "", ""))[0]
            params = clean_overrides(shlex.split(rc["parameters"]))
            commands.setdefault((module, config_name, tuple(params)), "pycharm")
    runs = EXTRACTS / "hydra_runs.json"
    if runs.exists():
        for run in json.loads(runs.read_text(encoding="utf-8")):
            module = JOB_TO_MODULE.get(run["job_name"], run["job_name"])
            config_name = run["config_name"] or HYDRA_SCRIPTS.get(module, ("", "", ""))[0]
            commands.setdefault((module, config_name, tuple(clean_overrides(run["overrides"]))), "hydra")
    return commands


def check_commands(names: dict[str, set[str]], imports: dict[str, Result]) -> dict[str, Result]:
    out = {}
    for (module, config_name, overrides), source in sorted(collected_commands().items()):
        key = f"{module.rsplit('.', 1)[-1]} [{config_name or '-'}] {' '.join(overrides)}".strip()
        if module not in HYDRA_SCRIPTS:
            status = imports.get(module, fail("module introuvable dans src/"))
            detail = f"script sans Hydra ({source}) : seul l'import est vérifié. {status['detail']}".strip()
            out[key] = {"ok": status["ok"], "detail": detail}
            continue
        out[key] = check_composition(config_name, list(overrides), names)
    return out


def notebook_imports() -> tuple[dict[str, set[tuple[str, str]]], int]:
    found: dict[str, set[tuple[str, str]]] = {}
    unparsed = 0
    root = EXTRACTS / "notebooks"
    for path in sorted(root.rglob("*.py")):
        notebook = path.relative_to(root).as_posix()[: -len(".py")]
        for chunk in CELL_MARKER.split(path.read_text(encoding="utf-8")):
            try:
                tree = ast.parse(chunk)
            except SyntaxError:
                unparsed += 1
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module and (
                    node.module == "src" or node.module.startswith("src.")
                ):
                    for alias in node.names:
                        found.setdefault(notebook, set()).add((node.module, alias.name))
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == "src" or alias.name.startswith("src."):
                            found.setdefault(notebook, set()).add((alias.name, ""))
    return found, unparsed


def check_import_target(module: str, name: str) -> Result:
    status = try_import(module)
    if not status["ok"] or module in NEVER_IMPORT or not name or name == "*":
        return status
    if hasattr(sys.modules[module], name):
        return ok()
    return ok() if try_import(f"{module}.{name}")["ok"] else fail(f"nom absent du module : {name}")


def check_notebooks() -> tuple[dict[str, Result], int]:
    found, unparsed = notebook_imports()
    out = {}
    for notebook, pairs in sorted(found.items()):
        for module, name in sorted(pairs):
            key = f"{notebook} :: {module}" + (f" import {name}" if name else "")
            out[key] = check_import_target(module, name)
    return out, unparsed


def check_cli() -> dict[str, Result]:
    out = {}
    for module, (_, group, preset) in HYDRA_SCRIPTS.items():
        cmd = [sys.executable, "-m", module, f"{group}={preset}", "--cfg", "job"]
        key = " ".join(["python", *cmd[1:]])
        try:
            proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=900, env=os.environ.copy())
        except subprocess.TimeoutExpired:
            out[key] = fail("délai dépassé (900 s)")
            continue
        if proc.returncode == 0:
            out[key] = ok()
        else:
            lines = [line for line in proc.stderr.strip().splitlines() if line.strip()] or ["(stderr vide)"]
            out[key] = fail(lines[-1][:300])
    return out


def summarize(results: dict[str, dict[str, Result]]) -> None:
    for category, checks in results.items():
        failures = {k: v for k, v in checks.items() if not v["ok"]}
        print(f"\n== {category} : {len(checks) - len(failures)} / {len(checks)} OK")
        for key, value in failures.items():
            print(f"  ÉCHEC {key} : {value['detail']}")


def compare(results: dict[str, dict[str, Result]], baseline_path: Path) -> int:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))["results"]
    regressions, fixed = [], []
    for category, checks in results.items():
        before = baseline.get(category, {})
        for key, value in checks.items():
            prev = before.get(key)
            if not value["ok"] and (prev is None or prev["ok"]):
                regressions.append(f"{category} | {key} : {value['detail']}")
            elif value["ok"] and prev is not None and not prev["ok"]:
                fixed.append(f"{category} | {key}")
    disappeared = [
        f"{category} | {key}"
        for category, checks in baseline.items()
        if category in results
        for key in checks
        if key not in results[category]
    ]
    print(f"\n== Comparaison avec {baseline_path}")
    for title, lines in (("Régressions", regressions), ("Réparés", fixed), ("Contrôles disparus", disappeared)):
        print(f"{title} : {len(lines)}")
        for line in lines:
            print(f"  {line}")
    return 1 if regressions else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--save", type=Path, help="enregistre le résultat (JSON)")
    parser.add_argument("--compare", type=Path, help="compare à un résultat enregistré")
    parser.add_argument("--skip-cli", action="store_true", help="ne lance pas les sous-processus --cfg job")
    args = parser.parse_args()
    if not (ROOT / ".git").exists() or not SRC.is_dir():
        print("À lancer depuis la racine du repo.", file=sys.stderr)
        return 2

    for var, value in (("HF_HUB_OFFLINE", "1"), ("TRANSFORMERS_OFFLINE", "1"), ("HYDRA_FULL_ERROR", "1")):
        os.environ.setdefault(var, value)
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    sys.argv = sys.argv[:1]  # des modules du projet pourraient lire sys.argv à l'import

    names = factory_names()
    results: dict[str, dict[str, Result]] = {"imports": check_imports()}
    results["presets"] = check_presets(names)
    results["commandes"] = check_commands(names, results["imports"])
    results["notebooks"], unparsed = check_notebooks()
    if not args.skip_cli:
        results["cli"] = check_cli()

    summarize(results)
    print(f"\nCellules de notebooks non analysables (ast) : {unparsed}")
    if args.save:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True)
        meta = {
            "date": datetime.now().isoformat(timespec="seconds"),
            "git_head": head.stdout.strip(),
            "notebook_cells_unparsed": unparsed,
        }
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(
            json.dumps({"meta": meta, "results": results}, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        print(f"Résultat enregistré dans {args.save}")
    return compare(results, args.compare) if args.compare else 0


if __name__ == "__main__":
    sys.exit(main())
