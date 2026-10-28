# file: src/detector/subjectivity_aware_generation_estimator/xai/results_store.py
from __future__ import annotations

import json
import platform
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import RESULT_DIR
from src.detector.subjectivity_aware_generation_estimator.xai.experiments import EXPERIMENTS

STORE_DIR = Path(RESULT_DIR) / "xai" / "store"
MEASUREMENTS = STORE_DIR / "measurements.jsonl"
RUNS = STORE_DIR / "runs.jsonl"

# Protocole : l'appariement (pair_and_balance) a réduit le train d'environ 40 %.
# Les mesures antérieures ne sont PAS comparables aux mesures appariées.
PROTO_PRE_PAIRING = "pre_pairing"
PROTO_PAIRED = "paired"


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def _append(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")


def record_run(experiment: str, protocol: str, spec: str | None = None,
               spec_hash: str | None = None, model: str = "SimpleGRU",
               notes: str = "", **context) -> str:
    """Enregistre une exécution et retourne son run_id, à passer à record_metrics."""
    if experiment not in EXPERIMENTS:
        raise KeyError(f"expérience inconnue : {experiment!r} ; "
                       f"déclare-la dans experiments.py avant d'enregistrer des mesures")
    if protocol not in (PROTO_PRE_PAIRING, PROTO_PAIRED):
        raise ValueError(f"protocole inconnu : {protocol!r}")

    exp = EXPERIMENTS[experiment]
    run_id = uuid.uuid4().hex[:12]
    _append(RUNS, [{
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "experiment": experiment, "axis": exp.axis.value, "method": exp.method.value,
        "protocol": protocol, "spec": spec, "spec_hash": spec_hash, "model": model,
        "sources": list(exp.sources), "notes": notes,
        "git": _git_commit(), "python": platform.python_version(),
        **context,
    }])
    return run_id


def record_metrics(run_id: str, df: pd.DataFrame, condition_col: str,
                   metric_cols: list[str], replicate_col: str | None = None) -> int:
    """Convertit un DataFrame large en lignes longues (une par métrique) et les append.

    df doit contenir une colonne de condition, éventuellement une colonne de réplicat
    (seed ou numéro de permutation), et une colonne par métrique.
    """
    missing = [c for c in [condition_col, *metric_cols] if c not in df.columns]
    if missing:
        raise KeyError(f"colonnes absentes du DataFrame : {missing}")

    records = []
    for _, row in df.iterrows():
        base = {"run_id": run_id, "condition": str(row[condition_col]),
                "replicate": (int(row[replicate_col]) if replicate_col else None)}
        for m in metric_cols:
            val = row[m]
            if pd.isna(val):
                continue
            records.append({**base, "metric": m, "value": float(val)})
    _append(MEASUREMENTS, records)
    return len(records)


def load() -> pd.DataFrame:
    """Mesures jointes à leur provenance. Une ligne = (run, condition, réplicat, métrique)."""
    if not MEASUREMENTS.exists():
        return pd.DataFrame()
    m = pd.read_json(MEASUREMENTS, lines=True)
    r = pd.read_json(RUNS, lines=True)
    return m.merge(r, on="run_id", how="left", validate="many_to_one")


def summary(protocol: str | None = None) -> pd.DataFrame:
    """Agrégé par axe / expérience / condition / métrique, avec moyenne et écart-type."""
    d = load()
    if d.empty:
        return d
    if protocol is not None:
        d = d[d["protocol"] == protocol]
    out = (d.groupby(["axis", "experiment", "condition", "metric"])["value"]
             .agg(moyenne="mean", ecart_type="std", n="size")
             .reset_index())
    return out.sort_values(["axis", "experiment", "metric", "condition"])