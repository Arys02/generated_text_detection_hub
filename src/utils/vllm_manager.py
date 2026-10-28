# src/utils/vllm_manager.py
"""Pilotage du serveur vLLM local depuis Python (mêmes sessions tmux que vllm-start / vllm-stop) :
arrêt avec attente de libération du GPU, démarrage d'un profil, attente de disponibilité réelle
(health + modèle listé + requête de test), réutilisation si le bon modèle est déjà prêt.
"""
from __future__ import annotations

import logging
import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import requests

logger = logging.getLogger(__name__)

# dossier de l'environnement vLLM (launch_server.sh, profiles/, logs/) : VLLM_ENV_DIR, sinon ~/vllm-env
VLLM_ENV = Path(os.getenv("VLLM_ENV_DIR") or Path.home() / "vllm-env")
SESSION = "vllm"
WATCHDOG_SESSION = "watchdog"


@dataclass
class VLLMManager:
    gpu: int = 1
    host: str = "127.0.0.1"
    port: int = 8000
    env_dir: Path = VLLM_ENV
    log_dir: Optional[Path] = None
    start_timeout: float = 1800.0   # premier chargement + compilation : peut être long
    stop_timeout: float = 180.0
    poll_every: float = 5.0
    gpu_free_mib: int = 1500        # mémoire résiduelle tolérée sur le GPU après arrêt
    restart_watchdog: bool = False

    def __post_init__(self):
        self.log_dir = Path(self.log_dir) if self.log_dir else self.env_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._current_log: Optional[Path] = None

    # ----------------------------------------------------------------- état
    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def profile_served_name(self, profile: str) -> str:
        path = self.env_dir / "profiles" / f"{profile}.env"
        if not path.exists():
            raise FileNotFoundError(f"profil introuvable : {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("SERVED_NAME="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
        raise ValueError(f"SERVED_NAME absent de {path}")

    def _tmux_has(self, session: str) -> bool:
        return subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode == 0

    def healthy(self) -> bool:
        try:
            return requests.get(f"{self.base_url}/health", timeout=5).status_code == 200
        except requests.RequestException:
            return False

    def served_models(self) -> List[str]:
        try:
            r = requests.get(f"{self.base_url}/v1/models", timeout=5)
            r.raise_for_status()
            return [m["id"] for m in r.json().get("data", [])]
        except requests.RequestException:
            return []

    def answers(self, served: str) -> bool:
        """Requête minimale : le modèle répond réellement, pas seulement le processus."""
        try:
            r = requests.post(f"{self.base_url}/v1/chat/completions", timeout=120, json={
                "model": served, "max_tokens": 1,
                "messages": [{"role": "user", "content": "ping"}],
            })
            return r.status_code == 200
        except requests.RequestException:
            return False

    def gpu_used_mib(self) -> int:
        # index nvidia-smi ; identique à l'index CUDA pour deux cartes identiques (ordre PCI)
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", str(self.gpu)],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return int(out.splitlines()[0])

    def log_tail(self, n: int = 60) -> str:
        if self._current_log is None or not self._current_log.exists():
            return "(pas de journal)"
        return "\n".join(self._current_log.read_text(encoding="utf-8", errors="replace").splitlines()[-n:])

    # ---------------------------------------------------------- transitions
    def stop(self) -> None:
        for session in (WATCHDOG_SESSION, SESSION):  # watchdog d'abord : il ne doit pas relancer vLLM
            if self._tmux_has(session):
                subprocess.run(["tmux", "kill-session", "-t", session], capture_output=True)
        t0 = time.monotonic()
        while time.monotonic() - t0 < self.stop_timeout:
            used = self.gpu_used_mib()
            if not self.healthy() and used <= self.gpu_free_mib:
                logger.info(f"vLLM arrêté, GPU {self.gpu} libéré ({used} MiB utilisés)")
                return
            time.sleep(self.poll_every)
        raise RuntimeError(f"GPU {self.gpu} non libéré après {self.stop_timeout:.0f} s "
                           f"({self.gpu_used_mib()} MiB utilisés) : processus résiduel ?")

    def start(self, profile: str) -> float:
        served = self.profile_served_name(profile)
        if self._tmux_has(SESSION):
            raise RuntimeError("une session tmux 'vllm' existe déjà : appeler stop() d'abord")

        self._current_log = self.log_dir / f"{profile}_{datetime.now():%Y%m%d_%H%M%S}.log"
        launch = self.env_dir / "launch_server.sh"
        cmd = (f"CUDA_VISIBLE_DEVICES={self.gpu} {shlex.quote(str(launch))} {shlex.quote(profile)} "
               f"2>&1 | tee -a {shlex.quote(str(self._current_log))}")
        subprocess.run(["tmux", "new", "-d", "-s", SESSION, cmd], check=True)
        logger.info(f"démarrage de {profile} ({served}) sur GPU {self.gpu}, journal : {self._current_log}")

        elapsed = self.wait_ready(served)
        if self.restart_watchdog:
            activate = self.env_dir / "bin" / "activate"
            watchdog = self.env_dir / "watchdog.py"
            subprocess.run(["tmux", "new", "-d", "-s", WATCHDOG_SESSION,
                            f"source {shlex.quote(str(activate))} && python {shlex.quote(str(watchdog))}"], check=True)
        return elapsed

    def wait_ready(self, served: str) -> float:
        t0 = time.monotonic()
        while time.monotonic() - t0 < self.start_timeout:
            if not self._tmux_has(SESSION):
                raise RuntimeError(f"vLLM s'est arrêté pendant le démarrage. Fin du journal :\n{self.log_tail()}")
            if self.healthy() and served in self.served_models() and self.answers(served):
                elapsed = time.monotonic() - t0
                logger.info(f"{served} prêt en {elapsed:.0f} s")
                return elapsed
            time.sleep(self.poll_every)
        raise TimeoutError(f"{served} non prêt après {self.start_timeout:.0f} s. Fin du journal :\n{self.log_tail()}")

    def ensure(self, profile: str) -> None:
        """Garantit que `profile` est servi et opérationnel ; réutilise le serveur s'il l'est déjà."""
        served = self.profile_served_name(profile)
        if self.healthy() and self.served_models() == [served] and self.answers(served):
            logger.info(f"{served} déjà servi et opérationnel : réutilisé")
            return
        self.stop()
        self.start(profile)