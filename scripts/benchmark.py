"""Mesures de performance (PARTIE VI).

    python scripts/benchmark.py

1. latence d'inférence du modèle seul (une demande déjà transformée) ;
2. latence du pipeline complet dans le processus (validation + prétraitement + modèle) ;
3. débit d'inférence par lots ;
4. temps de réponse de l'API à travers un vrai serveur uvicorn (HTTP local, séquentiel) ;
5. charge progressive avec Locust (1 à 40 utilisateurs simultanés, sans pause).

Résultats : reports/performance.json. Les objectifs sont lus dans app/config.py.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import config  # noqa: E402
from app.prediction import ServicePrediction  # noqa: E402
from app.preprocessing import preparer_variables  # noqa: E402
from app.validation import DemandeCredit  # noqa: E402

PORT = 8011  # le port 8000 est déjà utilisé sur ce poste
URL = f"http://127.0.0.1:{PORT}"
REPETITIONS = 2000
ECHAUFFEMENT = 100
REQUETES_HTTP = 1000
DEMANDE = {
    "age": 35,
    "revenu_mensuel": 250000,
    "montant_credit": 1000000,
    "duree_mois": 12,
    "anciennete_activite": 4,
}


def resume(latences_ms) -> dict:
    x = np.asarray(latences_ms, dtype=float)
    return {
        "n": int(len(x)),
        "moyenne_ms": round(float(x.mean()), 3),
        "p50_ms": round(float(np.percentile(x, 50)), 3),
        "p95_ms": round(float(np.percentile(x, 95)), 3),
        "p99_ms": round(float(np.percentile(x, 99)), 3),
        "max_ms": round(float(x.max()), 3),
    }


def mesurer(fonction, n: int = REPETITIONS) -> dict:
    for _ in range(ECHAUFFEMENT):
        fonction()
    latences = []
    for _ in range(n):
        debut = time.perf_counter()
        fonction()
        latences.append((time.perf_counter() - debut) * 1000)
    return resume(latences)


def environnement() -> dict:
    processeur = platform.processor()
    if sys.platform == "win32":
        try:
            sortie = subprocess.run(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            processeur = sortie.stdout.strip() or processeur
        except (OSError, subprocess.SubprocessError):
            pass
    libs = ["numpy", "pandas", "scikit-learn", "fastapi", "pydantic", "uvicorn", "locust"]
    return {
        "date": datetime.now().isoformat(timespec="seconds"),
        "systeme": platform.platform(),
        "processeur": processeur,
        "coeurs_physiques": psutil.cpu_count(logical=False),
        "coeurs_logiques": psutil.cpu_count(logical=True),
        "memoire_totale_go": round(psutil.virtual_memory().total / 1e9, 1),
        "python": platform.python_version(),
        "bibliotheques": {lib: version(lib) for lib in libs},
        "serveur": "uvicorn, 1 processus (worker), boucle locale 127.0.0.1",
    }


def demarrer_serveur() -> subprocess.Popen:
    serveur = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app.main:app",
            "--port",
            str(PORT),
            "--log-level",
            "warning",
        ],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(100):
        try:
            if httpx.get(f"{URL}/health", timeout=1).status_code == 200:
                return serveur
        except httpx.HTTPError:
            pass
        time.sleep(0.3)
    serveur.kill()
    raise RuntimeError("le serveur uvicorn n'a pas démarré")


def processus_serveur(pid: int) -> list[psutil.Process]:
    """Le processus lancé et ses enfants.

    Sous Windows, le python.exe d'un venv est un lanceur qui démarre le vrai
    interpréteur dans un processus enfant : mesurer le seul PID lancé donnait
    5 Mo et 0 % de CPU (première mesure, corrigée).
    """
    parent = psutil.Process(pid)
    return [parent, *parent.children(recursive=True)]


def memoire_mo(pid: int) -> float:
    return sum(p.memory_info().rss for p in processus_serveur(pid)) / 1e6


class SuiviMemoire(threading.Thread):
    """Relève la mémoire (RSS) et le CPU du serveur toutes les 0,5 s."""

    def __init__(self, pid: int):
        super().__init__(daemon=True)
        self.pid = pid
        self.processus = processus_serveur(pid)
        self.maximum = 0.0
        self.cpu = []
        self.actif = True

    def run(self):
        for p in self.processus:
            p.cpu_percent(None)
        while self.actif:
            self.maximum = max(self.maximum, memoire_mo(self.pid))
            time.sleep(0.5)
            self.cpu.append(sum(p.cpu_percent(None) for p in self.processus))


def mesures_http_sequentielles() -> dict:
    with httpx.Client(base_url=URL, timeout=10) as client:
        for _ in range(50):
            client.post("/predict", json=DEMANDE)
        latences, erreurs = [], 0
        debut_total = time.perf_counter()
        for _ in range(REQUETES_HTTP):
            debut = time.perf_counter()
            r = client.post("/predict", json=DEMANDE)
            latences.append((time.perf_counter() - debut) * 1000)
            erreurs += r.status_code != 200
        duree = time.perf_counter() - debut_total
    return {**resume(latences), "erreurs": erreurs, "debit_req_s": round(REQUETES_HTTP / duree, 1)}


def charge_progressive() -> list[dict]:
    fichier = config.REPORTS_DIR / "perf" / "locust_requetes.csv"
    fichier.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "locust",
            "-f",
            "scripts/locustfile.py",
            "--headless",
            "--host",
            URL,
            "--only-summary",
            "--loglevel",
            "WARNING",
        ],
        cwd=ROOT,
        env={**os.environ, "CHARGE_CSV": str(fichier)},
        check=True,
        stdout=subprocess.DEVNULL,
    )
    with open(fichier, encoding="utf-8") as f:
        lignes = list(csv.DictReader(f))
    df = pd.DataFrame(lignes).astype(
        {"instant": float, "utilisateurs": int, "latence_ms": float, "statut": int}
    )
    df["succes"] = df["succes"] == "True"

    paliers = []
    for utilisateurs, groupe in df.groupby("utilisateurs"):
        # on écarte la première seconde de chaque palier (démarrage des utilisateurs)
        stable = groupe[groupe["instant"] >= groupe["instant"].min() + 1]
        duree = stable["instant"].max() - stable["instant"].min()
        paliers.append(
            {
                "utilisateurs": int(utilisateurs),
                **resume(stable["latence_ms"]),
                "debit_req_s": round(len(stable) / duree, 1) if duree > 0 else None,
                "taux_erreur": round(
                    float(1 - (stable["succes"] & (stable["statut"] == 200)).mean()), 4
                ),
            }
        )
    return paliers


def main() -> None:
    resultats = {"environnement": environnement(), "objectifs": config.OBJECTIFS_PERFORMANCE}

    service = ServicePrediction()
    service.charger()
    variables = preparer_variables(pd.DataFrame([DEMANDE]))
    resultats["inference_modele"] = mesurer(lambda: service.predire_proba(variables))
    resultats["pipeline_dans_le_processus"] = mesurer(
        lambda: service.evaluer_demande(DemandeCredit(**DEMANDE))
    )

    lot = preparer_variables(
        pd.read_csv(config.DATA_PATH).sample(5000, random_state=1, replace=False)
    )
    durees = []
    for _ in range(20):
        debut = time.perf_counter()
        service.predire_proba(lot)
        durees.append(time.perf_counter() - debut)
    resultats["inference_par_lot"] = {
        "taille_lot": len(lot),
        "duree_mediane_ms": round(statistics.median(durees) * 1000, 2),
        "debit_lignes_s": round(len(lot) / statistics.median(durees)),
    }
    resultats["taille_modele_ko"] = round(config.MODEL_PATH.stat().st_size / 1024, 1)
    print("modèle et pipeline mesurés")

    serveur = demarrer_serveur()
    try:
        resultats["memoire_serveur_repos_mo"] = round(memoire_mo(serveur.pid), 1)
        resultats["api_http_sequentiel"] = mesures_http_sequentielles()
        print("API séquentielle mesurée, charge progressive en cours (environ 75 s)")
        suivi = SuiviMemoire(serveur.pid)
        suivi.start()
        resultats["charge_progressive"] = charge_progressive()
        suivi.actif = False
        suivi.join()
        resultats["memoire_serveur_max_mo"] = round(suivi.maximum, 1)
        resultats["cpu_serveur_max_pct"] = round(max(suivi.cpu, default=0), 1)
    finally:
        serveur.terminate()
        serveur.wait(timeout=10)

    o = config.OBJECTIFS_PERFORMANCE
    palier_cible = next(
        p for p in resultats["charge_progressive"] if p["utilisateurs"] == o["utilisateurs_cible"]
    )
    # bool() : les comparaisons sur des valeurs NumPy donnent des numpy.bool_, refusés par json
    resultats["verdict"] = {
        "inference_p95": bool(resultats["inference_modele"]["p95_ms"] <= o["inference_p95_ms"]),
        "api_p95_sous_charge_cible": bool(palier_cible["p95_ms"] <= o["api_p95_ms"]),
        "debit_sous_charge_cible": bool(palier_cible["debit_req_s"] >= o["debit_min_req_s"]),
        "aucune_erreur": bool(
            all(p["taux_erreur"] <= o["taux_erreur_max"] for p in resultats["charge_progressive"])
        ),
    }

    (config.REPORTS_DIR / "performance.json").write_text(
        json.dumps(resultats, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in resultats.items() if k != "environnement"}, indent=1))


if __name__ == "__main__":
    main()
