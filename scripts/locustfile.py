"""Test de charge progressive (Locust), lancé par scripts/benchmark.py.

Chaque utilisateur simulé envoie des demandes en continu, sans pause : c'est
volontairement plus dur que la réalité (un agent saisit un dossier en quelques
minutes). Les paliers montent de 1 à 40 utilisateurs simultanés.

Chaque requête est enregistrée (instant, nombre d'utilisateurs, latence, succès)
dans le fichier indiqué par la variable d'environnement CHARGE_CSV.
"""

import csv
import os
import random
import time

from locust import FastHttpUser, LoadTestShape, events, task

PALIERS = [1, 5, 10, 20, 40]
DUREE_PALIER_S = 15

enregistrements = []


class AgentCredit(FastHttpUser):
    @task
    def evaluer_dossier(self):
        demande = {
            "age": random.randint(22, 60),
            "revenu_mensuel": random.choice([90_000, 150_000, 250_000, 400_000]),
            "montant_credit": random.choice([300_000, 1_000_000, 2_500_000]),
            "duree_mois": random.choice([6, 12, 24]),
            "anciennete_activite": random.choice([0.5, 2, 5]),
        }
        self.client.post("/predict", json=demande)


class MonteeProgressive(LoadTestShape):
    def tick(self):
        palier = int(self.get_run_time() // DUREE_PALIER_S)
        if palier >= len(PALIERS):
            return None
        return PALIERS[palier], PALIERS[palier]


environnement = {}


@events.init.add_listener
def memoriser_environnement(environment, **_kwargs):
    environnement["runner"] = environment.runner


@events.request.add_listener
def enregistrer(response_time, exception, response, **_kwargs):
    runner = environnement.get("runner")
    utilisateurs = runner.user_count if runner else 0
    statut = getattr(response, "status_code", 0)
    enregistrements.append((time.time(), utilisateurs, response_time, exception is None, statut))


@events.quitting.add_listener
def ecrire(**_kwargs):
    chemin = os.environ.get("CHARGE_CSV", "charge.csv")
    with open(chemin, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["instant", "utilisateurs", "latence_ms", "succes", "statut"])
        w.writerows(enregistrements)
