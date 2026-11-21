"""Génère le jeu de données SYNTHÉTIQUE data/credit_synthetique.csv.

Ces données servent à tester le système (validation, API, pipeline, non-régression).
Elles ne permettent PAS de mesurer la performance réelle d'un modèle de crédit.

Règle de génération du défaut (logit), choisie pour être plausible et documentée :
  - le risque augmente avec le taux d'effort (mensualité / revenu) et la durée ;
  - il diminue avec l'ancienneté de l'activité, avec un surcroît de risque
    pour les activités de moins d'un an ;
  - il est plus élevé aux deux extrémités de l'âge (moins de 25 ans, plus de 60 ans).
Les deux derniers effets ne sont pas linéaires : un modèle linéaire ne peut pas
les reproduire exactement, ce qui crée volontairement des profils mal détectés.

    python scripts/generate_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import config  # noqa: E402

N_DOSSIERS = 6000
DUREES = [3, 6, 9, 12, 18, 24, 36, 48, 60]
POIDS_DUREES = [0.06, 0.16, 0.10, 0.24, 0.14, 0.15, 0.09, 0.04, 0.02]


def generer(n: int = N_DOSSIERS, seed: int = config.SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = np.clip(np.round(rng.normal(40, 12, n)), 18, 75).astype(int)
    # 15 % d'activités récentes (moins d'un an), le reste selon une loi gamma
    anciennete = np.where(rng.random(n) < 0.15, rng.uniform(0, 1, n), rng.gamma(2.0, 3.0, n))
    anciennete = np.minimum(anciennete, age - config.AGE_DEBUT_ACTIVITE)
    anciennete = np.round(np.clip(anciennete, 0, None), 1)

    revenu = np.exp(np.log(150_000) + 0.03 * anciennete + rng.normal(0, 0.55, n))
    revenu = np.round(np.clip(revenu, 30_000, 5_000_000), -3)

    duree = rng.choice(DUREES, size=n, p=POIDS_DUREES)
    effort = np.clip(rng.lognormal(np.log(0.22), 0.45, n), 0.03, 1.2)
    montant = np.round(np.clip(revenu * duree * effort, 50_000, config.MONTANT_MAX), -4)
    taux_effort = montant / duree / revenu

    # Réglage (sans aucun modèle) : taux de défaut d'environ 10 % et AUC de la vraie
    # probabilité proche de 0,80, ordre de grandeur d'un score de crédit utilisable.
    # Un premier réglage donnait une AUC maximale de 0,68 : signal trop faible pour
    # que le critère d'acceptation (AUC >= 0,75) soit atteignable (anomalie AN-01).
    logit = (
        -2.2
        + 6.5 * (taux_effort - 0.25)
        - 0.20 * np.minimum(anciennete, 10)
        + 0.80 * (anciennete < 1)
        + 0.02 * (duree - 12)
        + 1.00 * (age < 25)
        + 0.90 * (age > 60)
    )
    probabilite = 1 / (1 + np.exp(-logit))
    defaut = rng.binomial(1, probabilite)

    df = pd.DataFrame(
        {
            "age": age,
            "revenu_mensuel": revenu,
            "montant_credit": montant,
            "duree_mois": duree,
            "anciennete_activite": anciennete,
            config.TARGET: defaut,
        }
    )
    return df, probabilite


def main() -> None:
    df, probabilite = generer()
    config.DATA_PATH.parent.mkdir(exist_ok=True)
    # fins de ligne LF imposées : sous Windows, pandas écrirait CRLF et l'empreinte
    # SHA-256 du fichier ne serait plus la même après passage par Git (Linux en CI)
    df.to_csv(config.DATA_PATH, index=False, lineterminator="\n")
    print(f"{len(df)} dossiers écrits dans {config.DATA_PATH.name}")
    print(f"taux de défaut : {df[config.TARGET].mean():.1%}")
    # AUC obtenue avec la vraie probabilité : plafond qu'aucun modèle ne peut dépasser
    print(f"AUC maximale atteignable : {roc_auc_score(df[config.TARGET], probabilite):.3f}")
    print(df.describe().round(2).to_string())


if __name__ == "__main__":
    main()
