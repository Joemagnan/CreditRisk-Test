"""Module de prétraitement : passage des données du demandeur aux variables du modèle.

La même fonction sert à l'entraînement et à la prédiction, pour éviter tout
décalage entre les deux (une cause classique d'erreurs silencieuses).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.config import FEATURES

# Variables effectivement vues par le modèle, dans cet ordre
VARIABLES_MODELE = [
    "age",
    "anciennete_activite",
    "duree_mois",
    "log_revenu",
    "log_montant",
    "taux_effort",
]


class ErreurPretraitement(ValueError):
    """Les données ne permettent pas de construire les variables du modèle."""


def demande_vers_dataframe(demande) -> pd.DataFrame:
    """Une demande validée (DemandeCredit) devient un DataFrame d'une ligne."""
    valeurs = demande.model_dump()
    return pd.DataFrame([{nom: valeurs[nom] for nom in FEATURES}], columns=FEATURES)


def preparer_variables(donnees: pd.DataFrame) -> pd.DataFrame:
    """Construit les variables du modèle à partir des cinq variables du demandeur.

    - log du revenu et du montant : ces montants sont très asymétriques ;
    - taux d'effort = mensualité estimée / revenu, avec mensualité = montant / durée
      (simplification : les intérêts sont ignorés).
    """
    manquantes = [nom for nom in FEATURES if nom not in donnees.columns]
    if manquantes:
        raise ErreurPretraitement(f"variables absentes : {', '.join(manquantes)}")
    if donnees.empty:
        raise ErreurPretraitement("aucune ligne à traiter")

    try:
        x = donnees[FEATURES].astype(float)
    except (TypeError, ValueError) as exc:
        raise ErreurPretraitement("variables non numériques") from exc

    if not np.isfinite(x.to_numpy()).all():
        raise ErreurPretraitement("valeurs manquantes ou non finies")
    if (x[["revenu_mensuel", "montant_credit", "duree_mois"]] <= 0).to_numpy().any():
        raise ErreurPretraitement("revenu, montant et durée doivent être strictement positifs")

    variables = pd.DataFrame(
        {
            "age": x["age"],
            "anciennete_activite": x["anciennete_activite"],
            "duree_mois": x["duree_mois"],
            "log_revenu": np.log(x["revenu_mensuel"]),
            "log_montant": np.log(x["montant_credit"]),
            "taux_effort": x["montant_credit"] / x["duree_mois"] / x["revenu_mensuel"],
        },
        columns=VARIABLES_MODELE,
    )
    return variables
