"""Module de prédiction : chargement contrôlé du modèle versionné et calcul du score."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from app import config
from app.preprocessing import demande_vers_dataframe, preparer_variables
from app.validation import detecter_alertes


class ModeleIndisponible(RuntimeError):
    """Le modèle n'est pas chargé ou n'a pas pu l'être."""


class ErreurPrediction(RuntimeError):
    """Le modèle a été appelé mais son résultat n'est pas exploitable."""


def sha256_fichier(chemin: Path) -> str:
    empreinte = hashlib.sha256()
    with open(chemin, "rb") as fichier:
        for bloc in iter(lambda: fichier.read(1 << 16), b""):
            empreinte.update(bloc)
    return empreinte.hexdigest()


class ServicePrediction:
    def __init__(
        self, model_path: Path = config.MODEL_PATH, metadata_path: Path = config.METADATA_PATH
    ):
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)
        self.modele = None
        self.metadata: dict | None = None

    @property
    def pret(self) -> bool:
        return self.modele is not None and self.metadata is not None

    def charger(self) -> None:
        """Charge le modèle après avoir vérifié son empreinte SHA-256.

        joblib (pickle) peut exécuter du code au chargement : on ne charge donc
        qu'un fichier identique à celui enregistré dans les métadonnées.
        """
        self.modele = None
        self.metadata = None
        try:
            metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ModeleIndisponible("métadonnées du modèle absentes ou illisibles") from exc

        if not self.model_path.is_file():
            raise ModeleIndisponible("fichier du modèle absent")
        if sha256_fichier(self.model_path) != metadata.get("sha256"):
            raise ModeleIndisponible("empreinte SHA-256 différente : fichier modifié ou corrompu")

        try:
            modele = joblib.load(self.model_path)
        except Exception as exc:
            raise ModeleIndisponible("fichier du modèle illisible") from exc

        if not hasattr(modele, "predict_proba"):
            raise ModeleIndisponible("l'objet chargé n'est pas un classifieur probabiliste")
        variables = list(getattr(modele, "feature_names_in_", []))
        if variables != metadata.get("variables_modele"):
            raise ModeleIndisponible("variables du modèle différentes des métadonnées")

        self.modele = modele
        self.metadata = metadata

    def predire_proba(self, variables: pd.DataFrame) -> np.ndarray:
        """Probabilité de défaut pour chaque ligne, avec contrôle du contrat de sortie."""
        if not self.pret:
            raise ModeleIndisponible("modèle non chargé")

        attendues = self.metadata["variables_modele"]
        if list(variables.columns) != attendues:
            raise ErreurPrediction(
                f"dimensions incompatibles : {variables.shape[1]} variables reçues, "
                f"{len(attendues)} attendues par le modèle"
            )

        sortie = np.asarray(self.modele.predict_proba(variables))
        if sortie.shape != (len(variables), 2):
            raise ErreurPrediction(f"sortie du modèle de forme inattendue {sortie.shape}")
        probas = sortie[:, 1]
        if not np.isfinite(probas).all() or (probas < 0).any() or (probas > 1).any():
            raise ErreurPrediction("probabilité hors de l'intervalle [0, 1]")
        return probas

    def evaluer_demande(self, demande) -> dict:
        """Chaîne complète pour une demande validée : variables -> score -> réponse."""
        if not self.pret:
            raise ModeleIndisponible("modèle non chargé")
        variables = preparer_variables(demande_vers_dataframe(demande))
        probabilite = float(self.predire_proba(variables)[0])
        seuil = self.metadata["seuil"]
        return {
            "probabilite_defaut": round(probabilite, 6),
            "niveau_risque": "eleve" if probabilite >= seuil else "faible",
            "seuil_decision": seuil,
            "version_modele": self.metadata["version"],
            "alertes": detecter_alertes(demande, self.metadata["plages_entrainement"]),
        }
