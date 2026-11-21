"""API de prédiction du risque de crédit.

    GET  /health      disponibilité du service (200, ou 503 si le modèle n'est pas chargé)
    POST /predict     probabilité de défaut d'une demande
    GET  /model-info  métadonnées de la version du modèle

Codes d'erreur : 400 JSON mal formé, 422 données invalides, 503 modèle indisponible,
500 erreur interne (message générique, jamais de trace technique côté client).
"""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app import config
from app.prediction import ModeleIndisponible, ServicePrediction
from app.validation import DemandeCredit, traduire_erreurs

logger = logging.getLogger("creditrisk")
service = ServicePrediction()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        service.charger()
        logger.info("modèle %s chargé", service.metadata["version"])
    except ModeleIndisponible as exc:
        # L'API démarre quand même : /health le signale, /predict répond 503
        logger.error("modèle non chargé : %s", exc)
    yield


app = FastAPI(
    title="CreditRisk-Test",
    description="Prototype pédagogique d'estimation du risque de défaut (données synthétiques).",
    version=config.MODEL_VERSION,
    lifespan=lifespan,
)


class Alerte(BaseModel):
    code: str
    champ: str
    message: str


class ReponsePrediction(BaseModel):
    probabilite_defaut: float
    niveau_risque: str
    seuil_decision: float
    version_modele: str
    alertes: list[Alerte]


def _erreur(statut: int, code: str, message: str, **extra) -> JSONResponse:
    return JSONResponse(status_code=statut, content={"erreur": code, "message": message, **extra})


def _modele_indisponible() -> JSONResponse:
    return _erreur(503, "modele_indisponible", "Le modèle de scoring n'est pas disponible.")


@app.exception_handler(RequestValidationError)
async def erreur_validation(_request: Request, exc: RequestValidationError) -> JSONResponse:
    erreurs = exc.errors()
    if any(e.get("type") == "json_invalid" for e in erreurs):
        return _erreur(
            400, "requete_mal_formee", "Le corps de la requête n'est pas un JSON valide."
        )
    return _erreur(
        422,
        "donnees_invalides",
        "La demande ne respecte pas le contrat d'entrée.",
        details=traduire_erreurs(erreurs),
    )


@app.get("/health")
def health():
    if not service.pret:
        return JSONResponse(
            status_code=503, content={"statut": "indisponible", "modele_charge": False}
        )
    return {"statut": "ok", "modele_charge": True, "version_modele": service.metadata["version"]}


@app.get("/model-info")
def model_info():
    if not service.pret:
        return _modele_indisponible()
    champs = (
        "version",
        "date_entrainement",
        "algorithme",
        "donnees",
        "variables_entree",
        "variables_modele",
        "seuil",
        "metriques_validation",
        "metriques_test",
        "sha256",
    )
    return {champ: service.metadata[champ] for champ in champs}


@app.post("/predict", response_model=ReponsePrediction)
def predict(demande: DemandeCredit):
    try:
        return service.evaluer_demande(demande)
    except ModeleIndisponible:
        return _modele_indisponible()
    except Exception:
        # La trace complète reste dans les journaux du serveur, identifiée par id_requete.
        # Les valeurs de la demande ne sont pas journalisées (données personnelles).
        id_requete = uuid.uuid4().hex[:12]
        logger.exception("erreur interne pendant la prédiction [id_requete=%s]", id_requete)
        return _erreur(
            500,
            "erreur_interne",
            "La prédiction n'a pas pu être calculée.",
            id_requete=id_requete,
        )
