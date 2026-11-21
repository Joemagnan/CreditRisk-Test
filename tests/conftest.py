import pytest
from fastapi.testclient import TestClient

from app import config
from app.main import app, service


@pytest.fixture(scope="session")
def client():
    # Le bloc "with" déclenche le démarrage de l'API, donc le chargement du modèle
    with TestClient(app) as c:
        yield c


@pytest.fixture
def demande_valide():
    """Exemple de l'énoncé."""
    return {
        "age": 35,
        "revenu_mensuel": 250000,
        "montant_credit": 1000000,
        "duree_mois": 12,
        "anciennete_activite": 4,
    }


@pytest.fixture
def appels_modele(monkeypatch):
    """Compte les appels au calcul du score (une entrée rejetée ne doit jamais y arriver)."""
    compteur = {"n": 0}
    original = service.evaluer_demande

    def espion(demande):
        compteur["n"] += 1
        return original(demande)

    monkeypatch.setattr(service, "evaluer_demande", espion)
    return compteur


@pytest.fixture
def remettre_modele():
    """Pour les tests qui cassent volontairement le service : on recharge le vrai modèle après."""
    yield
    service.model_path = config.MODEL_PATH
    service.metadata_path = config.METADATA_PATH
    service.charger()
