"""Tests d'intégration (PARTIE III) : validation + prétraitement + modèle + API.

Le système doit gérer chaque panne proprement, sans exposer de trace technique au client.
"""

import json
import logging
import math
import shutil

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import config
from app.main import app, service
from app.prediction import ErreurPrediction, ModeleIndisponible, ServicePrediction, sha256_fichier
from app.preprocessing import VARIABLES_MODELE, ErreurPretraitement, preparer_variables
from app.training import sauvegarder
from app.validation import DemandeCredit

# --- Scénario principal, étape par étape ------------------------------------------


def test_scenario_principal_de_bout_en_bout(client, demande_valide):
    # 1. envoi d'une requête valide
    reponse = client.post("/predict", json=demande_valide)
    assert reponse.status_code == 200

    # 2. validation des données
    demande = DemandeCredit(**demande_valide)

    # 3. transformation des variables (valeurs recalculées à la main)
    variables = preparer_variables(pd.DataFrame([demande.model_dump()]))
    assert list(variables.columns) == VARIABLES_MODELE
    assert variables.shape == (1, 6)
    ligne = variables.iloc[0]
    assert ligne["log_revenu"] == pytest.approx(math.log(250_000))
    assert ligne["log_montant"] == pytest.approx(math.log(1_000_000))
    assert ligne["taux_effort"] == pytest.approx(1_000_000 / 12 / 250_000)

    # 4. prédiction : l'API renvoie bien ce que calcule le modèle
    proba = service.predire_proba(variables)[0]
    resultat = reponse.json()
    assert resultat["probabilite_defaut"] == pytest.approx(proba, abs=1e-6)

    # 5. structure de la réponse
    assert isinstance(resultat["probabilite_defaut"], float)
    assert isinstance(resultat["alertes"], list)

    # 6. cohérence des métadonnées : réponse, /model-info, fichier et code
    info = client.get("/model-info").json()
    fichier = json.loads(config.METADATA_PATH.read_text(encoding="utf-8"))
    assert (
        resultat["version_modele"] == info["version"] == fichier["version"] == config.MODEL_VERSION
    )
    assert resultat["seuil_decision"] == info["seuil"] == fichier["seuil"]
    assert info["sha256"] == sha256_fichier(config.MODEL_PATH)


# --- Le modèle ne peut pas être chargé ---------------------------------------------------


def copier_modele(tmp_path):
    modele = tmp_path / "modele.joblib"
    metadata = tmp_path / "metadata.json"
    shutil.copy(config.MODEL_PATH, modele)
    shutil.copy(config.METADATA_PATH, metadata)
    return modele, metadata


def test_chargement_normal(tmp_path):
    s = ServicePrediction(*copier_modele(tmp_path))
    s.charger()
    assert s.pret


def test_fichier_modele_absent(tmp_path):
    modele, metadata = copier_modele(tmp_path)
    modele.unlink()
    with pytest.raises(ModeleIndisponible, match="absent"):
        ServicePrediction(modele, metadata).charger()


def test_fichier_modele_corrompu(tmp_path):
    modele, metadata = copier_modele(tmp_path)
    contenu = bytearray(modele.read_bytes())
    contenu[len(contenu) // 2] ^= 0xFF
    modele.write_bytes(bytes(contenu))
    s = ServicePrediction(modele, metadata)
    with pytest.raises(ModeleIndisponible, match="SHA-256"):
        s.charger()
    assert not s.pret


def test_metadonnees_absentes(tmp_path):
    modele, metadata = copier_modele(tmp_path)
    metadata.unlink()
    with pytest.raises(ModeleIndisponible, match="métadonnées"):
        ServicePrediction(modele, metadata).charger()


def test_objet_qui_n_est_pas_un_classifieur(tmp_path):
    modele, metadata = copier_modele(tmp_path)
    infos = json.loads(metadata.read_text(encoding="utf-8"))
    joblib.dump({"pas": "un modèle"}, modele)
    infos["sha256"] = sha256_fichier(modele)
    metadata.write_text(json.dumps(infos), encoding="utf-8")
    with pytest.raises(ModeleIndisponible, match="classifieur"):
        ServicePrediction(modele, metadata).charger()


def test_variables_du_modele_differentes_des_metadonnees(tmp_path):
    modele, metadata = copier_modele(tmp_path)
    infos = json.loads(metadata.read_text(encoding="utf-8"))
    infos["variables_modele"] = VARIABLES_MODELE[:-1]
    metadata.write_text(json.dumps(infos), encoding="utf-8")
    with pytest.raises(ModeleIndisponible, match="variables"):
        ServicePrediction(modele, metadata).charger()


def test_api_demarre_sans_modele(tmp_path, demande_valide, remettre_modele, caplog):
    """Si le modèle manque au démarrage, l'API démarre quand même et le signale."""
    service.model_path = tmp_path / "absent.joblib"
    with caplog.at_level(logging.ERROR, logger="creditrisk"):
        with TestClient(app) as c:
            assert c.get("/health").status_code == 503
            assert c.post("/predict", json=demande_valide).status_code == 503
            assert c.get("/model-info").status_code == 503
    assert "modèle non chargé" in caplog.text


# --- Une variable attendue est absente / le prétraitement échoue --------------------


def test_variable_absente_au_pretraitement(demande_valide):
    donnees = pd.DataFrame([demande_valide]).drop(columns=["montant_credit"])
    with pytest.raises(ErreurPretraitement, match="montant_credit"):
        preparer_variables(donnees)


@pytest.mark.parametrize(
    "modification, message",
    [
        ({"revenu_mensuel": np.nan}, "non finies"),
        ({"duree_mois": 0}, "strictement positifs"),
        ({"age": "trente"}, "non numériques"),
    ],
)
def test_pretraitement_refuse_des_donnees_inexploitables(demande_valide, modification, message):
    with pytest.raises(ErreurPretraitement, match=message):
        preparer_variables(pd.DataFrame([{**demande_valide, **modification}]))


def test_pretraitement_sans_ligne():
    with pytest.raises(ErreurPretraitement, match="aucune ligne"):
        preparer_variables(pd.DataFrame(columns=config.FEATURES))


def verifier_erreur_500_propre(reponse, *textes_internes):
    assert reponse.status_code == 500
    corps = reponse.json()
    assert corps["erreur"] == "erreur_interne"
    assert corps["message"] == "La prédiction n'a pas pu être calculée."
    assert len(corps["id_requete"]) == 12
    assert "Traceback" not in reponse.text
    for texte in textes_internes:
        assert texte not in reponse.text
    return corps["id_requete"]


def test_erreur_de_pretraitement_dans_l_api(client, demande_valide, monkeypatch, caplog):
    def pretraitement_en_panne(_df):
        raise ErreurPretraitement("détail interne du prétraitement")

    monkeypatch.setattr("app.prediction.preparer_variables", pretraitement_en_panne)
    with caplog.at_level(logging.ERROR, logger="creditrisk"):
        r = client.post("/predict", json=demande_valide)
    id_requete = verifier_erreur_500_propre(r, "détail interne", "ErreurPretraitement")
    # la trace complète est côté serveur, retrouvable grâce à l'identifiant
    assert id_requete in caplog.text
    assert "Traceback" in caplog.text


# --- Dimensions incompatibles avec le modèle -------------------------------------------


def test_dimensions_incompatibles_detectees(demande_valide):
    variables = preparer_variables(pd.DataFrame([demande_valide]))
    with pytest.raises(ErreurPrediction, match="5 variables reçues, 6 attendues"):
        service.predire_proba(variables.drop(columns=["taux_effort"]))
    with pytest.raises(ErreurPrediction, match="dimensions"):
        service.predire_proba(variables[list(reversed(VARIABLES_MODELE))])


def test_dimensions_incompatibles_dans_l_api(client, demande_valide, monkeypatch):
    original = preparer_variables
    monkeypatch.setattr(
        "app.prediction.preparer_variables", lambda df: original(df).drop(columns=["age"])
    )
    verifier_erreur_500_propre(client.post("/predict", json=demande_valide), "dimensions")


# --- Erreur interne pendant la prédiction -----------------------------------------------


class FauxModele:
    def __init__(self, sortie=None, exception=None):
        self.sortie = sortie
        self.exception = exception

    def predict_proba(self, x):
        if self.exception:
            raise self.exception
        return np.asarray(self.sortie)


@pytest.mark.parametrize(
    "faux",
    [
        FauxModele(exception=RuntimeError("mot de passe base = secret")),
        FauxModele(sortie=[[0.5, 1.7]]),  # probabilité > 1
        FauxModele(sortie=[[np.nan, np.nan]]),  # valeur non finie
        FauxModele(sortie=[[0.2, 0.3, 0.5]]),  # trois classes au lieu de deux
    ],
    ids=["exception", "proba_superieure_a_1", "nan", "mauvaise_forme"],
)
def test_erreur_interne_du_modele(client, demande_valide, monkeypatch, faux):
    monkeypatch.setattr(service, "modele", faux)
    verifier_erreur_500_propre(client.post("/predict", json=demande_valide), "secret", "1.7")


def test_journaux_sans_donnees_du_demandeur(client, monkeypatch, caplog):
    demande = {
        "age": 41,
        "revenu_mensuel": 987654,
        "montant_credit": 3456789,
        "duree_mois": 24,
        "anciennete_activite": 7,
    }
    monkeypatch.setattr(service, "modele", FauxModele(exception=RuntimeError("panne")))
    with caplog.at_level(logging.DEBUG):
        client.post("/predict", json=demande)
    assert "987654" not in caplog.text
    assert "3456789" not in caplog.text


# --- Sauvegarde et rechargement -------------------------------------------------------------


def test_sauvegarde_puis_rechargement_identiques(tmp_path, demande_valide):
    infos = json.loads(config.METADATA_PATH.read_text(encoding="utf-8"))
    infos.pop("sha256")
    meta = sauvegarder(service.modele, infos, tmp_path / "m.joblib", tmp_path / "m.json")
    recharge = ServicePrediction(tmp_path / "m.joblib", tmp_path / "m.json")
    recharge.charger()

    cas = json.loads(config.REFERENCE_CASES_PATH.read_text(encoding="utf-8"))
    variables = preparer_variables(pd.DataFrame([c["entree"] for c in cas]))
    assert np.array_equal(service.predire_proba(variables), recharge.predire_proba(variables))
    assert meta["sha256"] == sha256_fichier(tmp_path / "m.joblib")
