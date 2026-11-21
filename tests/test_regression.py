"""Tests de non-régression (PARTIE V).

La référence (reports/reference_v1.0.0.json) a été figée une fois, avec la v1.0.0.
Elle n'est jamais recalculée par les tests : si le code ou le modèle change, ce sont
les sorties actuelles qui sont comparées à elle.
"""

import json

import numpy as np
import pandas as pd
import pytest

from app import config
from app.evaluation import comparer_a_reference, metriques
from app.main import service
from app.prediction import sha256_fichier
from app.preprocessing import preparer_variables
from app.training import decouper, entrainer_version

REFERENCE = json.loads(config.REFERENCE_PATH.read_text(encoding="utf-8"))
CAS = json.loads(config.REFERENCE_CASES_PATH.read_text(encoding="utf-8"))
ENTREES = [c["entree"] for c in CAS]


@pytest.fixture(scope="module")
def donnees():
    return pd.read_csv(config.DATA_PATH)


def sorties_actuelles(modele, seuil, donnees) -> dict:
    """Mesures de la version courante, sur les mêmes jeux que la référence."""
    validation = decouper(donnees)[1]
    p_val = modele.predict_proba(preparer_variables(validation))[:, 1]
    p_cas = modele.predict_proba(preparer_variables(pd.DataFrame(ENTREES)))[:, 1]
    return {
        "metriques": metriques(validation[config.TARGET], p_val, seuil),
        "probas_reference": p_cas.tolist(),
        "contrat_ok": bool(np.isfinite(p_cas).all() and (p_cas >= 0).all() and (p_cas <= 1).all()),
    }


# --- La référence et ce qu'elle protège ---------------------------------------------


def test_le_modele_versionne_est_celui_de_la_reference():
    assert REFERENCE["version"] == config.MODEL_VERSION
    assert REFERENCE["sha256_modele"] == sha256_fichier(config.MODEL_PATH)


def test_les_donnees_n_ont_pas_change():
    assert REFERENCE["sha256_donnees"] == sha256_fichier(config.DATA_PATH)


def test_predictions_identiques_sur_les_cas_de_reference():
    """Toute modification du prétraitement ou du chargement se voit ici."""
    p = service.predire_proba(preparer_variables(pd.DataFrame(ENTREES)))
    attendu = np.array([c["proba_attendue"] for c in CAS])
    ecart = np.abs(p - attendu).max()
    assert ecart <= config.TOLERANCES_REGRESSION["ecart_proba_meme_modele"], f"écart max {ecart}"


def test_metriques_conformes_a_la_reference(donnees):
    actuel = sorties_actuelles(service.modele, service.metadata["seuil"], donnees)
    resultat = comparer_a_reference(REFERENCE, actuel)
    assert resultat["decision"] == "accepter", resultat["motifs"]
    assert actuel["metriques"] == REFERENCE["metriques"]


def test_sorties_de_l_api_conformes_au_contrat(client):
    """Les 206 cas de référence passent par l'API et respectent le contrat de sortie."""
    for entree, cas in zip(ENTREES, CAS, strict=True):
        r = client.post("/predict", json=entree)
        assert r.status_code == 200
        resultat = r.json()
        assert set(resultat) == {
            "probabilite_defaut",
            "niveau_risque",
            "seuil_decision",
            "version_modele",
            "alertes",
        }
        assert resultat["probabilite_defaut"] == pytest.approx(cas["proba_attendue"], abs=1e-6)


def test_reentrainement_reproductible(donnees):
    """Même données, même graine, même code : même modèle (à l'arrondi machine près)."""
    run = entrainer_version(
        donnees, "rejeu", "regression_logistique", C=1.0, class_weight="balanced"
    )
    p = run["modele"].predict_proba(preparer_variables(pd.DataFrame(ENTREES)))[:, 1]
    assert run["metadata"]["seuil"] == REFERENCE["metriques"]["seuil"]
    assert np.abs(p - np.array(REFERENCE["probas_reference"])).max() < 1e-9


# --- La règle de décision détecte vraiment les régressions ------------------------------


def test_une_version_degradee_est_rejetee(donnees):
    """Vraie version candidate entraînée sans pondération des classes : le rappel s'effondre."""
    run = entrainer_version(donnees, "candidate", "regression_logistique", class_weight=None)
    # on garde volontairement le seuil de la référence pour isoler l'effet du modèle
    candidat = sorties_actuelles(run["modele"], REFERENCE["metriques"]["seuil"], donnees)
    resultat = comparer_a_reference(REFERENCE, candidat)
    assert resultat["decision"] == "rejeter"
    assert any("rappel" in m for m in resultat["motifs"])


def test_contrat_casse_entraine_un_rejet():
    candidat = {**REFERENCE, "contrat_ok": False}
    assert comparer_a_reference(REFERENCE, candidat)["decision"] == "rejeter"


def test_baisse_d_auc_seule_entraine_une_investigation():
    m = {**REFERENCE["metriques"], "roc_auc": REFERENCE["metriques"]["roc_auc"] - 0.05}
    resultat = comparer_a_reference(REFERENCE, {**REFERENCE, "metriques": m})
    assert resultat["decision"] == "investiguer"


def test_baisse_dans_la_tolerance_acceptee():
    m = {**REFERENCE["metriques"], "rappel": REFERENCE["metriques"]["rappel"] - 0.01}
    assert comparer_a_reference(REFERENCE, {**REFERENCE, "metriques": m})["decision"] == "accepter"
