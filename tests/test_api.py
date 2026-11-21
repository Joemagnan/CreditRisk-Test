"""Tests fonctionnels de l'API (PARTIE II) : les 10 scénarios obligatoires et le contrat.

Pour chaque scénario : code HTTP attendu, contenu de la réponse et comportement
(une requête rejetée ne doit jamais atteindre le modèle).
"""

import json

import pytest

from app import config
from app.main import service

CHAMPS_REPONSE = {
    "probabilite_defaut",
    "niveau_risque",
    "seuil_decision",
    "version_modele",
    "alertes",
}


def champs_en_erreur(reponse):
    return {d["champ"] for d in reponse.json()["details"]}


# --- Endpoints de service ---------------------------------------------------------


def test_health_modele_charge(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"statut": "ok", "modele_charge": True, "version_modele": "1.0.0"}


def test_model_info(client):
    r = client.get("/model-info")
    assert r.status_code == 200
    info = r.json()
    assert info["version"] == config.MODEL_VERSION
    assert info["variables_entree"] == config.FEATURES
    assert 0 < info["seuil"] < 1
    assert "SYNTHÉTIQUES" in info["donnees"]
    assert len(info["sha256"]) == 64


# --- Scénario 1 : requête valide -----------------------------------------------------


def test_requete_valide(client, demande_valide, appels_modele):
    r = client.post("/predict", json=demande_valide)
    assert r.status_code == 200
    resultat = r.json()
    assert set(resultat) == CHAMPS_REPONSE
    assert 0 <= resultat["probabilite_defaut"] <= 1
    assert resultat["niveau_risque"] in ("faible", "eleve")
    attendu = "eleve" if resultat["probabilite_defaut"] >= resultat["seuil_decision"] else "faible"
    assert resultat["niveau_risque"] == attendu
    assert resultat["version_modele"] == "1.0.0"
    assert resultat["alertes"] == []
    assert appels_modele["n"] == 1


def test_meme_requete_meme_reponse(client, demande_valide):
    reponses = [client.post("/predict", json=demande_valide).json() for _ in range(3)]
    assert reponses[0] == reponses[1] == reponses[2]


# --- Scénarios 2 à 8 : entrées invalides -> 422, sans appel au modèle -------------------


@pytest.mark.parametrize(
    "scenario, changements, champ",
    [
        ("âge négatif", {"age": -5}, "age"),
        ("revenu en texte", {"revenu_mensuel": "deux cent mille"}, "revenu_mensuel"),
        ("revenu nombre écrit en texte", {"revenu_mensuel": "250000"}, "revenu_mensuel"),
        ("montant nul", {"montant_credit": 0}, "montant_credit"),
        ("montant négatif", {"montant_credit": -500000}, "montant_credit"),
        ("durée non entière", {"duree_mois": 12.5}, "duree_mois"),
        ("revenu extrêmement élevé", {"revenu_mensuel": 1e12}, "revenu_mensuel"),
        ("montant extrêmement élevé", {"montant_credit": 1e15}, "montant_credit"),
        ("âge extrêmement élevé", {"age": 999}, "age"),
    ],
)
def test_entree_invalide_rejetee(
    client, demande_valide, appels_modele, scenario, changements, champ
):
    r = client.post("/predict", json={**demande_valide, **changements})
    assert r.status_code == 422, scenario
    corps = r.json()
    assert corps["erreur"] == "donnees_invalides"
    assert champs_en_erreur(r) == {champ}
    assert appels_modele["n"] == 0


def test_champ_obligatoire_absent(client, demande_valide, appels_modele):
    demande_valide.pop("revenu_mensuel")
    r = client.post("/predict", json=demande_valide)
    assert r.status_code == 422
    assert r.json()["details"] == [
        {"champ": "revenu_mensuel", "message": "champ obligatoire absent"}
    ]
    assert appels_modele["n"] == 0


def test_champ_supplementaire_non_autorise(client, demande_valide, appels_modele):
    # un attribut sensible ne doit pas pouvoir être glissé dans la requête
    r = client.post("/predict", json={**demande_valide, "sexe": "F"})
    assert r.status_code == 422
    assert r.json()["details"] == [{"champ": "sexe", "message": "champ non autorisé"}]
    assert appels_modele["n"] == 0


def test_valeurs_tres_elevees_mais_autorisees(client, demande_valide):
    """Dans les bornes du contrat mais loin des données d'entraînement : 200 + alerte."""
    r = client.post("/predict", json={**demande_valide, "revenu_mensuel": 40_000_000})
    assert r.status_code == 200
    codes = [a["code"] for a in r.json()["alertes"]]
    assert "hors_plage_entrainement" in codes


# --- Scénario 9 : modèle indisponible -----------------------------------------------


def test_modele_indisponible(client, demande_valide, monkeypatch):
    monkeypatch.setattr(service, "modele", None)
    r = client.post("/predict", json=demande_valide)
    assert r.status_code == 503
    assert r.json()["erreur"] == "modele_indisponible"
    assert client.get("/health").status_code == 503
    assert client.get("/health").json()["modele_charge"] is False
    assert client.get("/model-info").status_code == 503


# --- Scénario 10 : requêtes mal formées -------------------------------------------------


@pytest.mark.parametrize(
    "corps",
    ['{"age": 35, "revenu_mensuel": ', "{age: 35}", "pas du json", '{"age": 35,}'],
)
def test_json_mal_forme(client, appels_modele, corps):
    r = client.post("/predict", content=corps, headers={"content-type": "application/json"})
    assert r.status_code == 400
    assert r.json()["erreur"] == "requete_mal_formee"
    assert appels_modele["n"] == 0


@pytest.mark.parametrize("corps", ["[1, 2, 3]", '"texte"', "42", "null"])
def test_corps_qui_n_est_pas_un_objet(client, corps):
    r = client.post("/predict", content=corps, headers={"content-type": "application/json"})
    assert r.status_code == 422
    assert r.json()["erreur"] == "donnees_invalides"


def test_corps_vide(client):
    r = client.post("/predict", content="", headers={"content-type": "application/json"})
    assert r.status_code == 422
    assert r.json()["details"][0]["champ"] == "corps"


def test_methode_non_autorisee(client):
    assert client.get("/predict").status_code == 405


def test_erreurs_toujours_en_json(client, demande_valide):
    r = client.post("/predict", json={**demande_valide, "age": "trente"})
    assert r.headers["content-type"].startswith("application/json")
    json.loads(r.text)
