"""Tests de robustesse (PARTIE IV) : entrées limites et comportements inattendus.

Règle : une entrée INVALIDE est rejetée (422) ; une entrée INHABITUELLE MAIS VALIDE
est traitée (200) avec une alerte. Dans aucun cas l'API ne doit répondre 500.
"""

import json
import math

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from app import config
from app.main import service
from app.preprocessing import preparer_variables

NE_JAMAIS_REPONDRE = {500, 502, 503}


def envoyer_brut(client, texte):
    return client.post("/predict", content=texte, headers={"content-type": "application/json"})


# --- Valeurs manquantes -------------------------------------------------------------


@pytest.mark.parametrize("champ", config.FEATURES)
def test_valeur_nulle_rejetee(client, demande_valide, champ):
    r = client.post("/predict", json={**demande_valide, champ: None})
    assert r.status_code == 422
    assert r.json()["details"][0]["champ"] == champ


def test_objet_vide(client):
    r = client.post("/predict", json={})
    assert r.status_code == 422
    assert {d["champ"] for d in r.json()["details"]} == set(config.FEATURES)


# --- Valeurs non numériques ----------------------------------------------------------


@pytest.mark.parametrize("valeur", ["abc", "", "12", "1e5", [1], {"a": 1}, True, False])
@pytest.mark.parametrize("champ", config.FEATURES)
def test_valeur_non_numerique_rejetee(client, demande_valide, champ, valeur):
    r = client.post("/predict", json={**demande_valide, champ: valeur})
    assert r.status_code == 422


# --- Valeurs infinies ou non finies (JSON brut, car json= refuse NaN) ------------------


@pytest.mark.parametrize("litteral", ["NaN", "Infinity", "-Infinity", "1e400"])
@pytest.mark.parametrize("champ", ["revenu_mensuel", "montant_credit", "anciennete_activite"])
def test_valeur_non_finie_rejetee(client, demande_valide, champ, litteral):
    texte = json.dumps(demande_valide).replace(
        f'"{champ}": {demande_valide[champ]}', f'"{champ}": {litteral}'
    )
    assert litteral in texte
    r = envoyer_brut(client, texte)
    assert r.status_code == 422
    assert r.json()["details"][0]["champ"] == champ


# --- Valeurs aux limites des intervalles -------------------------------------------------

LIMITES = [
    ("age", config.AGE_MIN, config.AGE_MIN - 1),
    ("age", config.AGE_MAX, config.AGE_MAX + 1),
    ("duree_mois", config.DUREE_MIN, config.DUREE_MIN - 1),
    ("duree_mois", config.DUREE_MAX, config.DUREE_MAX + 1),
    ("revenu_mensuel", config.REVENU_MAX, config.REVENU_MAX + 0.01),
    ("montant_credit", config.MONTANT_MAX, config.MONTANT_MAX + 0.01),
    ("revenu_mensuel", 0.01, 0),
    ("anciennete_activite", 0, -0.01),
]


@pytest.mark.parametrize("champ, accepte, refuse", LIMITES)
def test_limites_des_intervalles(client, demande_valide, champ, accepte, refuse):
    base = {**demande_valide, "anciennete_activite": 0}
    ok = client.post("/predict", json={**base, champ: accepte})
    ko = client.post("/predict", json={**base, champ: refuse})
    assert ok.status_code == 200
    assert 0 <= ok.json()["probabilite_defaut"] <= 1
    assert ko.status_code == 422


# --- Entrées très éloignées des données d'entraînement (valides) ----------------------------


@pytest.mark.parametrize(
    "changements",
    [
        {"revenu_mensuel": 45_000_000},
        {"montant_credit": 49_000_000, "duree_mois": 60},
        {"revenu_mensuel": 1},  # 1 FCFA : valide selon le contrat, mais absurde
        {"age": 75, "anciennete_activite": 55},
    ],
)
def test_entree_atypique_traitee_avec_alerte(client, demande_valide, changements):
    r = client.post("/predict", json={**demande_valide, **changements})
    assert r.status_code == 200
    resultat = r.json()
    assert math.isfinite(resultat["probabilite_defaut"])
    assert 0 <= resultat["probabilite_defaut"] <= 1
    assert resultat["alertes"], "une entrée atypique doit produire au moins une alerte"


# --- Schéma incorrect -------------------------------------------------------------------


@pytest.mark.parametrize(
    "corps",
    [
        [{"age": 35}],
        {"demande": {"age": 35, "revenu_mensuel": 250000}},
        {"Age": 35, "Revenu_Mensuel": 250000, "Montant_Credit": 1000000, "Duree_Mois": 12},
        {
            "age": "35",
            "revenu_mensuel": "250000",
            "montant_credit": "1000000",
            "duree_mois": "12",
            "anciennete_activite": "4",
        },
    ],
    ids=["liste", "objet_imbrique", "noms_en_majuscules", "tout_en_texte"],
)
def test_schema_incorrect(client, corps):
    r = client.post("/predict", json=corps)
    assert r.status_code == 422
    assert r.json()["erreur"] == "donnees_invalides"


# --- Tests générés (Hypothesis) ---------------------------------------------------------

demandes_valides = st.integers(config.AGE_MIN, config.AGE_MAX).flatmap(
    lambda age: st.fixed_dictionaries(
        {
            "age": st.just(age),
            "revenu_mensuel": st.floats(1, config.REVENU_MAX),
            "montant_credit": st.floats(1, config.MONTANT_MAX),
            "duree_mois": st.integers(config.DUREE_MIN, config.DUREE_MAX),
            "anciennete_activite": st.floats(0, age - config.AGE_DEBUT_ACTIVITE),
        }
    )
)

valeurs_json = st.recursive(
    st.none()
    | st.booleans()
    | st.integers()
    | st.floats(allow_nan=False, allow_infinity=False)
    | st.text(max_size=8),
    lambda enfants: st.lists(enfants, max_size=3) | st.dictionaries(st.text(max_size=5), enfants),
    max_leaves=6,
)
demandes_quelconques = st.dictionaries(
    st.sampled_from([*config.FEATURES, "autre"]), valeurs_json, max_size=6
)

REGLAGES = settings(
    max_examples=150, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)


@REGLAGES
@given(demande=demandes_valides)
def test_toute_demande_valide_donne_une_probabilite(client, demande):
    r = client.post("/predict", json=demande)
    assert r.status_code == 200
    p = r.json()["probabilite_defaut"]
    assert math.isfinite(p) and 0 <= p <= 1


@REGLAGES
@given(demande=demandes_quelconques)
def test_aucune_entree_ne_provoque_d_erreur_serveur(client, demande):
    r = client.post("/predict", json=demande)
    assert r.status_code not in NE_JAMAIS_REPONDRE
    assert r.status_code in (200, 422)


# --- Sensibilité : sens de variation et petites perturbations --------------------------


def probas(demandes):
    return service.predire_proba(preparer_variables(pd.DataFrame(demandes)))


def test_emprunter_plus_n_abaisse_pas_le_risque(demande_valide):
    montants = np.linspace(100_000, 6_000_000, 30)
    p = probas([{**demande_valide, "montant_credit": m} for m in montants])
    assert np.all(np.diff(p) >= 0)


@pytest.mark.xfail(
    strict=True,
    reason="AN-02 : au-delà d'environ 2,4 M FCFA, le risque remonte légèrement avec le revenu "
    "(coefficient de log_revenu positif). Anomalie connue, non corrigée en v1.0.0.",
)
def test_gagner_plus_n_augmente_pas_le_risque(demande_valide):
    revenus = np.linspace(50_000, 3_000_000, 30)
    p = probas([{**demande_valide, "revenu_mensuel": r} for r in revenus])
    assert np.all(np.diff(p) <= 0)


def test_petite_variation_du_revenu(demande_valide):
    """±1 % de revenu (erreur d'arrondi, prime ponctuelle) : variation mesurée et bornée.

    On ne demande pas que la classe reste identique (un dossier proche du seuil peut
    basculer), mais que la probabilité ne saute pas.
    """
    cas = json.loads(config.REFERENCE_CASES_PATH.read_text(encoding="utf-8"))
    entrees = [c["entree"] for c in cas]
    p0 = probas(entrees)
    for facteur in (0.99, 1.01):
        p1 = probas([{**e, "revenu_mensuel": e["revenu_mensuel"] * facteur} for e in entrees])
        assert np.max(np.abs(p1 - p0)) < 0.02
